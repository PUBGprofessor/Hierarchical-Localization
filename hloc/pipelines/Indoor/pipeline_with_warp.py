"""使用共视区域裁剪完成 Indoor query 图像定位的主流程。

与原始 HLoc pipeline 的主要区别是局部匹配阶段：原流程把完整 query
图像和完整 map 候选图像的 SuperPoint 特征交给 LightGlue；本流程先用
query 的外部估计位姿和 map SfM 3D 点计算每个图像对的共视掩膜，只把
掩膜内的关键点和描述子交给 LightGlue。

完整步骤：
  1. 检查数据集、相机和已有 HLoc 中间结果；
  2. 读取已有 MegaLoc top-k retrieval pairs；
  3. 对每个 query/map 对进行 3D 点可见性筛选和像素共视裁剪；
  4. 删除共享 3D 点为 0 的无关候选；
  5. 对剩余候选执行掩膜局部特征匹配；
  6. 将匹配结果送入原有 localize_sfm，执行 2D-3D 建立和 PnP/RANSAC；
  7. 输出位姿、日志和评估结果。

query 的 ground-truth 位姿只在定位完成后的 ``evaluate`` 中读取，绝不
参与共视裁剪或 PnP。
"""

import argparse
import csv
import gc
import json
import os
import pickle
import sys
from functools import lru_cache
from pathlib import Path

from hloc.utils.io import list_h5_names
import h5py
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from hloc import (
    extract_features,
    localize_sfm,
    logger,
    match_features,
    pairs_from_covisibility,
    pairs_from_retrieval,
    triangulation,
)
from hloc import covis_warp, match_covis_features
from hloc.utils.read_write_model import qvec2rotmat, read_cameras_text, read_images_text
from hloc.utils.parsers import parse_retrieval

ROOT = Path(__file__).resolve().parents[3]


def dataset_path(name):
    """按仓库目录或本机 Datasets 目录查找数据集。"""
    for path in (ROOT / name, ROOT.parent.parent / "Datasets" / name):
        if path.is_dir():
            return path
    return ROOT / name


def model_images(model):
    """读取 COLMAP 文本模型中的图像记录。

    主流程只用 map 模型建立参考图像集合；query 模型的图像位姿记录仅
    在评估阶段读取，用于和预测位姿计算误差。
    """
    path = model / "images.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Expected COLMAP text model: {path}")
    return read_images_text(path)


def model_image_names(model):
    """只读取 COLMAP images.txt 的图像名，避免加载全部 2D 观测数组。

    map 数据库规模较大时，完整调用 ``read_images_text`` 会为每个图像
    创建 xys、point3D_ids 和 namedtuple；主流程这里只需要名字，因此
    使用轻量的两行步进解析。
    """
    names = []
    with (Path(model) / "images.txt").open("r", encoding="utf-8") as handle:
        while True:
            header = handle.readline()
            if not header:
                break
            if not header.strip() or header.lstrip().startswith("#"):
                continue
            fields = header.split()
            if len(fields) < 10:
                raise ValueError(f"Invalid COLMAP image header: {header.rstrip()}")
            names.append(fields[9])
            # 每条图像记录紧跟一行 2D 观测，跳过它而不解析成数组。
            if not handle.readline():
                raise ValueError(f"Missing observations for image {fields[9]}")
    return names


def prepare_queries(model, image_dir, output):
    """生成带逐图相机内参的 HLoc 查询列表，不写入 query 外参。

    该文件不包含 query 外参，因此不会把 query 的真值位姿泄漏给定位器。
    COLMAP 可以为不同查询图像分配不同的 camera_id，因此不能假设只有
    一个共享相机。
    """
    cameras = read_cameras_text(model / "cameras.txt")
    query_model_images = read_images_text(model / "images.txt")
    name_to_camera = {
        image.name: cameras[image.camera_id] for image in query_model_images.values()
    }
    suffixes = {".jpg", ".jpeg", ".png"}
    names = sorted(
        path.relative_to(image_dir).as_posix()
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )
    if not names:
        raise ValueError(f"No query images in {image_dir}")
    lines = []
    for name in names:
        if name not in name_to_camera:
            raise ValueError(
                f"Image {name} is missing from COLMAP query images.txt; "
                "cannot determine its camera intrinsics."
            )
        camera = name_to_camera[name]
        lines.append(
            f"{name} {camera.model} {camera.width} {camera.height} "
            + " ".join(map(str, camera.params))
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return names


def link_features(query_features, map_features, output):
    """Expose both feature files to HLoc without duplicating descriptor data."""
    # HDF5 files may store images below a shared top-level directory such as
    # ``db``.  list_h5_names() returns complete image names like db/1003.jpg;
    # checking only ``h5file.keys()`` would report a false collision on ``db``.
    query_names = set(list_h5_names(query_features))
    map_names = set(list_h5_names(map_features))
    overlap = query_names & map_names
    if overlap:
        raise ValueError(f"Query and map feature names overlap: {sorted(overlap)[:5]}")

    def add_links(combined, source, names):
        for name in sorted(names):
            parent = combined
            parts = name.split("/")
            for part in parts[:-1]:
                parent = parent.require_group(part)
            parent[parts[-1]] = h5py.ExternalLink(str(source.resolve()), name)

    with h5py.File(output, "w") as combined:
        add_links(combined, query_features, query_names)
        add_links(combined, map_features, map_names)



def h5_contains_images(path, image_names):
    """判断特征 H5 是否存在且包含本次所需的全部图像 group。

    只检查文件存在是不够的：上一次提取可能被中断，留下一个非空但不
    完整的 H5。发现缺少 group 时仍调用 ``extract_features.main``，它会
    复用已有 group 并只追加缺少的图像。
    """
    path = Path(path)
    if not path.is_file() or not path.stat().st_size:
        return False
    try:
        with h5py.File(path, "r") as handle:
            # HDF5 group 可能是 db/1003.jpg；不能用 name in handle 判断
            # 根层，因为根层只包含 db。逐级访问才支持带前缀的图像名。
            return all(name in handle for name in image_names)
    except (OSError, ValueError):
        return False


def lookup_by_image_name(table, name):
    """按完整名查找位姿/日志，必要时唯一地回退到 basename。

    HLoc 的 ``write_poses`` 会把 ``db/1003.jpg`` 写成 ``1003.jpg``，而
    Aachen 的 query H5、images.txt 和 reloc3r 文件保留了 ``db/`` 前缀。
    先精确匹配，再在 basename 唯一时回退，兼容两种格式并避免同名歧义。
    返回 ``(匹配键, 值)``，找不到时返回 ``(None, None)``。
    """
    if name in table:
        return name, table[name]
    basename = Path(name).name
    candidates = [(key, value) for key, value in table.items() if Path(key).name == basename]
    if len(candidates) == 1:
        return candidates[0]
    return None, None


def read_pose_table(path):
    """读取 COLMAP 风格位姿文本，返回 ``图像名 -> 7 个位姿数值``。

    reloc3r 文件带有 ``#`` 注释头，而 localize_sfm 输出通常没有注释；
    因此这里统一跳过空行和注释行，只接受 ``name qw qx qy qz tx ty tz``。
    """
    poses = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        if len(fields) != 8:
            raise ValueError(f"Invalid pose line in {path}: {line}")
        poses[fields[0]] = np.asarray([float(value) for value in fields[1:]], dtype=float)
    return poses


def rotation_error_deg(predicted_qvec, estimated_qvec):
    """计算两个 COLMAP w2c 旋转之间的最小夹角（单位：度）。"""
    predicted_rotation = qvec2rotmat(predicted_qvec)
    estimated_rotation = qvec2rotmat(estimated_qvec)
    relative = predicted_rotation @ estimated_rotation.T
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def apply_estimated_pose_fallback(
    results, estimated_poses, output_dir, threshold_deg=25.0
):
    """按 PnP 是否成功和旋转误差替换明显错误的 HLoc 位姿。

    ``localize_sfm`` 在 PnP 失败时会把 top1 map 图像的位姿写入结果，
    同时在 ``query_poses.txt_logs.pkl`` 中将 ``PnP_ret`` 置为 ``None``。
    因此这里有两个独立的触发条件：

    1. ``PnP_ret is None``，表示当前结果是 top1 回退位姿；
    2. HLoc 与 reloc3r 估计位姿的旋转误差严格大于 ``threshold_deg``。

    任一条件成立且存在对应的 reloc3r 位姿，就使用 reloc3r 的完整 7
    参数位姿。原始 HLoc 结果另存为 ``query_poses_hloc.txt``。
    """
    predicted = read_pose_table(results)
    estimated = read_pose_table(estimated_poses)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_results = output_dir / "query_poses_hloc.txt"
    raw_results.write_text(Path(results).read_text(encoding="utf-8"), encoding="utf-8")

    logs_path = Path(f"{results}_logs.pkl")
    if not logs_path.is_file():
        raise FileNotFoundError(f"localize_sfm log is required for PnP fallback detection: {logs_path}")
    with logs_path.open("rb") as handle:
        localization_logs = pickle.load(handle).get("loc", {})

    fallback_names = set()
    report = {
        "enabled": True,
        "rotation_threshold_deg": threshold_deg,
        "triggers": ["pnp_failed", "rotation_error_exceeded"],
        "replaced_count": 0,
        "queries": {},
    }
    for name, pose in predicted.items():
        estimated_name, estimated_pose = lookup_by_image_name(estimated, name)
        if estimated_pose is None:
            report["queries"][name] = {
                "status": "no_estimated_pose",
                "estimated_pose_lookup": "exact_or_unique_basename_failed",
            }
            continue
        # localize_sfm 的日志键可能保留 db/ 前缀，也可能和输出位姿一样
        # 只有 basename；对两种键使用相同的兼容查找。
        _, localization_log = lookup_by_image_name(localization_logs, name)
        localization_log = localization_log or {}
        error = rotation_error_deg(pose[:4], estimated_pose[:4])
        # PnP_ret 缺失正是 localize_sfm 使用 top1 位姿兜底的标志。
        pnp_failed = localization_log.get("PnP_ret") is None
        rotation_exceeded = error > threshold_deg
        use_estimated = pnp_failed or rotation_exceeded
        triggers = []
        if pnp_failed:
            triggers.append("pnp_failed")
        if rotation_exceeded:
            triggers.append("rotation_error_exceeded")
        report["queries"][name] = {
            "rotation_error_deg": error,
            "estimated_pose_key": estimated_name,
            "pnp_failed": pnp_failed,
            "triggers": triggers,
            "status": "estimated_pose_fallback" if use_estimated else "hloc",
        }
        if use_estimated:
            # 写出时保留 HLoc 当前使用的图像键（通常是 basename），只
            # 替换 7 个位姿数值；estimated_pose 可能来自 db/ 前缀键。
            predicted[name] = estimated_pose
            fallback_names.add(name)

    # 按原 HLoc 图像顺序重写最终文件，只替换位姿数值，不改图像名。
    output_lines = [
        f"{name} " + " ".join(map(str, pose)) for name, pose in predicted.items()
    ]
    Path(results).write_text("\n".join(output_lines) + ("\n" if output_lines else ""), encoding="utf-8")
    report["replaced_count"] = len(fallback_names)
    (output_dir / "fallback_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    logger.info(
        "Estimated-pose fallback: replaced %d/%d HLoc poses (PnP failure or rotation error > %.1f deg)",
        len(fallback_names), len(predicted), threshold_deg,
    )
    return fallback_names


def evaluate(results, query_model, query_names, output, fallback_names=None):
    """比较预测和真值 w2c 位姿，并分别统计 PnP 与回退结果。"""
    # This is the first read of query images.txt, after localize_sfm has returned.
    fallback_names = fallback_names or set()
    ground_truth = {image.name: image for image in model_images(query_model).values()}
    with open(f"{results}_logs.pkl", "rb") as handle:
        logs = pickle.load(handle)["loc"]
    predictions = {}
    for line in results.read_text(encoding="utf-8").splitlines():
        name, *pose = line.split()
        if len(pose) != 7:
            raise ValueError(f"Invalid pose in {results}: {line}")
        predictions[name] = np.asarray(pose, dtype=float)

    # HLoc writes only the basename (e.g. 1003.jpg), while Aachen query
    # names can include a prefix (e.g. db/1003.jpg).
    basename_to_query = {}
    for name in query_names:
        basename = Path(name).name
        basename_to_query[basename] = (
            name if basename not in basename_to_query else None
        )

    rows = []
    for name in query_names:
        image = ground_truth[name]
        prediction = predictions.get(name)
        if prediction is None:
            matched_name = basename_to_query.get(Path(name).name)
            if matched_name == name:
                prediction = predictions.get(Path(name).name)
        method = "missing"
        translation = rotation = None
        if prediction is not None:
            method = (
                "estimated_pose_fallback"
                if (name in fallback_names or Path(name).name in fallback_names)
                else (
                    "pnp"
                    if (lookup_by_image_name(logs, name)[1] or {}).get("PnP_ret") is not None
                    else "retrieval_fallback"
                )
            )
            gt_rotation = image.qvec2rotmat()
            predicted_rotation = qvec2rotmat(prediction[:4])
            # Translation error compares camera centers, not w2c translation vectors.
            gt_center = -gt_rotation.T @ image.tvec
            predicted_center = -predicted_rotation.T @ prediction[4:]
            translation = float(np.linalg.norm(gt_center - predicted_center))
            cosine = np.clip(
                (np.trace(gt_rotation.T @ predicted_rotation) - 1) / 2, -1, 1
            )
            rotation = float(np.degrees(np.arccos(cosine)))
        rows.append(dict(image=name, method=method, translation_m=translation, rotation_deg=rotation))

    def metrics(items):
        if not items:
            return None
        return {
            "count": len(items),
            "mean_translation_m": float(np.mean([item["translation_m"] for item in items])),
            "median_translation_m": float(np.median([item["translation_m"] for item in items])),
            "mean_rotation_deg": float(np.mean([item["rotation_deg"] for item in items])),
            "median_rotation_deg": float(np.median([item["rotation_deg"] for item in items])),
        }

    estimated = [row for row in rows if row["translation_m"] is not None]
    pnp = [row for row in rows if row["method"] == "pnp"]
    summary = {
        "total_queries": len(rows),
        "pnp_localized": len(pnp),
        "estimated_pose_fallback": len(fallback_names),
        "retrieval_fallback": sum(row["method"] == "retrieval_fallback" for row in rows),
        "missing": len(rows) - len(estimated),
        "all_estimated": metrics(estimated),
        "pnp_only": metrics(pnp),
    }
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Indoor evaluation: %s", summary)


def run(args):
    """复用已有 HLoc 中间结果，按图像对共视区域完成定位。

    新建的配对文件、掩膜匹配文件、位姿和评估报告都写入
    ``args.output_dir``；数据集目录中的原始特征和旧匹配文件只读使用。
    """
    map_dir, query_dir = args.map_dir.resolve(), args.query_dir.resolve()
    map_model, query_model = map_dir / "sparse" / "0", query_dir / "sparse" / "0"
    map_images, query_images = map_dir / "images", query_dir / "images"
    for path in (map_images, query_images, map_model, query_model):
        if not path.is_dir():
            raise FileNotFoundError(path)
    map_output, query_output = map_dir / "hloc_outputs", query_dir / "hloc_outputs"
    final_output = args.output_dir.resolve()
    for path in (map_output, query_output, final_output):
        path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(map_output / "torch_cache"))

    map_names = set(model_image_names(map_model))
    if not map_names:
        raise ValueError(f"No reference images in {map_model}")
    for name in map_names:
        if not (map_images / name).is_file():
            raise FileNotFoundError(map_images / name)
    query_list = final_output / "queries_with_intrinsics.txt"
    # 查询列表也是可复用的中间结果：存在且非空时直接读取，避免每次
    # 运行都重新写文件；缺失时才按照 query 相机内参重新生成。
    if query_list.is_file() and query_list.stat().st_size:
        query_names = [line.split()[0] for line in query_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        query_names = prepare_queries(query_model, query_images, query_list)
    if map_names & set(query_names):
        raise ValueError("Map and query image names must be distinct")

    # 配置中的 output 名称用于确定中间文件路径。下面每个文件都采用
    # “存在则复用、缺失才生成”的策略，生成逻辑与原 pipeline 一致。
    feature_conf = extract_features.confs["superpoint_my"]
    retrieval_conf = extract_features.confs["megaloc"]
    matcher_conf = match_features.confs["superpoint+lightglue"]

    map_local = map_output / f"{feature_conf['output']}.h5"
    map_global = map_output / f"{retrieval_conf['output']}.h5"
    query_local = query_output / f"{feature_conf['output']}.h5"
    query_global = query_output / f"{retrieval_conf['output']}.h5"

    if not h5_contains_images(map_local, map_names):
        logger.info("Map local features missing; extracting SuperPoint features: %s", map_local)
        map_local = extract_features.main(
            feature_conf, map_images, map_output, image_list=sorted(map_names))
    if not h5_contains_images(map_global, map_names):
        logger.info("Map global features missing; extracting MegaLoc features: %s", map_global)
        map_global = extract_features.main(
            retrieval_conf, map_images, map_output, image_list=sorted(map_names))
    if not h5_contains_images(query_local, query_names):
        logger.info("Query local features missing; extracting SuperPoint features: %s", query_local)
        query_local = extract_features.main(
            feature_conf, query_images, query_output, image_list=query_names)
    if not h5_contains_images(query_global, query_names):
        logger.info("Query global features missing; extracting MegaLoc features: %s", query_global)
        query_global = extract_features.main(
            retrieval_conf, query_images, query_output, image_list=query_names)

    reference_sfm = map_output / f"sfm-{feature_conf['output']}-covis{args.num_covis}"
    sfm_pairs = map_output / f"pairs-covis{args.num_covis}.txt"
    if not sfm_pairs.is_file() or not sfm_pairs.stat().st_size:
        logger.info("Map covisibility pairs missing; generating: %s", sfm_pairs)
        pairs_from_covisibility.main(map_model, sfm_pairs, num_matched=args.num_covis)
    if not sfm_pairs.stat().st_size:
        raise ValueError("No covisible map image pairs; cannot triangulate landmarks")

    sfm_matches = map_output / f"{matcher_conf['output']}-covis{args.num_covis}.h5"
    if not sfm_matches.is_file() or not sfm_matches.stat().st_size:
        logger.info("Map covisibility matches missing; matching: %s", sfm_matches)
        match_features.main(matcher_conf, sfm_pairs, map_local, matches=sfm_matches)

    if not (reference_sfm / "images.bin").is_file() or not (reference_sfm / "images.bin").stat().st_size:
        logger.info("Reference SfM model missing; triangulating: %s", reference_sfm)
        triangulation.main(
            reference_sfm, map_model, map_images, sfm_pairs, map_local, sfm_matches)

    loc_pairs = query_output / f"pairs-query-megaloc{args.num_loc}.txt"
    if not loc_pairs.is_file() or not loc_pairs.stat().st_size:
        logger.info("Retrieval pairs missing; running MegaLoc retrieval: %s", loc_pairs)
        pairs_from_retrieval.main(
            query_global, loc_pairs, min(args.num_loc, len(map_names)),
            query_list=query_names, db_descriptors=map_global, db_model=reference_sfm)
    if not loc_pairs.stat().st_size:
        raise ValueError("No retrieval pairs; check the descriptors and map model")

    # 上述分支可能刚刚生成文件，最后统一确认所有下游输入可读。
    for artifact in (map_local, map_global, query_local, query_global,
                     sfm_matches, reference_sfm / "images.bin", loc_pairs):
        if not artifact.exists() or (artifact.is_file() and not artifact.stat().st_size):
            raise FileNotFoundError(f"HLoc artifact was not created: {artifact}")
    # 候选仍来自已有 MegaLoc top-k。共视掩膜使用 reloc3r 估计的 query
    # 位姿与 map 观测到的 3D 点，绝不使用 query 真值位姿。
    estimated_poses = args.estimated_poses.resolve()
    query_cameras = read_cameras_text(query_model / "cameras.txt")
    query_camera_by_name = {
        image.name: query_cameras[image.camera_id]
        for image in model_images(query_model).values()
    }
    # 每个掩膜约为 width*height 字节；不能把全部 top-k pair 的掩膜都
    # 放在字典中。小容量 LRU 只保留最近少量 pair，控制峰值内存。
    @lru_cache(maxsize=4)
    def pair_masks(qname, mname):
        query_camera = query_camera_by_name[qname]
        return covis_warp.compute_pair_masks(
            qname, mname, estimated_poses, map_model, query_camera, margin=args.margin)

    # 第一层几何过滤：零共享 3D 点意味着候选 map 图像与 query 没有
    # 共视支持。该候选既不进入 LightGlue，也不进入后续 PnP。
    filtered_pairs = final_output / f"pairs-query-megaloc{args.num_loc}-covis.txt"
    raw_retrieval = parse_retrieval(loc_pairs)
    retained = {}
    removed = 0
    for qname, candidates in raw_retrieval.items():
        good = []
        for mname in candidates:
            masks = pair_masks(qname, mname)
            if masks is None:
                removed += 1
            else:
                good.append(mname)
        if good:
            retained[qname] = good
    filtered_pairs.write_text(
        "".join(f"{qname} {mname}\n" for qname, candidates in retained.items() for mname in candidates),
        encoding="utf-8")
    logger.info("Covisibility filter: kept %d pairs, rejected %d zero-point pairs; %d queries retain candidates",
                sum(map(len, retained.values())), removed, len(retained))

    # 第二层裁剪在 match_covis_features 中完成：只有每个图像对的共视
    # 掩膜内关键点会送入 LightGlue，匹配索引随后恢复到原始 H5 索引。
    loc_matches = final_output / "matches-covis-megaloc10.h5"
    match_covis_features.main(
        matcher_conf, filtered_pairs, query_local, map_local, loc_matches,
        mask_provider=pair_masks, overwrite=args.overwrite_matches)
    # 共视过滤和匹配阶段结束后不再需要 3D 点投影索引。释放它，避免
    # 与 localize_sfm 即将加载的 pycolmap Reconstruction 同时占用内存。
    covis_warp.clear_model_cache()
    pair_masks.cache_clear()
    gc.collect()
    combined_features = final_output / f"{feature_conf['output']}-combined.h5"
    if not combined_features.exists():
        link_features(query_local, map_local, combined_features)

    results = final_output / "query_poses.txt"
    # PnP uses query intrinsics, query keypoints, map keypoints, and map 3D points.
    # Query ground-truth poses are never an input to this call.
    localize_sfm.main(
        reference_sfm, query_list, filtered_pairs, combined_features, loc_matches, results,
        covisibility_clustering=False,
    )
    # 可选兜底：PnP 失败（localize_sfm 使用 top1 回退位姿），或者 HLoc
    # 与 reloc3r 估计位姿的旋转差超过阈值时，替换最终位姿。默认关闭，
    # 因此不改变原有 HLoc 结果。
    fallback_names = set()
    if args.use_estimated_fallback:
        fallback_names = apply_estimated_pose_fallback(
            results,
            estimated_poses,
            final_output,
            threshold_deg=args.fallback_rotation_threshold,
        )
    evaluate(
        results,
        query_model,
        query_names,
        final_output / "evaluation.json",
        fallback_names=fallback_names,
    )
    logger.info("Poses: %s; evaluation: %s", results, final_output / "evaluation.json")


if __name__ == "__main__":
    # 默认路径对应本次 Indoor 共视定位实验；也可以通过命令行替换数据集
    # 和输出目录。运行时应使用包含 PyTorch/LightGlue 依赖的 dl 环境。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-dir", type=Path, default=dataset_path("Indoor_map_51"))
    parser.add_argument("--query-dir", type=Path, default=dataset_path("Indoor_query_38_38"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "Indoor_127_covis" / "all_pipeline")
    parser.add_argument("--num-covis", type=int, default=20)
    parser.add_argument("--num-loc", type=int, default=10)
    parser.add_argument(
        "--estimated-poses", type=Path,
        default=ROOT / "output/reloc3r_Indoor127_megaloc/query_poses.txt",
        help="query 的估计 COLMAP w2c 位姿文件，不是 query 真值文件",
    )
    parser.add_argument(
        "--margin", type=int, default=50,
        help="投影凸包向外扩展的像素数，补偿位姿和稀疏重建误差",
    )
    parser.add_argument(
        "--overwrite-matches", action="store_true",
        help="重新计算输出目录中已经存在的共视匹配组",
    )
    parser.add_argument(
        "--use-estimated-fallback", action="store_true",
        help="启用兜底：PnP 失败或旋转误差超过阈值时使用估计位姿",
    )
    parser.add_argument(
        "--fallback-rotation-threshold", type=float, default=25.0,
        help="兜底旋转误差阈值（度），默认 25 度",
    )
    arguments = parser.parse_args()
    if arguments.num_covis < 1 or arguments.num_loc < 1:
        parser.error("--num-covis and --num-loc must be positive")
    run(arguments)
