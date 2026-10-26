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
import json
import os
import pickle
import sys
from pathlib import Path

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


def prepare_queries(model, image_dir, output):
    """生成 HLoc 查询列表文件，只写入图像名和相机内参。

    该文件不包含 query 外参，因此不会把 query 的真值位姿泄漏给定位器。
    """
    cameras = read_cameras_text(model / "cameras.txt")
    if len(cameras) != 1:
        raise ValueError("Query cameras.txt must contain one shared camera")
    camera = next(iter(cameras.values()))
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
        lines.append(
            f"{name} {camera.model} {camera.width} {camera.height} "
            + " ".join(map(str, camera.params))
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return names


def link_features(query_features, map_features, output):
    """创建合并特征 H5 的外部链接，不复制或修改原始特征数据。"""
    with h5py.File(query_features, "r") as query_file, h5py.File(map_features, "r") as map_file:
        query_names, map_names = set(query_file), set(map_file)
        if query_names & map_names:
            raise ValueError("Query and map feature names overlap")
        with h5py.File(output, "w") as combined:
            for name in query_names:
                combined[name] = h5py.ExternalLink(str(query_features.resolve()), name)
            for name in map_names:
                combined[name] = h5py.ExternalLink(str(map_features.resolve()), name)


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


def apply_estimated_pose_fallback(results, estimated_poses, output_dir, threshold_deg=20.0):
    """按旋转误差把明显错误的 HLoc 位姿替换为外部估计位姿。

    该机制只比较旋转，不比较平移：当 HLoc 与 reloc3r 的旋转差严格
    大于 ``threshold_deg`` 时，最终 ``query_poses.txt`` 使用 reloc3r
    的完整 7 参数位姿。原始 HLoc 结果另存为 ``query_poses_hloc.txt``，
    便于复盘；返回被替换的图像名集合，供评估结果标记方法。
    """
    predicted = read_pose_table(results)
    estimated = read_pose_table(estimated_poses)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_results = output_dir / "query_poses_hloc.txt"
    raw_results.write_text(Path(results).read_text(encoding="utf-8"), encoding="utf-8")

    fallback_names = set()
    report = {
        "enabled": True,
        "rotation_threshold_deg": threshold_deg,
        "replaced_count": 0,
        "queries": {},
    }
    for name, pose in predicted.items():
        if name not in estimated:
            report["queries"][name] = {"status": "no_estimated_pose"}
            continue
        error = rotation_error_deg(pose[:4], estimated[name][:4])
        use_estimated = error > threshold_deg
        report["queries"][name] = {
            "rotation_error_deg": error,
            "status": "estimated_pose_fallback" if use_estimated else "hloc",
        }
        if use_estimated:
            predicted[name] = estimated[name]
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
        "Estimated-pose fallback: replaced %d/%d HLoc poses when rotation error exceeded %.1f deg",
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

    rows = []
    for name in query_names:
        image = ground_truth[name]
        prediction = predictions.get(name)
        method = "missing"
        translation = rotation = None
        if prediction is not None:
            method = (
                "estimated_pose_fallback" if name in fallback_names
                else ("pnp" if logs.get(name, {}).get("PnP_ret") is not None
                      else "retrieval_fallback")
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

    map_names = {image.name for image in model_images(map_model).values()}
    if not map_names:
        raise ValueError(f"No reference images in {map_model}")
    for name in map_names:
        if not (map_images / name).is_file():
            raise FileNotFoundError(map_images / name)
    query_list = final_output / "queries_with_intrinsics.txt"
    query_names = prepare_queries(query_model, query_images, query_list)
    if map_names & set(query_names):
        raise ValueError("Map and query image names must be distinct")

    # 这里只用配置中的 output 名称定位已有 H5，不重新提取局部特征。
    feature_conf = extract_features.confs["superpoint_my"]
    matcher_conf = match_features.confs["superpoint+lightglue"]
    # 只读复用已有局部特征、三角化 SfM 模型和 MegaLoc 检索结果。缺失时
    # 直接报错，避免主流程悄悄覆盖数据集目录中的旧中间结果。
    map_local = map_output / f"{feature_conf['output']}.h5"
    query_local = query_output / f"{feature_conf['output']}.h5"
    reference_sfm = map_output / f"sfm-{feature_conf['output']}-covis{args.num_covis}"
    loc_pairs = query_output / f"pairs-query-megaloc{args.num_loc}.txt"
    for artifact in (map_local, query_local, reference_sfm / "images.bin", loc_pairs):
        if not artifact.exists():
            raise FileNotFoundError(f"Required existing HLoc artifact: {artifact}")
    if not loc_pairs.stat().st_size:
        raise ValueError("No retrieval pairs; check the descriptors and map model")
    # 候选仍来自已有 MegaLoc top-k。共视掩膜使用 reloc3r 估计的 query
    # 位姿与 map 观测到的 3D 点，绝不使用 query 真值位姿。
    estimated_poses = args.estimated_poses.resolve()
    query_camera = next(iter(read_cameras_text(query_model / "cameras.txt").values()))
    mask_cache = {}

    def pair_masks(qname, mname):
        key = (qname, mname)
        if key not in mask_cache:
            mask_cache[key] = covis_warp.compute_pair_masks(
                qname, mname, estimated_poses, map_model, query_camera, margin=args.margin)
        return mask_cache[key]

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
    # 可选兜底：只在 HLoc 位姿与 reloc3r 估计位姿的旋转差超过阈值时
    # 替换最终位姿。默认关闭，因此不改变原有 HLoc 结果。
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
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "Indoor_127_covis" / "alll_pipeline")
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
        help="启用旋转误差兜底：HLoc 与估计位姿差异超过阈值时使用估计位姿",
    )
    parser.add_argument(
        "--fallback-rotation-threshold", type=float, default=20.0,
        help="兜底旋转误差阈值（度），默认 20 度",
    )
    arguments = parser.parse_args()
    if arguments.num_covis < 1 or arguments.num_loc < 1:
        parser.error("--num-covis and --num-loc must be positive")
    run(arguments)
