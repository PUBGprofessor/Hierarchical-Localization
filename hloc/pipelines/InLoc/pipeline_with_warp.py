"""使用初始查询位姿和 InLoc XYZcut 深度进行共视裁剪后定位。

与 Indoor 版本的区别是 InLoc 没有 map COLMAP 重建模型。本流程从每个数据库
cutout 对应的 ``.mat`` 文件读取 ``XYZcut``，将有效像素三维坐标通过扫描对齐矩阵
变换到全局坐标，再投影到已有 query 初始位姿对应的相机平面，构造 query/map
共视掩膜。裁剪只作用于局部特征匹配；最终仍调用原 ``localize_inloc``，用扫描
深度和 PnP/RANSAC 估计位姿。

运行示例（在 conda dl 环境、仓库根目录执行）：
    python -m hloc.pipelines.InLoc.pipeline_with_warp
"""

import argparse
import hashlib
import json
import pickle
from functools import lru_cache
from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

from ... import extract_features, localize_inloc, logger, match_features
from ...match_covis_features import main as match_covis_features
from ...utils.parsers import names_to_pair, parse_retrieval
from ...utils.read_write_model import qvec2rotmat, rotmat2qvec
from ...localize_inloc import get_scan_pose


def read_initial_poses(path):
    """读取 query 初始位姿文件，返回 ``basename -> (R, t)`` 的 world-to-camera 表。

    文件每行至少包含 ``image_name qw qx qy qz tx ty tz``，四元数采用 COLMAP
    的 wxyz 顺序。允许位姿后附加检索统计字段（例如 reloc3r 输出的
    ``retained_candidates``），这些字段不参与位姿计算。按 basename 建索引，
    以兼容 InLoc 位姿文件只写图像名的格式。
    """
    poses = {}
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        if len(fields) < 8:
            raise ValueError(f"Invalid pose at {path}:{line_number}: {line}")
        try:
            qvec = np.asarray([float(value) for value in fields[1:5]], dtype=np.float64)
            tvec = np.asarray([float(value) for value in fields[5:8]], dtype=np.float64)
        except ValueError as error:
            raise ValueError(
                f"Invalid pose values at {path}:{line_number}: {line}"
            ) from error
        name = Path(fields[0].replace("\\", "/")).name
        if name in poses:
            raise ValueError(f"Duplicate query basename {name!r} in {path}")
        poses[name] = (qvec2rotmat(qvec), tvec)
    if not poses:
        raise ValueError(f"No query poses found in {path}")
    return poses


def merge_match_files(source_paths, output_path):
    """将若干 HLoc matches HDF5 中的配对组汇总到一个定位输入文件。"""
    with h5py.File(output_path, "a", libver="latest") as output:
        for source_path in source_paths:
            if not Path(source_path).is_file():
                continue
            with h5py.File(source_path, "r", libver="latest") as source:
                for name in source:
                    if name in output:
                        del output[name]
                    source.copy(name, output)


def has_complete_match_groups(pairs_path, matches_path):
    """确认匹配 HDF5 覆盖 pairs 文件中的所有图像对，可安全跳过匹配步骤。"""
    pairs = parse_retrieval(pairs_path)
    pair_keys = [
        names_to_pair(query_name, map_name)
        for query_name, map_names in pairs.items()
        for map_name in map_names
    ]
    if not pair_keys or not Path(matches_path).is_file():
        return False, len(pair_keys), 0
    try:
        with h5py.File(matches_path, "r", libver="latest") as matches:
            missing = sum(pair_key not in matches for pair_key in pair_keys)
    except OSError:
        return False, len(pair_keys), len(pair_keys)
    return missing == 0, len(pair_keys), missing


def read_pose_table(path):
    """读取位姿文本为 ``图像 basename -> [qw,qx,qy,qz,tx,ty,tz]``。"""
    poses = {}
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        if len(fields) < 8:
            raise ValueError(f"Invalid pose at {path}:{line_number}: {line}")
        name = Path(fields[0].replace("\\", "/")).name
        poses[name] = np.asarray([float(value) for value in fields[1:8]], dtype=np.float64)
    return poses


def rotation_error_deg(qvec_a, qvec_b):
    """计算两个 COLMAP w2c 四元数旋转之间的最小夹角（度）。"""
    rotation_a = qvec2rotmat(qvec_a)
    rotation_b = qvec2rotmat(qvec_b)
    relative = rotation_a @ rotation_b.T
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def apply_estimated_pose_fallback(results, initial_poses, threshold_deg, output_dir):
    """PnP 失败或旋转误差超阈值时，用初始估计位姿替换并输出报告。

    原始位姿另存为 ``<results stem>_pnp.txt``；最终结果保留成功 PnP 位姿，
    对触发回退且有初始位姿的 query 写入初始位姿。失败 query 若也没有初始
    位姿则继续缺省，并在报告中标明。
    """
    results = Path(results)
    output_dir = Path(output_dir)
    raw_results = output_dir / f"{results.stem}_pnp{results.suffix}"
    raw_results.write_text(results.read_text(encoding="utf-8"), encoding="utf-8")
    predicted = read_pose_table(results)
    logs_path = Path(f"{results}_logs.pkl")
    with logs_path.open("rb") as handle:
        localization_logs = pickle.load(handle).get("loc", {})

    report = {
        "enabled": True,
        "rotation_threshold_deg": threshold_deg,
        "triggers": ["pnp_failed", "rotation_error_exceeded"],
        "replaced_count": 0,
        "queries": {},
    }
    fallback_names = set()
    for query_name, localization in localization_logs.items():
        name = Path(query_name.replace("\\", "/")).name
        initial = initial_poses.get(name)
        pose = predicted.get(name)
        pnp_failed = localization.get("status") != "localized" or pose is None
        if initial is None:
            report["queries"][query_name] = {
                "status": "no_initial_pose",
                "pnp_failed": pnp_failed,
                "triggers": ["pnp_failed"] if pnp_failed else [],
            }
            continue

        initial_qvec = rotmat2qvec(initial[0])
        error = rotation_error_deg(pose[:4], initial_qvec) if pose is not None else None
        rotation_exceeded = error is not None and error > threshold_deg
        triggers = []
        if pnp_failed:
            triggers.append("pnp_failed")
        if rotation_exceeded:
            triggers.append("rotation_error_exceeded")
        use_initial = bool(triggers)
        report["queries"][query_name] = {
            "rotation_error_deg": error,
            "pnp_failed": pnp_failed,
            "triggers": triggers,
            "status": "initial_pose_fallback" if use_initial else "pnp",
        }
        if use_initial:
            predicted[name] = np.concatenate((initial_qvec, initial[1]))
            fallback_names.add(name)

    ordered_names = []
    for query_name in localization_logs:
        name = Path(query_name.replace("\\", "/")).name
        if name in predicted and name not in ordered_names:
            ordered_names.append(name)
    ordered_names.extend(name for name in predicted if name not in ordered_names)
    lines = [f"{name} " + " ".join(map(str, predicted[name])) for name in ordered_names]
    results.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    report["replaced_count"] = len(fallback_names)
    (output_dir / "fallback_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.info(
        "Initial-pose fallback replaced %d poses; raw PnP poses: %s",
        len(fallback_names),
        raw_results,
    )
    return fallback_names


def make_hull_mask(points, width, height, margin):
    """將共視像素點轉成外擴凸包布爾掩膜；點不足時返回全空掩膜。"""
    points = np.asarray(points)
    if len(points) < 3:
        return np.zeros((height, width), dtype=bool)
    points = np.unique(np.round(points).astype(np.int32), axis=0)
    if len(points) < 3:
        return np.zeros((height, width), dtype=bool)
    hull = cv2.convexHull(points.reshape(-1, 1, 2))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 1)
    if margin > 0:
        size = 2 * margin + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        mask = cv2.dilate(mask, kernel)
    return mask.astype(bool)


@lru_cache(maxsize=8)
def load_sampled_map_points(dataset_root, map_name):
    """读取一次 InLoc cutout 的采样 XYZ，并缓存全局坐标及其 map 像素坐标。"""
    dataset = Path(dataset_root)
    mat_path = Path(str(dataset / map_name) + ".mat")
    if not mat_path.is_file():
        raise FileNotFoundError(f"InLoc XYZcut file not found: {mat_path}")
    scan = np.asarray(loadmat(mat_path)["XYZcut"])
    if scan.ndim != 3 or scan.shape[2] != 3:
        raise ValueError(f"Expected XYZcut with shape (H, W, 3), got {scan.shape}: {mat_path}")
    map_height, map_width = scan.shape[:2]
    sample_step = max(1, int(np.ceil(max(map_height, map_width) / 700)))
    yy, xx = np.mgrid[0:map_height:sample_step, 0:map_width:sample_step]
    xyz_scan = scan[yy, xx].reshape(-1, 3).astype(np.float32)
    valid = np.isfinite(xyz_scan).all(axis=1)
    if not np.any(valid):
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 2), dtype=np.int16),
            map_height,
            map_width,
        )
    xyz_scan = xyz_scan[valid]
    map_xy = np.column_stack((xx.ravel(), yy.ravel()))[valid].astype(np.int16)
    scan_to_world = get_scan_pose(dataset, map_name)
    xyz_world = xyz_scan @ scan_to_world[:3, :3].astype(np.float32).T
    xyz_world += scan_to_world[:3, 3].astype(np.float32)
    return xyz_world, map_xy, map_height, map_width


@lru_cache(maxsize=128)
def get_image_size(path):
    """缓存 query 原图尺寸，避免每个候选都重新从磁盘解码 query 图。"""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return image.shape[1], image.shape[0]


def mask_cache_signature(dataset, query_name, map_name, poses_path, margin):
    """生成单对掩膜缓存签名；输入图、深度、对齐、位姿或 margin 变化会失效。"""
    query_path = dataset / query_name
    mat_path = Path(str(dataset / map_name) + ".mat")
    floor_name, scan_id, image_name = map_name.split("/")[-3:]
    transform_path = (
        dataset / f"{floor_name}_alignment" / "transformations"
        / f"{image_name[:3]}_trans_{scan_id}.txt"
    )
    stat_paths = (query_path, mat_path, transform_path, poses_path)
    stats = []
    for path in stat_paths:
        if not path.is_file():
            stats.append(None)
        else:
            stat = path.stat()
            stats.append((stat.st_size, stat.st_mtime_ns))
    return json.dumps(
        {"version": 1, "query": query_name, "map": map_name,
         "margin": margin, "stats": stats},
        sort_keys=True,
    )


def load_cached_masks(cache_path, signature):
    """读取签名匹配的磁盘掩膜缓存；缓存损坏或过期时返回 None。"""
    if not cache_path.is_file():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as cached:
            if str(cached["signature"].item()) != signature:
                return None
            if int(cached["has_masks"].item()) == 0:
                return False
            qshape = tuple(int(value) for value in cached["query_shape"])
            mshape = tuple(int(value) for value in cached["map_shape"])
            query_mask = np.unpackbits(cached["query_bits"], count=np.prod(qshape)).reshape(qshape).astype(bool)
            map_mask = np.unpackbits(cached["map_bits"], count=np.prod(mshape)).reshape(mshape).astype(bool)
            return query_mask, map_mask
    except (OSError, KeyError, ValueError):
        return None


def save_cached_masks(cache_path, signature, masks):
    """压缩保存一对掩膜或无共视结果，单对完成即落盘以支持中断续跑。"""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = cache_path.with_name(cache_path.stem + ".tmp.npz")
    if masks is None:
        np.savez_compressed(
            temp_path, signature=np.asarray(signature), has_masks=np.asarray(0, dtype=np.uint8)
        )
    else:
        query_mask, map_mask = masks
        np.savez_compressed(
            temp_path,
            signature=np.asarray(signature),
            has_masks=np.asarray(1, dtype=np.uint8),
            query_shape=np.asarray(query_mask.shape, dtype=np.int32),
            map_shape=np.asarray(map_mask.shape, dtype=np.int32),
            query_bits=np.packbits(query_mask),
            map_bits=np.packbits(map_mask),
        )
    temp_path.replace(cache_path)


def compute_pair_masks(dataset, query_name, map_name, initial_poses, margin=80):
    """由 InLoc ``XYZcut`` 和 query 初始位姿计算一对图像的共视掩膜。

    读取：query 原图用于获取 query 尺寸；``<map_name>.mat`` 中的 XYZcut；
    ``get_scan_pose`` 解析的数据库扫描到全局坐标变换；以及位姿文件中的
    query world-to-camera 外参。相机内参沿用 InLoc ``localize_inloc`` 的
    SIMPLE_PINHOLE 固定焦距设定。

    对 XYZcut 以 ``sample_step`` 下採樣，排除 NaN/Inf、相機後方點和投影到
    query 畫面外的點。共視 map 掩膜由這些點在 XYZcut 中的原始像素座標形成，
    query 掩膜由對應投影座標形成。零共視點或不足以形成區域時返回 None。
    """
    query_width, query_height = get_image_size(dataset / query_name)

    pose = initial_poses.get(Path(query_name).name)
    if pose is None:
        raise KeyError(f"No initial pose for query {query_name!r}")
    rotation, translation = pose
    xyz_world, map_xy, map_height, map_width = load_sampled_map_points(
        str(dataset.resolve()), map_name
    )
    if not len(xyz_world):
        return None
    xyz_query = xyz_world @ rotation.astype(np.float32).T + translation.astype(np.float32)
    depth = xyz_query[:, 2]
    positive = depth > 1e-6
    if not np.any(positive):
        return None
    xyz_query = xyz_query[positive]
    depth = depth[positive]
    map_xy = map_xy[positive]

    focal = 4032.0 * 28.0 / 36.0
    cx, cy = query_width * 0.5, query_height * 0.5
    query_xy = np.column_stack(
        (focal * xyz_query[:, 0] / depth + cx,
         focal * xyz_query[:, 1] / depth + cy)
    )
    in_frame = (
        (query_xy[:, 0] >= 0) & (query_xy[:, 0] < query_width)
        & (query_xy[:, 1] >= 0) & (query_xy[:, 1] < query_height)
    )
    if np.count_nonzero(in_frame) < 3:
        return None

    map_xy = map_xy[in_frame]
    query_xy = query_xy[in_frame]
    map_mask = make_hull_mask(map_xy, map_width, map_height, margin)
    query_mask = make_hull_mask(query_xy, query_width, query_height, margin)
    if not map_mask.any() or not query_mask.any():
        return None
    return query_mask, map_mask


def run(args):
    """抽取/复用局部特征，执行共视掩膜匹配并调用 InLoc PnP 定位。"""
    dataset = args.dataset.resolve()
    outputs = args.outputs.resolve()
    pairs_path = args.pairs.resolve()
    poses_path = args.initial_poses.resolve()
    for path in (dataset, pairs_path):
        if not path.exists():
            raise FileNotFoundError(path)
    outputs.mkdir(parents=True, exist_ok=True)

    retrieval = parse_retrieval(pairs_path)
    query_names = list(retrieval)
    map_names = {name for names in retrieval.values() for name in names}
    for name in query_names + sorted(map_names):
        if not (dataset / name).is_file():
            raise FileNotFoundError(f"Image referenced by pairs file not found: {dataset / name}")

    feature_conf = extract_features.confs[args.feature]
    matcher_conf = match_features.confs[args.matcher]
    features = outputs / f"{feature_conf['output']}.h5"
    if poses_path.is_file():
        initial_poses = read_initial_poses(poses_path)
    else:
        initial_poses = {}
        logger.warning(
            "Initial-pose file is missing: %s. All queries will use ordinary HLoc matching without covisibility cropping.",
            poses_path,
        )

    # 若上次运行已经生成完整共视 pairs 和匹配 HDF5，则跳过特征提取、
    # 掩膜计算及匹配，直接重跑定位。完整性按每个 pair 的 HDF5 组逐一检查。
    cached_pairs = outputs / f"{pairs_path.stem}-covis-and-fallback.txt"
    cached_matches = outputs / (
        f"{feature_conf['output']}_{matcher_conf['output']}_"
        f"{pairs_path.stem}-covis-and-fallback.h5"
    )
    cache_complete, pair_count, missing_count = (
        has_complete_match_groups(cached_pairs, cached_matches)
        if cached_pairs.is_file() and cached_matches.is_file()
        else (False, 0, 0)
    )
    if args.overwrite_matches:
        cache_complete = False
    if cache_complete:
        logger.info(
            "Reusing complete covisibility matches (%d pairs); skipping mask extraction and matching",
            pair_count,
        )
        results = outputs / args.results_name
        localize_inloc.main(
            dataset,
            cached_pairs,
            features,
            cached_matches,
            results,
            skip_matches=args.skip_matches,
        )
        if args.use_estimated_fallback:
            apply_estimated_pose_fallback(
                results,
                initial_poses,
                args.fallback_rotation_threshold,
                outputs,
            )
        logger.info("InLoc localization results saved to %s", results)
        return
    elif cached_pairs.is_file() and cached_matches.is_file():
        logger.warning(
            "Cached covisibility matches are incomplete (%d/%d pairs missing); "
            "will recompute masks and matching",
            missing_count,
            pair_count,
        )

    features = extract_features.main(
        feature_conf, dataset, outputs, image_list=sorted(set(query_names) | map_names)
    )

    queries_without_pose = [
        name for name in query_names if Path(name).name not in initial_poses
    ]
    if poses_path.is_file() and queries_without_pose:
        logger.warning(
            "Initial poses are missing for %d/%d queries; these queries will use ordinary HLoc matching: %s",
            len(queries_without_pose),
            len(query_names),
            ", ".join(queries_without_pose[:10]),
        )
    pose_status = {
        "initial_pose_file": str(poses_path),
        "file_exists": poses_path.is_file(),
        "queries_without_initial_pose": queries_without_pose,
        "query_count": len(query_names),
    }
    (outputs / "initial_pose_status.json").write_text(
        json.dumps(pose_status, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    def cached_masks_for_pair(query_name, map_name):
        cache_key = hashlib.sha1(
            f"{query_name}\0{map_name}".encode("utf-8")
        ).hexdigest()
        cache_path = outputs / "pair_mask_cache" / f"{cache_key}.npz"
        signature = mask_cache_signature(
            dataset, query_name, map_name, poses_path, args.margin
        )
        cached = load_cached_masks(cache_path, signature)
        if cached is False:
            return None
        if cached is not None:
            return cached
        masks = compute_pair_masks(
            dataset, query_name, map_name, initial_poses, margin=args.margin
        )
        save_cached_masks(cache_path, signature, masks)
        return masks

    # 每个 pair 完成后立即写掩膜缓存；如果进程中断，重跑时可直接复用已完成项。
    masks_for_pair = lru_cache(maxsize=args.mask_cache_size)(cached_masks_for_pair)

    retained = {}
    skipped_pairs = 0
    pairs_to_check = [
        (query_name, map_name)
        for query_name, candidates in retrieval.items()
        if Path(query_name).name in initial_poses
        for map_name in candidates
    ]
    for query_name, map_name in tqdm(
        pairs_to_check, desc="Computing InLoc covisibility", unit="pair"
    ):
        masks = masks_for_pair(query_name, map_name)
        if masks is None:
            skipped_pairs += 1
        else:
            retained.setdefault(query_name, []).append(map_name)

    covis_pairs = outputs / f"{pairs_path.stem}-covis.txt"
    covis_pairs.write_text(
        "".join(
            f"{query_name} {map_name}\n"
            for query_name, candidates in retained.items()
            for map_name in candidates
        ),
        encoding="utf-8",
    )
    kept_count = sum(map(len, retained.values()))
    logger.info("InLoc covisibility: kept %d pairs, skipped %d", kept_count, skipped_pairs)
    fallback_pairs = outputs / f"{pairs_path.stem}-no-initial-pose.txt"
    fallback_pairs.write_text(
        "".join(
            f"{query_name} {map_name}\n"
            for query_name in queries_without_pose
            for map_name in retrieval[query_name]
        ),
        encoding="utf-8",
    )

    all_pairs = outputs / f"{pairs_path.stem}-covis-and-fallback.txt"
    all_pairs.write_text(covis_pairs.read_text(encoding="utf-8") + fallback_pairs.read_text(encoding="utf-8"), encoding="utf-8")
    if not all_pairs.stat().st_size:
        raise ValueError("No retrieval pairs remain after covisibility filtering")

    covis_matches = outputs / f"{feature_conf['output']}_{matcher_conf['output']}_{covis_pairs.stem}.h5"
    if kept_count:
        match_covis_features(
            matcher_conf,
            covis_pairs,
            features,
            features,
            covis_matches,
            mask_provider=masks_for_pair,
            overwrite=args.overwrite_matches,
        ) # 耗时较长

    fallback_matches = outputs / f"{feature_conf['output']}_{matcher_conf['output']}_{fallback_pairs.stem}.h5"
    if queries_without_pose:
        match_features.main(
            matcher_conf,
            fallback_pairs,
            features,
            matches=fallback_matches,
            overwrite=args.overwrite_matches,
        )

    matches = outputs / f"{feature_conf['output']}_{matcher_conf['output']}_{pairs_path.stem}-covis-and-fallback.h5"
    merge_match_files(
        [covis_matches, fallback_matches],
        matches,
    )
    masks_for_pair.cache_clear()

    results = outputs / args.results_name
    localize_inloc.main(
        dataset,
        all_pairs,
        features,
        matches,
        results,
        skip_matches=args.skip_matches,
    )
    if args.use_estimated_fallback:
        apply_estimated_pose_fallback(
            results,
            initial_poses,
            args.fallback_rotation_threshold,
            outputs,
        )
    logger.info("InLoc covisibility localization results saved to %s", results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(r"D:\Datasets\InLoc"))
    parser.add_argument(
        "--pairs", type=Path,
        default=Path("pairs/inloc/pairs-query-netvlad20-local.txt"),
        help="Retrieval pairs whose names match the local feature HDF5 keys",
    )
    parser.add_argument(
        "--initial-poses", type=Path,
        default=Path(r"D:\project\reloc3r\output\InLoc\query_poses.txt"),
        help="Existing query pose estimates in COLMAP w2c text format",
    )
    parser.add_argument("--outputs", type=Path, default=Path("outputs/inloc_covis_all"))
    parser.add_argument("--feature", default="superpoint_my", choices=extract_features.confs.keys())
    parser.add_argument("--matcher", default="superpoint+lightglue", choices=match_features.confs.keys())
    parser.add_argument("--margin", type=int, default=80, help="Mask dilation radius in pixels")
    parser.add_argument("--mask-cache-size", type=int, default=4)
    parser.add_argument("--skip-matches", type=int, default=20)
    parser.add_argument("--results-name", default="InLoc_covis_poses.txt")
    parser.add_argument("--overwrite-matches", action="store_true")
    parser.add_argument(
        "--use-estimated-fallback",
        action="store_true",
        default=True,
        help="Use the initial estimated pose when PnP fails or its rotation error exceeds the threshold",
    )
    parser.add_argument(
        "--no-estimated-fallback",
        dest="use_estimated_fallback",
        action="store_false",
        help="Disable initial-pose fallback",
    )
    parser.add_argument(
        "--fallback-rotation-threshold",
        type=float,
        default=25.0,
        help="Rotation error threshold in degrees for initial-pose fallback",
    )
    args = parser.parse_args()
    if args.margin < 0 or args.mask_cache_size < 1 or args.fallback_rotation_threshold < 0:
        parser.error("--margin and --fallback-rotation-threshold must be non-negative; --mask-cache-size must be positive")
    run(args)


if __name__ == "__main__":
    main()
