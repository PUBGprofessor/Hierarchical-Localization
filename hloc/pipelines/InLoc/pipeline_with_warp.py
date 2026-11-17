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
import json
import pickle
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from ... import extract_features, localize_inloc, logger, match_features
from ...match_covis_features import main as match_covis_features
from ...utils.parsers import names_to_pair, parse_retrieval
from ...utils.read_write_model import qvec2rotmat, rotmat2qvec
from .covis import (
    compute_pair_masks,
    load_cached_masks,
    mask_cache_path,
    mask_cache_signature,
    read_initial_poses,
    save_cached_masks,
)


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
        cache_path = mask_cache_path(outputs, query_name, map_name)
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
