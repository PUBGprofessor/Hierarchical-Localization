"""Aachen HLoc 定位流程：复用原始 top-k 与特征，加入共视区域裁剪。

候选图像、SuperPoint 特征、MegaLoc 检索结果和参考 SfM 都复用
``outputs/aachen`` 中原 pipeline 的结果，保持 ``num_covis=20``、
``num_loc=10``、``superpoint_my`` 和 ``superpoint+lightglue`` 配置一致。

裁剪位姿来自 reloc3r 的 query_poses.txt；query 列表只提供相机内参。
先按 query/map 双视锥筛选 SfM 3D 点，再仅在共享点凸包外扩区域内匹配。
PnP 失败或 HLoc 旋转误差超过 25 度时，最终位姿回退到 reloc3r 结果。
"""

import argparse
import csv
import json
import logging
import pickle
import sys
import time
from collections import namedtuple
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from hloc import covis_warp, extract_features, localize_sfm, logger, match_covis_features, match_features
from hloc.utils.parsers import parse_retrieval
from hloc.utils.read_write_model import qvec2rotmat


CameraData = namedtuple("CameraData", "model width height params")
PAIR_CSV_FIELDS = (
    "query",
    "map",
    "rank",
    "status",
    "shared_points",
    "query_mask_pixels",
    "query_mask_ratio",
    "map_mask_pixels",
    "map_mask_ratio",
)


def setup_logging(output_dir):
    """同时将 HLoc 和本脚本日志写到控制台与实验日志。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "pipeline_with_warp.log"
    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    formatter = logger.handlers[0].formatter if logger.handlers else None
    if formatter is not None:
        handler.setFormatter(formatter)
    logger.addHandler(handler)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)
    return log_path


def read_query_inputs(pattern):
    """读取 Aachen query 名称和内参，不读取 query 外参或真值位姿。"""
    files = sorted(Path(pattern.parent).glob(pattern.name))
    if not files:
        raise FileNotFoundError(f"No query lists matched: {pattern}")
    cameras = {}
    for path in files:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            fields = line.split()
            if not fields or fields[0].startswith("#"):
                continue
            if len(fields) < 5:
                raise ValueError(f"Invalid query camera at {path}:{line_number}: {line}")
            name, model, width, height, *params = fields
            if name in cameras:
                raise ValueError(f"Duplicate query name in query lists: {name}")
            cameras[name] = CameraData(
                model=model,
                width=int(width),
                height=int(height),
                params=np.asarray(params, dtype=np.float64),
            )
    if not cameras:
        raise ValueError(f"No queries were found in {files}")
    return cameras


def validate_features(feature_path, image_names):
    """检查复用的局部特征文件确实含有所有 query 和 top-k map 图。"""
    if not feature_path.is_file() or not feature_path.stat().st_size:
        raise FileNotFoundError(f"Expected existing SuperPoint features: {feature_path}")
    with h5py.File(feature_path, "r") as features:
        missing = sorted(name for name in image_names if name not in features)
    if missing:
        raise ValueError(
            f"SuperPoint feature file is missing {len(missing)} required image groups; "
            f"examples: {missing[:5]}"
        )


def lookup_by_image_name(table, name):
    """精确匹配图像名；仅在 basename 唯一时回退查找。"""
    if name in table:
        return name, table[name]
    basename = Path(name).name
    matches = [(key, value) for key, value in table.items() if Path(key).name == basename]
    if len(matches) == 1:
        return matches[0]
    return None, None


def read_pose_table(path):
    """读取 reloc3r/ HLoc 位姿文本，忽略注释和额外统计列。"""
    poses = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        if len(fields) < 8:
            raise ValueError(f"Invalid pose at {path}:{line_number}: {line}")
        values = np.asarray(fields[1:8], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Non-finite pose at {path}:{line_number}: {line}")
        poses[fields[0]] = values
    return poses


def rotation_error_deg(predicted_qvec, estimated_qvec):
    """计算两个 COLMAP world-to-camera 旋转之间的测地角度。"""
    predicted_rotation = qvec2rotmat(predicted_qvec)
    estimated_rotation = qvec2rotmat(estimated_qvec)
    relative = predicted_rotation @ estimated_rotation.T
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def apply_estimated_pose_fallback(
    results, estimated_poses, output_dir, threshold_deg, query_names=None
):
    """PnP 失败、HLoc 旋转偏差过大或没有 HLoc 位姿时回退到 reloc3r。"""
    if not results.is_file():
        raise FileNotFoundError(f"localize_sfm did not produce poses: {results}")
    predicted = read_pose_table(results)
    estimated = read_pose_table(estimated_poses)
    raw_results = output_dir / "query_poses_hloc.txt"
    raw_results.write_text(results.read_text(encoding="utf-8"), encoding="utf-8")

    logs_path = Path(f"{results}_logs.pkl")
    if not logs_path.is_file():
        raise FileNotFoundError(f"localize_sfm log is required for PnP fallback: {logs_path}")
    with logs_path.open("rb") as handle:
        localization_logs = pickle.load(handle).get("loc", {})

    report = {
        "rotation_threshold_deg": threshold_deg,
        "triggers": ["pnp_failed", "rotation_error_exceeded", "no_hloc_pose"],
        "replaced_count": 0,
        "queries": {},
    }
    fallback_names = set()
    for name, pose in predicted.items():
        estimated_name, estimated_pose = lookup_by_image_name(estimated, name)
        if estimated_pose is None:
            logger.warning("No reloc3r pose for HLoc result %r; retaining HLoc pose", name)
            report["queries"][name] = {"status": "no_estimated_pose"}
            continue

        _, loc_log = lookup_by_image_name(localization_logs, name)
        loc_log = loc_log or {}
        error = rotation_error_deg(pose[:4], estimated_pose[:4])
        pnp_failed = "PnP_ret" in loc_log and loc_log["PnP_ret"] is None
        rotation_exceeded = error > threshold_deg
        triggers = []
        if pnp_failed:
            triggers.append("pnp_failed")
        if rotation_exceeded:
            triggers.append("rotation_error_exceeded")
        report["queries"][name] = {
            "status": "estimated_pose_fallback" if triggers else "hloc",
            "estimated_pose_key": estimated_name,
            "rotation_error_deg": error,
            "pnp_failed": pnp_failed,
            "triggers": triggers,
        }
        if triggers:
            predicted[name] = estimated_pose
            fallback_names.add(name)

    for query_name in query_names or ():
        _, existing_pose = lookup_by_image_name(predicted, query_name)
        if existing_pose is not None:
            continue
        estimated_name, estimated_pose = lookup_by_image_name(estimated, query_name)
        output_name = Path(query_name).name
        if estimated_pose is None:
            logger.warning(
                "No HLoc or reloc3r pose for query %r; no final pose will be written",
                query_name,
            )
            report["queries"][output_name] = {"status": "no_hloc_or_estimated_pose"}
            continue
        predicted[output_name] = estimated_pose
        fallback_names.add(output_name)
        report["queries"][output_name] = {
            "status": "estimated_pose_fallback_no_hloc_pose",
            "query_name": query_name,
            "estimated_pose_key": estimated_name,
            "triggers": ["no_hloc_pose"],
        }
        logger.warning(
            "No HLoc pose for query %r; using its reloc3r pose in the final output",
            query_name,
        )

    output_lines = [
        f"{name} " + " ".join(map(str, pose)) for name, pose in predicted.items()
    ]
    results.write_text("\n".join(output_lines) + ("\n" if output_lines else ""), encoding="utf-8")
    report["replaced_count"] = len(fallback_names)
    (output_dir / "fallback_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    logger.info(
        "Reloc3r fallback replaced %d/%d poses (PnP failure, missing HLoc pose, "
        "or rotation error > %.1f deg)",
        len(fallback_names),
        len(predicted),
        threshold_deg,
    )
    return fallback_names


def write_pair_report(rows, output_dir):
    csv_path = output_dir / "covisibility_pairs.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIR_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    statuses = {}
    for row in rows:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    summary = {"pair_count": len(rows), "status_counts": statuses}
    (output_dir / "covisibility_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def run(args):
    started = time.perf_counter()
    dataset = args.dataset.resolve()
    outputs = args.outputs.resolve()
    output_dir = args.output_dir.resolve()
    log_path = setup_logging(output_dir)
    query_pattern = dataset / "queries" / "*_time_queries_with_intrinsics.txt"
    reference_sfm = outputs / "sfm_superpoint+lightglue"
    sfm_pairs = outputs / f"pairs-db-covis{args.num_covis}.txt"
    feature_conf = extract_features.confs["superpoint_my"]
    matcher_conf = match_features.confs["superpoint+lightglue"]
    feature_path = outputs / f"{feature_conf['output']}.h5"
    retrieval_pairs = outputs / f"pairs-query-megaloc{args.num_loc}.txt"
    estimated_poses = args.estimated_poses.resolve()

    logger.info("Starting Aachen covisibility localization")
    logger.info("Dataset: %s", dataset)
    logger.info("Existing HLoc artifacts: %s", outputs)
    logger.info("Experiment output: %s", output_dir)
    logger.info("Log file: %s", log_path)
    logger.info(
        "Parameters: num_covis=%d, num_loc=%d, margin=%d px, fallback_rotation=%.1f deg",
        args.num_covis,
        args.num_loc,
        args.margin,
        args.fallback_rotation_threshold,
    )

    for path in (
        dataset / "queries",
        reference_sfm,
        sfm_pairs,
        feature_path,
        retrieval_pairs,
        estimated_poses,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    query_cameras = read_query_inputs(query_pattern)
    query_names = set(query_cameras)
    raw_retrieval = parse_retrieval(retrieval_pairs)
    map_names = {name for candidates in raw_retrieval.values() for name in candidates}
    validate_features(feature_path, query_names | map_names)
    logger.info(
        "Reusing %s (%d query/map groups), %s, and %s",
        feature_path,
        len(query_names | map_names),
        retrieval_pairs,
        reference_sfm,
    )

    pose_names = set(read_pose_table(estimated_poses))
    missing_pose_queries = sorted(query_names - pose_names)
    if missing_pose_queries:
        logger.warning(
            "%d query images have no reloc3r pose; their pairs will use full-image masks. Examples: %s",
            len(missing_pose_queries),
            missing_pose_queries[:5],
        )

    retained = {}
    pair_rows = []
    row_by_pair = {}
    skipped_queries = 0
    removed_pairs = 0
    total_pairs = 0
    for query_name, candidates in raw_retrieval.items():
        if query_name not in query_cameras:
            skipped_queries += 1
            continue
        query_camera = query_cameras[query_name]
        good = []
        for rank, map_name in enumerate(candidates, 1):
            total_pairs += 1
            row = {
                "query": query_name,
                "map": map_name,
                "rank": rank,
                "status": "retained",
                "shared_points": "",
                "query_mask_pixels": "",
                "query_mask_ratio": "",
                "map_mask_pixels": "",
                "map_mask_ratio": "",
            }
            try:
                geometry = covis_warp.compute_pair_geometry(
                    query_name,
                    map_name,
                    estimated_poses,
                    reference_sfm,
                    query_camera,
                    margin=args.margin,
                    build_masks=False,
                )
            except covis_warp.PoseNotFoundError:
                # 缺少估计位姿时不能判定共视关系；保留候选并在匹配阶段用全图掩膜。
                row["status"] = "missing_pose_full_image"
                geometry = "missing_pose"
            if geometry is None:
                row["status"] = "zero_shared_points"
                removed_pairs += 1
            else:
                if isinstance(geometry, dict):
                    row["shared_points"] = int(len(geometry["point3D_ids"]))
                good.append(map_name)
            pair_rows.append(row)
            row_by_pair[(query_name, map_name)] = row
        if good:
            retained[query_name] = good

    filtered_pairs = output_dir / f"pairs-query-megaloc{args.num_loc}-covis.txt"
    filtered_pairs.write_text(
        "".join(
            f"{query_name} {map_name}\n"
            for query_name, candidates in retained.items()
            for map_name in candidates
        ),
        encoding="utf-8",
    )
    kept_pairs = sum(map(len, retained.values()))
    logger.info(
        "Covisibility prefilter: kept %d/%d pairs, removed %d zero-shared-point pairs; "
        "skipped %d retrieval query keys outside the Aachen query lists",
        kept_pairs,
        total_pairs,
        removed_pairs,
        skipped_queries,
    )
    if not kept_pairs:
        write_pair_report(pair_rows, output_dir)
        raise ValueError("No retrieval pairs have shared 3D points; localization cannot continue")

    loc_matches = output_dir / f"matches-covis-megaloc{args.num_loc}.h5"

    def pair_masks(query_name, map_name):
        masks = covis_warp.compute_pair_masks(
            query_name,
            map_name,
            estimated_poses,
            reference_sfm,
            query_cameras[query_name],
            margin=args.margin,
        )
        if masks is None:
            raise RuntimeError(
                f"Pair {query_name!r}, {map_name!r} passed the shared-point filter but has no mask"
            )
        query_mask, map_mask = masks
        row = row_by_pair[(query_name, map_name)]
        row["query_mask_pixels"] = int(query_mask.sum())
        row["query_mask_ratio"] = float(query_mask.mean())
        row["map_mask_pixels"] = int(map_mask.sum())
        row["map_mask_ratio"] = float(map_mask.mean())
        return masks

    logger.info("Running masked SuperPoint + LightGlue matching: %s", loc_matches)
    match_covis_features.main(
        matcher_conf,
        filtered_pairs,
        feature_path,
        feature_path,
        loc_matches,
        mask_provider=pair_masks,
        overwrite=args.overwrite_matches,
    )
    pair_summary = write_pair_report(pair_rows, output_dir)
    logger.info("Saved pair-level covisibility report: %s", output_dir / "covisibility_pairs.csv")

    # 释放文本 SfM 的 NumPy 索引，避免它与 pycolmap 定位模型同时占用内存。
    covis_warp.clear_model_cache()
    results = output_dir / "query_poses.txt"
    logger.info("Running HLoc PnP localization: %s", results)
    localize_sfm.main(
        reference_sfm,
        query_pattern,
        filtered_pairs,
        feature_path,
        loc_matches,
        results,
        covisibility_clustering=False,
    )
    fallback_names = apply_estimated_pose_fallback(
        results,
        estimated_poses,
        output_dir,
        threshold_deg=args.fallback_rotation_threshold,
        query_names=sorted(query_names),
    )

    pose_count = len(read_pose_table(results))
    run_summary = {
        "dataset": str(dataset),
        "reference_sfm": str(reference_sfm),
        "features": str(feature_path),
        "retrieval_pairs": str(retrieval_pairs),
        "estimated_poses": str(estimated_poses),
        "query_count": len(query_names),
        "retrieval_query_keys": len(raw_retrieval),
        "retrieval_pairs_considered": total_pairs,
        "retrieval_query_keys_skipped": skipped_queries,
        "pairs_kept": kept_pairs,
        "pairs_removed_zero_shared_points": removed_pairs,
        "queries_without_retained_pairs": len(query_names - set(retained)),
        "pair_status_counts": pair_summary["status_counts"],
        "localized_pose_count": pose_count,
        "estimated_pose_fallback_count": len(fallback_names),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "pipeline_summary.json").write_text(
        json.dumps(run_summary, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("Finished Aachen covisibility localization: %s", run_summary)
    logger.info("Final poses: %s", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(r"D:\Datasets\Aachen-Day-Night\dataset"),
        help="Aachen-Day-Night dataset directory",
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default=ROOT / "outputs" / "aachen",
        help="Existing HLoc outputs and reusable features",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "aachen_covis_all",
        help="Output directory for this covisibility experiment",
    )
    parser.add_argument("--num-covis", type=int, default=20)
    parser.add_argument("--num-loc", type=int, default=10)
    parser.add_argument(
        "--estimated-poses",
        type=Path,
        default=Path(r"D:\project\reloc3r\output\Aachen_full\query_poses.txt"),
        help="reloc3r estimated query poses in COLMAP world-to-camera format",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=80,
        help="Pixel expansion around each shared-point convex hull",
    )
    parser.add_argument(
        "--fallback-rotation-threshold",
        type=float,
        default=25.0,
        help="Use reloc3r pose when HLoc rotation error exceeds this threshold",
    )
    parser.add_argument(
        "--overwrite-matches",
        action="store_true",
        help="Recompute matching groups already present in the output HDF5",
    )
    args = parser.parse_args()
    if args.num_covis < 1 or args.num_loc < 1:
        parser.error("--num-covis and --num-loc must be positive")
    if args.margin < 0 or args.fallback_rotation_threshold <= 0:
        parser.error("--margin must be non-negative and fallback threshold must be positive")
    try:
        run(args)
    except Exception:
        logger.exception("Aachen covisibility pipeline failed")
        raise
