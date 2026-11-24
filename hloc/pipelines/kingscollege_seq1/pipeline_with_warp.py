"""KingsCollege_seq1 localization with covisibility-masked LightGlue matching.

The Query/Reference split, full COLMAP map, intrinsics, descriptors, and
retrieval candidates are shared with :mod:`pipeline`. Only the local matching
stage changes: estimated Reloc3r poses and map 3D points define pair masks.
"""

import argparse
import gc
import json
import sys
from functools import lru_cache
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from hloc import covis_warp, localize_sfm, logger, match_covis_features, match_features
from hloc.utils.parsers import parse_retrieval
from hloc.pipelines.kingscollege_seq1 import pipeline as baseline


def _run(args, context):
    estimated_poses = Path(args.estimated_poses).resolve()
    if not estimated_poses.is_file():
        raise FileNotFoundError(estimated_poses)
    feature_paths = baseline.ensure_features(context)
    reference_model = baseline.ensure_reference_model(
        context, args, feature_paths["local"]
    )
    retrieval_path, retrieval_pairs = baseline.make_retrieval_pairs(
        context, args, feature_paths["global"]
    )
    matcher_conf = match_features.confs["superpoint+lightglue"]

    @lru_cache(maxsize=4)
    def pair_masks(query_name, reference_name):
        return covis_warp.compute_pair_masks(
            query_name,
            reference_name,
            estimated_poses,
            reference_model,
            context["camera_by_name"][query_name],
            margin=args.margin,
        )

    # 先按两侧视锥的共享 3D 点删去不相关候选；位姿缺失时 covis_warp
    # 会明确退回全图掩膜，保留该检索候选继续执行普通局部匹配。
    raw_retrieval = parse_retrieval(retrieval_path)
    retained = {}
    removed_pairs = 0
    for query_name, candidates in raw_retrieval.items():
        kept = []
        for reference_name in candidates:
            if pair_masks(query_name, reference_name) is None:
                removed_pairs += 1
            else:
                kept.append(reference_name)
        if kept:
            retained[query_name] = kept

    filtered_pairs = context["output_dir"] / f"pairs-query-megaloc{args.num_loc}-covis.txt"
    filtered_pairs.write_text(
        "".join(
            f"{query_name} {reference_name}\n"
            for query_name, candidates in retained.items()
            for reference_name in candidates
        ),
        encoding="utf-8",
    )
    kept_pairs = sum(map(len, retained.values()))
    logger.info(
        "KingsCollege covisibility filter: kept %d/%d retrieval pairs, rejected %d",
        kept_pairs,
        len(retrieval_pairs),
        removed_pairs,
    )

    results = context["output_dir"] / "query_poses.txt"
    fallback_names = set()
    if kept_pairs:
        matches = context["output_dir"] / f"{matcher_conf['output']}-covis-megaloc{args.num_loc}.h5"
        match_covis_features.main(
            matcher_conf,
            filtered_pairs,
            feature_paths["local"],
            feature_paths["local"],
            matches,
            mask_provider=pair_masks,
            overwrite=args.overwrite_matches,
        )
        # 释放紧凑模型索引，避免与 localize_sfm 加载的 COLMAP 模型并存。
        covis_warp.clear_model_cache()
        pair_masks.cache_clear()
        gc.collect()
        localize_sfm.main(
            context["model_dir"],
            context["query_file"],
            filtered_pairs,
            feature_paths["local"],
            matches,
            results,
            covisibility_clustering=False,
        )
    else:
        # 不让无候选的 run 在局部化调用前中断；空预测会被评估器按完整
        # Query 名单展开为逐图 missing 记录，便于和 Baseline 对齐比较。
        results.write_text("", encoding="utf-8")
        covis_warp.clear_model_cache()
        pair_masks.cache_clear()

    fallback_names = baseline.apply_estimated_pose_fallback(
        results,
        estimated_poses,
        context["output_dir"],
        threshold_deg=args.fallback_rotation_threshold,
        query_names=context["query_names"],
    )

    baseline.evaluate(
        results,
        context,
        context["output_dir"] / "evaluation.json",
        fallback_names=fallback_names,
    )
    summary = baseline.write_run_summary(
        context, args, retrieval_pairs, feature_paths, fallback_count=len(fallback_names)
    )
    summary.update(
        {
            "estimated_poses": str(estimated_poses),
            "retrieval_pairs_source": str(retrieval_path.resolve()),
            "covisibility_margin_px": args.margin,
            "fallback_rotation_threshold_deg": args.fallback_rotation_threshold,
            "retrieval_pairs_considered": len(retrieval_pairs),
            "covisibility_pairs_kept": kept_pairs,
            "covisibility_pairs_removed": removed_pairs,
            "queries_without_covisible_candidates": len(
                set(context["query_names"]) - set(retained)
            ),
        }
    )
    (context["output_dir"] / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (context["output_dir"] / "stage_complete.json").write_text(
        json.dumps({"stage": "ours", "query_count": len(context["query_names"])}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    logger.info("KingsCollege Ours poses: %s", results)


def run(args):
    """Run masked matching, HLoc PnP, Reloc3r fallback, and per-query evaluation."""
    context = baseline.prepare_run(args)
    file_handler = baseline.setup_run_logging(context)
    try:
        baseline.log_run_context(context)
        _run(args, context)
    finally:
        baseline.close_run_logging(file_handler)


def main(argv=None):
    parser = baseline.add_common_arguments(
        argparse.ArgumentParser(description=__doc__), "ours"
    )
    parser.add_argument(
        "--estimated-poses",
        type=Path,
        required=True,
        help="Reloc3r query poses in COLMAP world-to-camera format",
    )
    parser.add_argument(
        "--retrieval-pairs",
        type=Path,
        default=None,
        help="Reuse a Baseline query/reference pairs file after split validation",
    )
    parser.add_argument("--margin", type=int, default=80,
                        help="Pixel expansion around projected shared-point regions")
    parser.add_argument("--fallback-rotation-threshold", type=float, default=25.0,
                        help="Use Reloc3r pose when HLoc rotation error is greater than this angle")
    parser.add_argument("--overwrite-matches", action="store_true",
                        help="Recompute existing masked matching groups")
    args = parser.parse_args(argv)
    if args.num_covis < 1 or args.num_loc < 1:
        parser.error("--num-covis and --num-loc must be positive")
    if args.margin < 0 or args.fallback_rotation_threshold <= 0:
        parser.error("--margin must be non-negative and fallback threshold must be positive")
    run(args)


if __name__ == "__main__":
    main()
