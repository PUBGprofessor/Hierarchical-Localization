"""HLoc baseline localization for an explicit KingsCollege_seq1 split.

All registered images and poses in the supplied ``sparse/0`` model contribute
to a SuperPoint re-triangulation. The Reference list restricts global retrieval
and the Query list defines localization and evaluation.
"""

import argparse
import csv
import json
import logging
import os
import pickle
import sys
from pathlib import Path, PurePosixPath

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
from hloc.utils.io import list_h5_names
from hloc.utils.read_write_model import (
    qvec2rotmat,
    read_cameras_binary,
    read_images_binary,
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = Path(r"D:\Datasets\KingsCollege_seq1")
DEFAULT_SPLIT_DIR = Path(r"D:\project\HLoc_Reloc3r_Experiments\splits\KingsCollege_seq1")
DEFAULT_CACHE_DIR = Path(r"D:\project\HLoc_Reloc3r_Experiments\cache\KingsCollege_seq1")


def _normalize_names(names, label):
    normalized = []
    seen = set()
    for raw_name in names:
        name = str(raw_name).strip().replace("\\", "/")
        if not name or name.startswith("#"):
            continue
        image_path = PurePosixPath(name)
        if image_path.is_absolute() or ".." in image_path.parts or ":" in image_path.parts[0]:
            raise ValueError(f"Invalid {label} image name: {name!r}")
        if name in seen:
            raise ValueError(f"Duplicate {label} image name: {name}")
        seen.add(name)
        normalized.append(name)
    if not normalized:
        raise ValueError(f"The {label} image list is empty")
    return normalized


def read_name_list(path):
    """Read an image-name list from a text file or a JSON string array."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise ValueError(f"Expected a JSON array in {path}")
        names = value
    else:
        names = [
            line.strip()
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    return names


def _manifest_lists(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in split manifest {path}")
    aliases = {
        "query": ("query", "queries", "query_names"),
        "reference": ("reference", "references", "reference_names"),
    }
    lists = {}
    for role, keys in aliases.items():
        found = next((key for key in keys if key in value), None)
        if found is None or not isinstance(value[found], list):
            raise ValueError(
                f"Split manifest {path} must contain a {role} or {role}s array"
            )
        names = []
        for entry in value[found]:
            if isinstance(entry, str):
                names.append(entry)
            elif isinstance(entry, dict):
                image_name = entry.get("image_name", entry.get("name"))
                if not isinstance(image_name, str):
                    raise ValueError(
                        f"Each {role} entry in {path} must be a name or contain "
                        "an image_name string"
                    )
                names.append(image_name)
            else:
                raise ValueError(
                    f"Each {role} entry in {path} must be a name or image record"
                )
        lists[role] = names
    return lists["query"], lists["reference"]


def load_split(split_dir, manifest=None, query_list=None, reference_list=None):
    """Load explicit Query and Reference names from a JSON manifest or lists."""
    if (query_list is None) != (reference_list is None):
        raise ValueError("--query-list and --reference-list must be provided together")
    if query_list is not None:
        query_names = read_name_list(query_list)
        reference_names = read_name_list(reference_list)
    else:
        split_dir = Path(split_dir)
        manifest_path = Path(manifest) if manifest is not None else split_dir / "manifest.json"
        if manifest_path.is_file():
            query_names, reference_names = _manifest_lists(manifest_path)
        elif manifest is not None:
            raise FileNotFoundError(manifest_path)
        else:
            pairs = (
                ("queries.txt", "references.txt"),
                ("query.txt", "reference.txt"),
                ("query_names.txt", "reference_names.txt"),
            )
            list_paths = next(
                ((split_dir / q, split_dir / r) for q, r in pairs
                 if (split_dir / q).is_file() and (split_dir / r).is_file()),
                None,
            )
            if list_paths is None:
                discovered = sorted(split_dir.glob("seed_*/ratio_*/manifest.json"))
                if discovered:
                    examples = [str(path.parent) for path in discovered[:3]]
                    raise ValueError(
                        "Multiple explicit split manifests are available; pass "
                        f"--split-dir for one seed/ratio directory, e.g. {examples}"
                    )
                raise FileNotFoundError(
                    f"No split manifest found in {split_dir}; provide manifest.json "
                    "or both --query-list and --reference-list"
                )
            query_names, reference_names = map(read_name_list, list_paths)

    query_names = _normalize_names(query_names, "Query")
    reference_names = _normalize_names(reference_names, "Reference")
    overlap = set(query_names) & set(reference_names)
    if overlap:
        raise ValueError(f"Query and Reference lists overlap: {sorted(overlap)[:5]}")
    return query_names, reference_names


def prepare_run(args):
    """Validate the split and full binary COLMAP model shared by both pipelines."""
    dataset = Path(args.dataset).resolve()
    image_dir = dataset / "images"
    model_dir = dataset / "sparse" / "0"
    for path in (image_dir, model_dir):
        if not path.is_dir():
            raise FileNotFoundError(path)
    required_model_files = ("cameras.bin", "images.bin", "points3D.bin")
    for name in required_model_files:
        path = model_dir / name
        if not path.is_file() or not path.stat().st_size:
            raise FileNotFoundError(f"Incomplete COLMAP model: {path}")

    query_names, reference_names = load_split(
        args.split_dir, args.manifest, args.query_list, args.reference_list
    )
    images = read_images_binary(model_dir / "images.bin")
    cameras = read_cameras_binary(model_dir / "cameras.bin")
    image_by_name = {image.name: image for image in images.values()}
    model_names = set(image_by_name)
    for role, names in (("Query", query_names), ("Reference", reference_names)):
        unknown = sorted(set(names) - model_names)
        if unknown:
            raise ValueError(f"{role} images are absent from {model_dir}: {unknown[:5]}")
        missing = [name for name in names if not (image_dir / Path(name)).is_file()]
        if missing:
            raise FileNotFoundError(f"{role} image files are missing: {missing[:5]}")

    output_dir = Path(args.output_dir).resolve()
    feature_dir = Path(args.cache_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(feature_dir / "torch_cache"))
    log_path = output_dir / "pipeline.log"
    query_file = output_dir / "queries_with_intrinsics.txt"
    camera_by_name = {
        name: cameras[image_by_name[name].camera_id] for name in query_names
    }
    lines = []
    for name in query_names:
        camera = camera_by_name[name]
        lines.append(
            f"{name} {camera.model} {camera.width} {camera.height} "
            + " ".join(map(str, camera.params))
        )
    query_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    context = {
        "dataset": dataset,
        "image_dir": image_dir,
        "model_dir": model_dir,
        "model_images": image_by_name,
        "cameras": cameras,
        "camera_by_name": camera_by_name,
        "query_names": query_names,
        "reference_names": reference_names,
        "query_file": query_file,
        "output_dir": output_dir,
        "feature_dir": feature_dir,
        "log_path": log_path,
        "model_image_count": len(image_by_name),
    }
    return context


def setup_run_logging(context):
    """Attach one UTF-8 file handler alongside HLoc's existing console handler."""
    file_handler = logging.FileHandler(context["log_path"], encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s %(name)s %(levelname)s] %(message)s",
            datefmt="%Y/%m/%d %H:%M:%S",
        )
    )
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    return file_handler


def close_run_logging(file_handler):
    logger.removeHandler(file_handler)
    file_handler.flush()
    file_handler.close()


def log_run_context(context):
    logger.info(
        "KingsCollege split: %d queries, %d retrieval references; complete model has "
        "%d registered images and all sparse/0 points",
        len(context["query_names"]),
        len(context["reference_names"]),
        context["model_image_count"],
    )


def ensure_features(context):
    """Extract only missing groups, sharing both feature caches across runs."""
    feature_conf = extract_features.confs["superpoint_my"]
    retrieval_conf = extract_features.confs["megaloc"]
    names_by_kind = {
        "local": sorted(context["model_images"]),
        "global": sorted(set(context["query_names"]) | set(context["reference_names"])),
    }
    paths = {}
    for kind, conf in (("local", feature_conf), ("global", retrieval_conf)):
        path = context["feature_dir"] / f"{conf['output']}.h5"
        existing = set(list_h5_names(path)) if path.is_file() else set()
        missing = [name for name in names_by_kind[kind] if name not in existing]
        if missing:
            logger.info("Extracting %d missing %s feature groups", len(missing), kind)
            extract_features.main(
                conf,
                context["image_dir"],
                context["feature_dir"],
                image_list=missing,
            )
        final_names = set(list_h5_names(path)) if path.is_file() else set()
        absent = sorted(set(names_by_kind[kind]) - final_names)
        if absent:
            raise RuntimeError(f"Feature cache {path} is missing images: {absent[:5]}")
        paths[kind] = path
    return paths


def ensure_reference_model(context, args, local_features):
    """使用完整固定 SfM 图像集合重三角化 SuperPoint landmarks。

    COLMAP 原模型中的 2D feature index 不对应 SuperPoint 索引，因此先保留
    所有原相机位姿，再利用共视 SuperPoint+LightGlue 匹配生成定位用 3D 点。
    """
    matcher_conf = match_features.confs["superpoint+lightglue"]
    stem = f"sfm-{extract_features.confs['superpoint_my']['output']}-covis{args.num_covis}"
    reference_model = context["feature_dir"] / stem
    sfm_pairs = context["feature_dir"] / f"pairs-covis{args.num_covis}.txt"
    if not sfm_pairs.is_file() or not sfm_pairs.stat().st_size:
        pairs_from_covisibility.main(
            context["model_dir"], sfm_pairs, num_matched=args.num_covis
        )
    if not sfm_pairs.stat().st_size:
        raise ValueError("The full KingsCollege SfM model has no covisible image pairs")

    sfm_matches = context["feature_dir"] / f"{matcher_conf['output']}-covis{args.num_covis}.h5"
    if not sfm_matches.is_file() or not sfm_matches.stat().st_size:
        match_features.main(
            matcher_conf, sfm_pairs, local_features, matches=sfm_matches
        )

    if not all((reference_model / name).is_file() for name in (
        "cameras.bin", "images.bin", "points3D.bin"
    )):
        triangulation.main(
            reference_model,
            context["model_dir"],
            context["image_dir"],
            sfm_pairs,
            local_features,
            sfm_matches,
        )
    images_path = reference_model / "images.bin"
    if any(not (reference_model / name).is_file() for name in (
        "cameras.bin", "images.bin", "points3D.bin"
    )) or not images_path.stat().st_size:
        raise RuntimeError(f"SuperPoint reference model was not created: {reference_model}")
    reference_names = {image.name for image in read_images_binary(images_path).values()}
    missing = sorted(set(context["model_images"]) - reference_names)
    if missing:
        raise RuntimeError(
            "The re-triangulated reference model does not contain every registered "
            f"image from sparse/0: {missing[:5]}"
        )
    context["reference_model"] = reference_model
    return reference_model


def make_retrieval_pairs(context, args, global_features):
    """Regenerate top-k pairs while restricting the database to Reference names."""
    output = context["output_dir"] / f"pairs-query-megaloc{args.num_loc}.txt"
    supplied_pairs = getattr(args, "retrieval_pairs", None)
    if supplied_pairs is not None:
        output = Path(supplied_pairs).resolve()
        if not output.is_file():
            raise FileNotFoundError(output)
    else:
        pairs_from_retrieval.main(
            global_features,
            output,
            min(args.num_loc, len(context["reference_names"])),
            query_list=context["query_names"],
            db_list=context["reference_names"],
            db_descriptors=global_features,
        )
    pairs = []
    for line in output.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"Invalid retrieval pair line: {line!r}")
        pairs.append((fields[0], fields[1]))
    if not pairs:
        raise ValueError("MegaLoc generated no retrieval pairs")
    query_set = set(context["query_names"])
    reference_set = set(context["reference_names"])
    invalid_queries = sorted({query for query, _ in pairs} - query_set)
    invalid_references = sorted({reference for _, reference in pairs} - reference_set)
    if invalid_queries or invalid_references:
        raise ValueError(
            "Retrieval pairs must stay inside this split; "
            f"invalid queries={invalid_queries[:5]}, references={invalid_references[:5]}"
        )
    if len(pairs) != len(set(pairs)):
        raise ValueError("Retrieval pairs contain duplicate query/reference rows")
    return output, pairs


def read_pose_table(path):
    """Read COLMAP world-to-camera poses from ``name qw qx qy qz tx ty tz``."""
    poses = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        if len(fields) < 8:
            raise ValueError(f"Invalid pose line in {path}: {line}")
        poses[fields[0]] = np.asarray([float(value) for value in fields[1:8]], dtype=float)
    return poses


def lookup_by_image_name(table, name):
    if name in table:
        return name, table[name]
    basename = Path(name).name
    matches = [(key, value) for key, value in table.items() if Path(key).name == basename]
    return matches[0] if len(matches) == 1 else (None, None)


def rotation_error_deg(predicted_qvec, estimated_qvec):
    predicted_rotation = qvec2rotmat(predicted_qvec)
    estimated_rotation = qvec2rotmat(estimated_qvec)
    relative = predicted_rotation @ estimated_rotation.T
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def apply_estimated_pose_fallback(
    results, estimated_poses, output_dir, threshold_deg=25.0, query_names=None
):
    """PnP 失败或旋转差严格大于阈值时，使用 Reloc3r 估计的完整 w2c 位姿。"""
    predicted = read_pose_table(results)
    estimated = read_pose_table(estimated_poses)
    logs_path = Path(f"{results}_logs.pkl")
    if logs_path.is_file():
        with logs_path.open("rb") as handle:
            localization_logs = pickle.load(handle).get("loc", {})
    else:
        localization_logs = {}

    output_dir.mkdir(parents=True, exist_ok=True)
    Path(output_dir, "query_poses_hloc.txt").write_text(
        Path(results).read_text(encoding="utf-8"), encoding="utf-8"
    )
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
            report["queries"][name] = {"status": "no_estimated_pose"}
            continue
        _, loc_log = lookup_by_image_name(localization_logs, name)
        pnp_failed = (loc_log or {}).get("PnP_ret") is None
        error = rotation_error_deg(pose[:4], estimated_pose[:4])
        rotation_exceeded = error > threshold_deg
        use_estimated = pnp_failed or rotation_exceeded
        triggers = []
        if pnp_failed:
            triggers.append("pnp_failed")
        if rotation_exceeded:
            triggers.append("rotation_error_exceeded")
        report["queries"][name] = {
            "estimated_pose_key": estimated_name,
            "rotation_error_deg": error,
            "pnp_failed": pnp_failed,
            "triggers": triggers,
            "status": "estimated_pose_fallback" if use_estimated else "hloc",
        }
        if use_estimated:
            predicted[name] = estimated_pose
            fallback_names.add(name)

    # localize_sfm skips queries with no retrieval candidates entirely. Include
    # their Reloc3r poses as the PnP-failed fallback when available, preserving
    # the same query coverage in the final pose file and evaluation report.
    for name in query_names or ():
        _, existing_pose = lookup_by_image_name(predicted, name)
        if existing_pose is not None:
            continue
        estimated_name, estimated_pose = lookup_by_image_name(estimated, name)
        if estimated_pose is None:
            report["queries"][name] = {"status": "no_estimated_pose"}
            continue
        predicted[name] = estimated_pose
        fallback_names.add(name)
        report["queries"][name] = {
            "estimated_pose_key": estimated_name,
            "pnp_failed": True,
            "triggers": ["pnp_failed"],
            "status": "estimated_pose_fallback",
        }

    Path(results).write_text(
        "\n".join(f"{name} " + " ".join(map(str, pose)) for name, pose in predicted.items())
        + ("\n" if predicted else ""),
        encoding="utf-8",
    )
    report["replaced_count"] = len(fallback_names)
    (Path(output_dir) / "fallback_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("Reloc3r fallback replaced %d/%d poses", len(fallback_names), len(predicted))
    return fallback_names


def evaluate(results, context, output, fallback_names=None):
    """Write one result row for every explicit Query, including failures."""
    fallback_names = fallback_names or set()
    predictions = read_pose_table(results) if Path(results).is_file() else {}
    logs_path = Path(f"{results}_logs.pkl")
    if logs_path.is_file():
        with logs_path.open("rb") as handle:
            logs = pickle.load(handle).get("loc", {})
    else:
        logs = {}
    rows = []
    for name in context["query_names"]:
        _, prediction = lookup_by_image_name(predictions, name)
        image = context["model_images"][name]
        method = "missing"
        status = "pnp_failed_no_pose"
        translation = rotation = None
        if prediction is not None:
            _, loc_log = lookup_by_image_name(logs, name)
            if name in fallback_names or Path(name).name in fallback_names:
                method = "estimated_pose_fallback"
                status = "estimated_pose_fallback_after_pnp_failure"
            else:
                if (loc_log or {}).get("PnP_ret") is not None:
                    method = "pnp"
                    status = "pnp_localized"
                else:
                    method = "retrieval_fallback"
                    status = "pnp_failed_top1_retrieval_pose"
            gt_rotation = image.qvec2rotmat()
            predicted_rotation = qvec2rotmat(prediction[:4])
            # COLMAP stores world-to-camera t; position error must use camera centers.
            gt_center = -gt_rotation.T @ image.tvec
            predicted_center = -predicted_rotation.T @ prediction[4:]
            translation = float(np.linalg.norm(gt_center - predicted_center))
            cosine = np.clip((np.trace(gt_rotation.T @ predicted_rotation) - 1.0) / 2.0, -1.0, 1.0)
            rotation = float(np.degrees(np.arccos(cosine)))
        rows.append({
            "image": name,
            "method": method,
            "status": status,
            "translation_m": translation,
            "rotation_deg": rotation,
        })

    def metrics(items):
        if not items:
            return None
        return {
            "count": len(items),
            "mean_translation_m": float(np.mean([row["translation_m"] for row in items])),
            "median_translation_m": float(np.median([row["translation_m"] for row in items])),
            "mean_rotation_deg": float(np.mean([row["rotation_deg"] for row in items])),
            "median_rotation_deg": float(np.median([row["rotation_deg"] for row in items])),
        }

    estimated = [row for row in rows if row["translation_m"] is not None]
    pnp = [row for row in rows if row["method"] == "pnp"]
    summary = {
        "total_queries": len(rows),
        "pnp_localized": len(pnp),
        "successful_localizations": len(pnp),
        "pnp_failed": len(rows) - len(pnp),
        "estimated_pose_fallback": sum(row["method"] == "estimated_pose_fallback" for row in rows),
        "retrieval_fallback": sum(row["method"] == "retrieval_fallback" for row in rows),
        "missing": len(rows) - len(estimated),
        "all_estimated": metrics(estimated),
        "pnp_only": metrics(pnp),
    }
    Path(output).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with Path(output).with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    logger.info("KingsCollege evaluation: %s", summary)
    return summary


def write_run_summary(context, args, pairs, feature_paths, fallback_count=0):
    summary = {
        "dataset": str(context["dataset"]),
        "split_dir": str(Path(args.split_dir).resolve()),
        "sfm_model": str(context["model_dir"]),
        "localization_model": str(context.get("reference_model", "")),
        "sfm_map_scope": (
            "All registered images and sparse points from sparse/0 contribute; "
            "source poses are fixed while SuperPoint landmarks are re-triangulated"
        ),
        "retrieval_scope": "Explicit Reference list only",
        "retrieval_pairs_source": str(
            Path(args.retrieval_pairs).resolve()
            if getattr(args, "retrieval_pairs", None) is not None
            else context["output_dir"] / f"pairs-query-megaloc{args.num_loc}.txt"
        ),
        "query_count": len(context["query_names"]),
        "reference_count": len(context["reference_names"]),
        "sfm_registered_image_count": context["model_image_count"],
        "retrieval_pair_count": len(pairs),
        "feature_files": {key: str(value) for key, value in feature_paths.items()},
        "cache_dir": str(context["feature_dir"]),
        "log_file": str(context["log_path"]),
        "num_covis": args.num_covis,
        "num_loc": args.num_loc,
        "estimated_pose_fallback_count": fallback_count,
    }
    path = context["output_dir"] / "pipeline_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _run(args, context):
    feature_paths = ensure_features(context)
    reference_model = ensure_reference_model(context, args, feature_paths["local"])
    retrieval_path, pairs = make_retrieval_pairs(context, args, feature_paths["global"])
    matcher_conf = match_features.confs["superpoint+lightglue"]
    matches = context["output_dir"] / f"{matcher_conf['output']}-megaloc{args.num_loc}.h5"
    match_features.main(
        matcher_conf,
        retrieval_path,
        feature_paths["local"],
        matches=matches,
    )

    results = context["output_dir"] / "query_poses.txt"
    localize_sfm.main(
        reference_model,
        context["query_file"],
        retrieval_path,
        feature_paths["local"],
        matches,
        results,
        covisibility_clustering=False,
    )
    evaluate(results, context, context["output_dir"] / "evaluation.json")
    write_run_summary(context, args, pairs, feature_paths)
    (context["output_dir"] / "stage_complete.json").write_text(
        json.dumps({"stage": "baseline", "query_count": len(context["query_names"])}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    logger.info("KingsCollege Baseline poses: %s", results)


def run(args):
    """Run SuperPoint + LightGlue and MegaLoc against the fixed full SfM model."""
    context = prepare_run(args)
    file_handler = setup_run_logging(context)
    try:
        log_run_context(context)
        _run(args, context)
    finally:
        close_run_logging(file_handler)


def add_common_arguments(parser, output_name):
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
                        help="Independent HLoc feature and re-triangulated-model cache")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="JSON split manifest with query/reference arrays")
    parser.add_argument("--query-list", type=Path, default=None,
                        help="Explicit query image-name list; pass together with --reference-list")
    parser.add_argument("--reference-list", type=Path, default=None,
                        help="Explicit retrieval-candidate image-name list")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "kingscollege_seq1" / output_name)
    parser.add_argument("--num-covis", type=int, default=20,
                        help="Recorded map covisibility setting; the supplied full model is reused unchanged")
    parser.add_argument("--num-loc", type=int, default=10)
    return parser


def main(argv=None):
    parser = add_common_arguments(argparse.ArgumentParser(description=__doc__), "baseline")
    args = parser.parse_args(argv)
    if args.num_covis < 1 or args.num_loc < 1:
        parser.error("--num-covis and --num-loc must be positive")
    run(args)


if __name__ == "__main__":
    main()
