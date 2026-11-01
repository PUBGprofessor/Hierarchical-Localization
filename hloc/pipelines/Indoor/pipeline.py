"""Localize Indoor_query_38_38 using only Indoor_map_51 poses and query intrinsics.

Query ground-truth poses in sparse/0/images.txt are read only after localization,
inside evaluate(). This dataset has one shared query camera in cameras.txt.
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
from hloc.utils.io import list_h5_names
from hloc.utils.read_write_model import qvec2rotmat, read_cameras_text, read_images_text

ROOT = Path(__file__).resolve().parents[3]


def dataset_path(name):
    """Support datasets next to the repo and the local Datasets layout."""
    for path in (ROOT / name, ROOT.parent.parent / "Datasets" / name):
        if path.is_dir():
            return path
    return ROOT / name


def model_images(model):
    """Read a COLMAP text model; only call with query_model during evaluation."""
    path = model / "images.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Expected COLMAP text model: {path}")
    return read_images_text(path)


def prepare_queries(model, image_dir, output):
    """Build HLoc's query list with per-image intrinsics, not GT poses.

    COLMAP may assign a different camera model to every query image.  The
    image-to-camera association is read from images.txt, while the camera
    parameters themselves come from cameras.txt.  The stored qvec/tvec are
    deliberately ignored here and are only used later by evaluate().
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


def evaluate(results, query_model, query_names, output):
    """Compare predicted and GT w2c poses; report PnP and fallback separately."""
    # This is the first read of query images.txt, after localize_sfm has returned.
    ground_truth = {image.name: image for image in model_images(query_model).values()}
    with open(f"{results}_logs.pkl", "rb") as handle:
        logs = pickle.load(handle)["loc"]
    predictions = {}
    for line in results.read_text(encoding="utf-8").splitlines():
        name, *pose = line.split()
        if len(pose) != 7:
            raise ValueError(f"Invalid pose in {results}: {line}")
        predictions[name] = np.asarray(pose, dtype=float)

    # HLoc's write_poses() intentionally writes only the basename.  Aachen
    # query names contain a directory prefix (for example db/1003.jpg), so
    # evaluate both the full name and its unique basename.
    basename_to_query = {}
    for name in query_names:
        basename = Path(name).name
        if basename in basename_to_query:
            basename_to_query[basename] = None
        else:
            basename_to_query[basename] = name

    rows = []
    for name in query_names:
        image = ground_truth[name]
        prediction = predictions.get(name)
        if prediction is None:
            output_name = Path(name).name
            matched_name = basename_to_query.get(output_name)
            if matched_name == name:
                prediction = predictions.get(output_name)
        method = "missing"
        translation = rotation = None
        if prediction is not None:
            method = (
                "pnp" if logs.get(name, {}).get("PnP_ret") is not None
                else "retrieval_fallback"
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
    """Extract, build map landmarks, retrieve, localize, then evaluate."""
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
    query_list = query_output / "queries_with_intrinsics.txt"
    query_names = prepare_queries(query_model, query_images, query_list)
    if map_names & set(query_names):
        raise ValueError("Map and query image names must be distinct")

    # retrieval_conf = extract_features.confs["netvlad"]
    retrieval_conf = extract_features.confs["megaloc"]
    feature_conf = extract_features.confs["superpoint_my"]
    matcher_conf = match_features.confs["superpoint+lightglue"]
    # Keep map and query descriptors in their respective dataset directories.
    map_local = extract_features.main(feature_conf, map_images, map_output, image_list=sorted(map_names))
    map_global = extract_features.main(retrieval_conf, map_images, map_output, image_list=sorted(map_names))
    query_local = extract_features.main(feature_conf, query_images, query_output, image_list=query_names)
    query_global = extract_features.main(retrieval_conf, query_images, query_output, image_list=query_names)

    sfm_pairs = map_output / f"pairs-covis{args.num_covis}.txt"
    # Original map 2D indices are not SuperPoint indices. Match covisible map
    # images and triangulate SuperPoint landmarks at the known map poses.
    # 对 map数据集 使用superpoint+lightglue匹配共视图像对，生成sfm_pairs文件，用于map数据集的重建
    pairs_from_covisibility.main(map_model, sfm_pairs, num_matched=args.num_covis)
    if not sfm_pairs.stat().st_size:
        raise ValueError("No covisible map images; cannot triangulate landmarks")
     # 使用上一步生成的匹配对，进行特征匹配（LightGlue）。
    sfm_matches = match_features.main(
        matcher_conf, sfm_pairs, map_local,
        matches=map_output / f"{matcher_conf['output']}-covis{args.num_covis}.h5",
    )
    reference_sfm = map_output / f"sfm-{feature_conf['output']}-covis{args.num_covis}"
    if not (reference_sfm / "images.bin").is_file():
        # 如果没有Hloc产生的sfm模型，则固定相机的位姿不变，使用lightglue的共视匹配对和匹配结果，进行三角化，生成sfm模型
        triangulation.main(reference_sfm, map_model, map_images, sfm_pairs, map_local, sfm_matches)

    # loc_pairs = query_output / f"pairs-query-netvlad{args.num_loc}.txt"
    loc_pairs = query_output / f"pairs-query-megaloc{args.num_loc}.txt"
    # Retrieval sees only query descriptors and registered map images.
    # 生成定位用的“匹配对”: 对于每一张 Query 图，计算它和所有 Database 图的全局特征相似度，选出 Top-N
    pairs_from_retrieval.main(
        query_global, loc_pairs, min(args.num_loc, len(map_names)),
        query_list=query_names, db_descriptors=map_global, db_model=reference_sfm,
    )
    if not loc_pairs.stat().st_size:
        raise ValueError("No retrieval pairs; check the descriptors and map model")
    # 对 Query 图和它找到的 Top-N 个“嫌疑人”进行精细匹配
    loc_matches = match_features.main(
        matcher_conf, loc_pairs, query_local, features_ref=map_local,
        # matches=query_output / f"{matcher_conf['output']}-netvlad{args.num_loc}.h5",
        matches=query_output / f"{matcher_conf['output']}-megaloc{args.num_loc}.h5",
    )
    combined_features = query_output / f"{feature_conf['output']}-combined.h5"
    link_features(query_local, map_local, combined_features)

    results = final_output / "query_poses.txt"
    # PnP uses query intrinsics, query keypoints, map keypoints, and map 3D points.
    localize_sfm.main(
        reference_sfm, query_list, loc_pairs, combined_features, loc_matches, results,
        covisibility_clustering=False,
    )
    # 评估位姿结果
    evaluate(results, query_model, query_names, final_output / "evaluation.json")
    logger.info("Poses: %s; evaluation: %s", results, final_output / "evaluation.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-dir", type=Path, default=dataset_path("Indoor_map_51"))
    parser.add_argument("--query-dir", type=Path, default=dataset_path("Indoor_query_38_38"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "Aachen_Hloc")
    parser.add_argument("--num-covis", type=int, default=20)
    parser.add_argument("--num-loc", type=int, default=10)
    arguments = parser.parse_args()
    if arguments.num_covis < 1 or arguments.num_loc < 1:
        parser.error("--num-covis and --num-loc must be positive")
    run(arguments)
