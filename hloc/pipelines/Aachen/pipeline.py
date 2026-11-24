import argparse
import json
from pathlib import Path
from pprint import pformat

from ... import (
    colmap_from_nvm,
    extract_features,
    localize_sfm,
    logger,
    match_features,
    pairs_from_covisibility,
    pairs_from_retrieval,
    triangulation,
)
from ...utils.read_write_model import read_cameras_binary, read_images_binary

ROOT = Path(__file__).resolve().parents[3]


def read_names(path):
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    if not names or len(names) != len(set(names)):
        raise ValueError(f"Image list must be non-empty and unique: {path}")
    return names


def write_query_list(model, names, output):
    """Write only query intrinsics from the shared COLMAP model."""
    images = read_images_binary(model / "images.bin")
    cameras = read_cameras_binary(model / "cameras.bin")
    by_name = {image.name: image for image in images.values()}
    lines = []
    for name in names:
        image = by_name.get(name)
        if image is None:
            raise ValueError(f"Query image is missing from the COLMAP model: {name}")
        camera = cameras[image.camera_id]
        lines.append(
            f"{name} {camera.model} {camera.width} {camera.height} "
            + " ".join(map(str, camera.params))
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def run(args):
    # Setup the paths
    dataset = args.dataset
    images = dataset / "images_upright/"

    outputs = args.outputs  # where everything will be saved
    sift_sfm = outputs / "sfm_sift"  # from which we extract the reference poses
    reference_sfm = args.sfm_model or outputs / "sfm_superpoint+lightglue"
    sfm_pairs = (
        outputs / f"pairs-db-covis{args.num_covis}.txt"
    )  # top-k most covisible in SIFT model
    loc_pairs = (
        # outputs / f"pairs-query-netvlad{args.num_loc}.txt"
        outputs / f"pairs-query-megaloc{args.num_loc}.txt"
    )  # top-k retrieved by NetVLAD
    experiment_mode = bool(args.query_list or args.reference_list or args.experiment_dir)
    if experiment_mode and not (args.query_list and args.reference_list and args.experiment_dir):
        raise ValueError("Split runs require --query-list, --reference-list, and --experiment-dir")
    result_dir = args.experiment_dir if experiment_mode else outputs
    result_dir.mkdir(parents=True, exist_ok=True)
    results = result_dir / ("query_poses.txt" if experiment_mode else f"Aachen_hloc_superpoint+lightglue_megaloc{args.num_loc}.txt")

    # list the standard configurations available
    logger.info("Configs for feature extractors:\n%s", pformat(extract_features.confs))
    logger.info("Configs for feature matchers:\n%s", pformat(match_features.confs))

    # pick one of the configurations for extraction and matching
    retrieval_conf = extract_features.confs["megaloc"]
    feature_conf = extract_features.confs["superpoint_my"]
    matcher_conf = match_features.confs["superpoint+lightglue"]

    if experiment_mode:
        query_names = read_names(args.query_list)
        reference_names = read_names(args.reference_list)
        if set(query_names) & set(reference_names):
            raise ValueError("Query and Reference lists overlap")
        model_images = read_images_binary(reference_sfm / "images.bin")
        registered = {image.name for image in model_images.values()}
        missing = (set(query_names) | set(reference_names)) - registered
        if missing:
            raise ValueError(f"Images missing from the shared COLMAP model: {sorted(missing)[:5]}")
        selected_images = sorted(set(query_names) | set(reference_names))
        features = extract_features.main(feature_conf, images, outputs, image_list=selected_images)
        global_descriptors = extract_features.main(retrieval_conf, images, outputs, image_list=selected_images)
        query_inputs = write_query_list(reference_sfm, query_names, result_dir / "queries_with_intrinsics.txt")
        loc_pairs = result_dir / f"pairs-query-megaloc{args.num_loc}.txt"
        pairs_from_retrieval.main(
            global_descriptors,
            loc_pairs,
            min(args.num_loc, len(reference_names)),
            query_list=query_names,
            db_list=reference_names,
        )
        loc_matches = match_features.main(
            matcher_conf,
            loc_pairs,
            features,
            features_ref=features,
            matches=(args.match_cache_dir or
                     ROOT.parent / "HLoc_Reloc3r_Experiments" / "cache" / "Aachen")
            / f"matches-query-megaloc{args.num_loc}.h5",
        )
        localize_sfm.main(
            reference_sfm,
            query_inputs,
            loc_pairs,
            features,
            loc_matches,
            results,
            covisibility_clustering=False,
        )
        (result_dir / "stage_complete.json").write_text(
            json.dumps({"stage": "baseline", "query_count": len(query_names)}, indent=2) + "\n",
            encoding="utf-8",
        )
        logger.info(
            "Completed Aachen split baseline: %d queries, %d references, shared full map %s",
            len(query_names), len(reference_names), reference_sfm,
        )
        return

    features = extract_features.main(feature_conf, images, outputs)

    colmap_from_nvm.main(
        dataset / "3D-models/aachen_cvpr2018_db.nvm",
        dataset / "3D-models/database_intrinsics.txt",
        dataset / "aachen.db",
        sift_sfm,
    )
    pairs_from_covisibility.main(sift_sfm, sfm_pairs, num_matched=args.num_covis)
    sfm_matches = match_features.main(
        matcher_conf, sfm_pairs, feature_conf["output"], outputs
    )

    triangulation.main(
        reference_sfm, sift_sfm, images, sfm_pairs, features, sfm_matches
    )

    global_descriptors = extract_features.main(retrieval_conf, images, outputs)
    pairs_from_retrieval.main(
        global_descriptors,
        loc_pairs,
        args.num_loc,
        query_prefix="query",
        db_model=reference_sfm,
    )
    loc_matches = match_features.main(
        matcher_conf, loc_pairs, feature_conf["output"], outputs
    )

    localize_sfm.main(
        reference_sfm,
        dataset / "queries/*_time_queries_with_intrinsics.txt",
        loc_pairs,
        features,
        loc_matches,
        results,
        covisibility_clustering=False,
    )  # not required with SuperPoint+LightGlue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(r"D:\Datasets\Aachen-Day-Night\dataset"),
        help="Path to the dataset, default: %(default)s",
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path("outputs/aachen"),
        help="Path to the output directory, default: %(default)s",
    )
    parser.add_argument(
        "--num_covis",
        type=int,
        default=20,
        help="Number of image pairs for SfM, default: %(default)s",
    )
    parser.add_argument(
        "--num_loc",
        type=int,
        # default=50,
        default=10,
        help="Number of image pairs for loc, default: %(default)s",
    )
    parser.add_argument("--query-list", type=Path, help="Split query names, one image per line")
    parser.add_argument("--reference-list", type=Path, help="Split Reference names, one image per line")
    parser.add_argument("--experiment-dir", type=Path, help="Isolated split-run output directory")
    parser.add_argument("--sfm-model", type=Path, help="Shared full COLMAP model for split runs")
    parser.add_argument("--match-cache-dir", type=Path,
                        help="Shared unmasked query/reference match cache for split runs")
    args = parser.parse_args()
    run(args)
