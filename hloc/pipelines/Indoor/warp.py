"""Compute a conservative pixel-space common region for one query/map pair.

Workflow:
  1. Read the map COLMAP model and the estimated query w2c pose.
  2. Take 3D points observed in the selected map image, and retain points in
     front of the query camera and inside its image frustum.
  3. Project the retained points into both cameras (including COLMAP radial
     distortion), form a convex hull in each image, and expand it by a margin.
  4. Save point projections/metadata and masked images. The masks are an
     approximate common-view region, not a dense geometric visibility map:
     sparse points cannot describe occlusion or every pixel boundary.

Run from any directory, e.g. with the HLoc conda environment:
  python hloc/pipelines/Indoor/warp.py
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image as PILImage, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))
from hloc.utils.read_write_model import qvec2rotmat, read_cameras_text, read_images_text, read_points3D_text


def parse_pose(path, name):
    for line in path.read_text(encoding="utf-8").splitlines():
        row = line.split()
        if row and row[0] == name:
            vals = np.asarray([float(v) for v in row[1:8]])
            return qvec2rotmat(vals[:4]), vals[4:]
    raise ValueError(f"Pose for {name!r} not found in {path}")


def project(xyz, rotation, translation, camera):
    """COLMAP projection for PINHOLE/SIMPLE_PINHOLE and radial models."""
    cam = xyz @ rotation.T + translation
    z = cam[:, 2]
    xy = cam[:, :2] / np.maximum(z[:, None], 1e-12)
    p, model = camera.params, camera.model
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        f, cx, cy = p[:3]
        if model == "SIMPLE_RADIAL":
            r2 = np.sum(xy**2, axis=1)
            xy *= (1 + p[3] * r2)[:, None]
        elif model == "RADIAL":
            r2 = np.sum(xy**2, axis=1)
            xy *= (1 + p[3] * r2 + p[4] * r2**2)[:, None]
        uv = np.column_stack((f * xy[:, 0] + cx, f * xy[:, 1] + cy))
    elif model == "PINHOLE":
        fx, fy, cx, cy = p[:4]
        uv = np.column_stack((fx * xy[:, 0] + cx, fy * xy[:, 1] + cy))
    else:
        raise ValueError(f"Unsupported COLMAP camera model {model}; add projection support first")
    return uv, z


def hull_mask(points, width, height, margin):
    """Rasterize the hull and quickly expand it by ``margin`` pixels."""
    # OpenCV's optimized dilation is much faster than a Python/PIL large
    # max-filter when this is repeated for 76 * 10 image pairs.
    try:
        import cv2
    except ImportError:
        cv2 = None

    pts = np.unique(np.round(points).astype(int), axis=0)
    if len(pts) < 3:
        raise ValueError("Fewer than three valid projected points; cannot form region")
    if cv2 is not None:
        hull = cv2.convexHull(pts.astype(np.int32)).reshape(-1, 1, 2)
        base = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(base, hull, 255)
    else:
        from scipy.spatial import ConvexHull
        hull = pts[ConvexHull(pts).vertices]
        base_image = PILImage.new("L", (width, height), 0)
        ImageDraw.Draw(base_image).polygon([tuple(p) for p in hull], fill=255)
        base = np.asarray(base_image)
    if margin:
        if cv2 is not None:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
            base = cv2.dilate(base, kernel)
        else:
            from scipy.ndimage import maximum_filter
            base = maximum_filter(base, size=2 * margin + 1)
    return base > 0


def load_descriptors(path):
    """Read HLoc global descriptor groups as an ordered name -> vector map."""
    with h5py.File(path, "r") as f:
        return {name: np.asarray(f[name]["global_descriptor"], dtype=np.float32)
                for name in sorted(f.keys())}


def process_pair(args, query_name, map_name, output_dir, model_data, query_camera):
    images, points, cameras = model_data
    map_image = next((im for im in images.values() if im.name == map_name), None)
    if map_image is None:
        raise ValueError(f"Map image {map_name} missing from map COLMAP model")
    map_cam = cameras[map_image.camera_id]
    qR, qt = parse_pose(args.poses, query_name)
    mR = qvec2rotmat(map_image.qvec)
    ids = np.unique(map_image.point3D_ids[map_image.point3D_ids >= 0])
    xyz = np.asarray([points[int(pid)].xyz for pid in ids], dtype=float)
    quv, qz = project(xyz, qR, qt, query_camera)
    inside = (qz > 0) & (quv[:, 0] >= 0) & (quv[:, 0] < query_camera.width) & (quv[:, 1] >= 0) & (quv[:, 1] < query_camera.height)
    ids, xyz, quv = ids[inside], xyz[inside], quv[inside]
    muv, mz = project(xyz, mR, map_image.tvec, map_cam)
    inside_map = (mz > 0) & (muv[:, 0] >= 0) & (muv[:, 0] < map_cam.width) & (muv[:, 1] >= 0) & (muv[:, 1] < map_cam.height)
    ids, xyz, quv, muv = ids[inside_map], xyz[inside_map], quv[inside_map], muv[inside_map]
    output_dir.mkdir(parents=True, exist_ok=True)
    record = {"query": query_name, "map": map_name, "points_in_map_image": int(len(np.unique(map_image.point3D_ids[map_image.point3D_ids >= 0]))),
              "shared_frustum_points": int(len(ids)), "margin_px": args.margin}
    if len(ids) >= 3:
        qmask = hull_mask(quv, query_camera.width, query_camera.height, args.margin)
        mmask = hull_mask(muv, map_cam.width, map_cam.height, args.margin)
        np.savez_compressed(output_dir / f"{Path(query_name).stem}_{Path(map_name).stem}_points.npz",
                            point3D_ids=ids, xyz=xyz, query_xy=quv, map_xy=muv)
        for label, root, name, mask in (("query", args.query_root, query_name, qmask),
                                        ("map", args.map_root, map_name, mmask)):
            image = PILImage.open(root / "images" / name).convert("RGB")
            if image.size != (mask.shape[1], mask.shape[0]):
                raise ValueError(f"Image size/model mismatch for {name}: {image.size}")
            out = np.asarray(image).copy()
            out[~mask] = 0
            PILImage.fromarray(out).save(output_dir / f"{label}_{name}")
        record.update(query_mask_pixels=int(qmask.sum()), map_mask_pixels=int(mmask.sum()))
    else:
        record["warning"] = "Fewer than three shared points; projections saved without hull mask"
        np.savez_compressed(output_dir / f"{Path(query_name).stem}_{Path(map_name).stem}_points.npz",
                            point3D_ids=ids, xyz=xyz, query_xy=quv, map_xy=muv)
    (output_dir / f"{Path(map_name).stem}_metadata.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map-root", type=Path, default=Path(r"D:\Datasets\Indoor_map_51"))
    ap.add_argument("--query-root", type=Path, default=Path(r"D:\Datasets\Indoor_query_38_38"))
    ap.add_argument("--poses", type=Path, default=ROOT / "output/reloc3r_Indoor127_megaloc/query_poses.txt")
    ap.add_argument("--query", default=None, help="process one query; by default process every query descriptor")
    ap.add_argument("--map", dest="map_name", default=None, help="with --query, optionally force one map image")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--margin", type=int, default=50, help="pixel expansion for pose/reconstruction error")
    ap.add_argument("--output", type=Path, default=ROOT / "output/Indoor_127_covis")
    args = ap.parse_args()

    model = args.map_root / "sparse/0"
    model_data = (read_images_text(model / "images.txt"), read_points3D_text(model / "points3D.txt"), read_cameras_text(model / "cameras.txt"))
    # This file contributes intrinsics only. Never read query sparse/0/images.txt:
    # its poses (if present) are ground truth and must not affect this estimate.
    query_camera = next(iter(read_cameras_text((args.query_root / "sparse/0/cameras.txt")).values()))
    # The query extrinsics are explicitly the estimated COLMAP w2c pose from
    # reloc3r_Indoor127_megaloc/query_poses.txt (selected by --poses).
    map_feats = load_descriptors(args.map_root / "hloc_outputs/global-feats-megaloc.h5")
    query_feats = load_descriptors(args.query_root / "hloc_outputs/global-feats-megaloc.h5")
    map_names = list(map_feats)
    map_matrix = np.stack([map_feats[n] for n in map_names])
    map_matrix /= np.maximum(np.linalg.norm(map_matrix, axis=1, keepdims=True), 1e-12)
    queries = [args.query] if args.query else list(query_feats)
    args.output.mkdir(parents=True, exist_ok=True)
    all_results = []
    for query_name in queries:
        qvec = query_feats[query_name]
        qvec = qvec / max(float(np.linalg.norm(qvec)), 1e-12)
        scores = map_matrix @ qvec
        ranked = np.argsort(-scores)[:args.top_k]
        candidates = [(map_names[i], float(scores[i])) for i in ranked]
        if args.map_name:
            candidates = [(args.map_name, float(scores[map_names.index(args.map_name)]))]
        qdir = args.output / Path(query_name).stem
        qdir.mkdir(parents=True, exist_ok=True)
        records = []
        for rank, (map_name, score) in enumerate(candidates, 1):
            # Each pair gets its own directory: both output images correspond
            # to this exact query/map pair and can never overwrite one another.
            pair_dir = qdir / f"rank{rank:02d}_{Path(map_name).stem}"
            record = process_pair(args, query_name, map_name, pair_dir, model_data, query_camera)
            record.update(rank=rank, cosine_similarity=score)
            (pair_dir / f"{Path(map_name).stem}_metadata.json").write_text(
                json.dumps(record, indent=2), encoding="utf-8")
            records.append(record)
        (qdir / "retrieval_top10.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        all_results.extend(records)
        print(f"{query_name}: processed {len(records)} map images")
    (args.output / "retrieval_summary.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
