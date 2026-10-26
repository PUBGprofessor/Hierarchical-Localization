"""计算一个 query/map 图像对的保守像素共视区域。

本文件同时承担两类职责：

1. ``compute_pair_masks`` 供定位主流程调用，返回一对布尔掩膜；
2. 文件底部的批处理入口用于生成可视化图片、投影点和元数据。

共视裁剪的核心算法是：读取 map 图像观测到的 3D 点，使用 query 的
估计 world-to-camera 位姿把点投影到 query 图像，筛掉深度为负或落在
图像外的点；再把剩余点投影到 map 图像。每幅图像上的投影点形成凸包，
并用像素边距做快速膨胀，得到送入局部匹配器的区域。

注意：这里的 query 位姿来自 reloc3r 的估计结果，不读取 query
``sparse/0/images.txt`` 中的真值位姿。稀疏点凸包是区域近似，不能表达
遮挡关系或每一个像素的真实可见性。

Run from any directory, e.g. with the HLoc conda environment:
  python hloc/pipelines/Indoor/warp.py
"""

import argparse
from functools import lru_cache
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
    """从文本位姿文件读取指定图像的 COLMAP w2c 位姿。

    文件行格式为 ``name qw qx qy qz tx ty tz``。四元数描述旋转
    ``R``，平移向量为 ``t``，满足 ``X_camera = R X_world + t``。
    """
    for line in path.read_text(encoding="utf-8").splitlines():
        row = line.split()
        if row and row[0] == name:
            vals = np.asarray([float(v) for v in row[1:8]])
            return qvec2rotmat(vals[:4]), vals[4:]
    raise ValueError(f"Pose for {name!r} not found in {path}")


def project(xyz, rotation, translation, camera):
    """按 COLMAP 相机模型把世界坐标 3D 点投影到像素坐标。

    返回 ``(uv, depth)``：``uv`` 为 ``N x 2`` 像素坐标，``depth`` 为
    相机坐标系中的 Z。这里实现了本数据使用的 SIMPLE_RADIAL、RADIAL、
    SIMPLE_PINHOLE 和 PINHOLE 模型，并保留 COLMAP 的径向畸变计算。
    """
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
    """把投影点凸包栅格化，并快速向外扩展 ``margin`` 个像素。

    优先使用 OpenCV 的凸包填充和椭圆形膨胀；没有 OpenCV 时才使用
    SciPy/PIL 回退实现。这个函数只生成二值区域，不修改原图像。
    """
    # 整个数据集有数百个图像对，因此优先选择 OpenCV 的底层实现，避免
    # 对每个 1280x720 掩膜使用较慢的 Python/PIL 大窗口滤波。
    try:
        import cv2
    except ImportError:
        cv2 = None

    pts = np.unique(np.round(points).astype(int), axis=0)
    if len(pts) < 3:
        raise ValueError("Fewer than three valid projected points; cannot form region")
    if cv2 is not None:
        # cv2.convexHull 返回凸包顶点；fillConvexPoly 将其变成像素掩膜。
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
        # 膨胀是对位姿误差、重建误差和稀疏采样误差的宽松补偿。
        if cv2 is not None:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
            base = cv2.dilate(base, kernel)
        else:
            from scipy.ndimage import maximum_filter
            base = maximum_filter(base, size=2 * margin + 1)
    return base > 0


def load_descriptors(path):
    """读取 HLoc 全局描述子，返回 ``图像名 -> 向量`` 字典。

    MegaLoc 特征文件的每个 HDF5 group 包含 ``global_descriptor``；这里
    只用于批处理入口的 top-k 检索，主定位流程复用已经生成的配对文件。
    """
    with h5py.File(path, "r") as f:
        return {name: np.asarray(f[name]["global_descriptor"], dtype=np.float32)
                for name in sorted(f.keys())}


def compute_pair_masks(query_name, map_name, poses, map_model, query_camera, margin=50):
    """计算一个 query/map 对的共视掩膜。

    ``query_camera`` 只提供 query 内参；query 外参始终从 ``poses`` 读取。
    返回 ``(query_mask, map_mask)``，或者在没有任何共享 3D 点时返回
    ``None``。返回 ``None`` 是重要的语义：该候选 map 图像与 query
    没有几何支持，不应送入 LightGlue 或 PnP。
    """
    map_model = Path(map_model).resolve()
    images, points, cameras = _read_model_cached(str(map_model))
    map_image = next((im for im in images.values() if im.name == map_name), None)
    if map_image is None:
        raise ValueError(f"Map image {map_name} not found in {map_model}")
    map_camera = cameras[map_image.camera_id]
    qR, qt = parse_pose(Path(poses), query_name)
    mR = qvec2rotmat(map_image.qvec)
    # map_image.point3D_ids 是 COLMAP 的 2D 观测到 3D 点 ID 的关联表。
    # -1 表示该像素没有有效 3D 点，因此先排除。
    ids = np.unique(map_image.point3D_ids[map_image.point3D_ids >= 0])
    xyz = np.asarray([points[int(pid)].xyz for pid in ids], dtype=float)
    quv, qz = project(xyz, qR, qt, query_camera)
    # query 视锥判定：深度必须为正，且投影点必须落在 query 图像边界内。
    keep = (qz > 0) & (quv[:, 0] >= 0) & (quv[:, 0] < query_camera.width) & (quv[:, 1] >= 0) & (quv[:, 1] < query_camera.height)
    xyz, quv = xyz[keep], quv[keep]
    muv, mz = project(xyz, mR, map_image.tvec, map_camera)
    keep = (mz > 0) & (muv[:, 0] >= 0) & (muv[:, 0] < map_camera.width) & (muv[:, 1] >= 0) & (muv[:, 1] < map_camera.height)
    quv, muv = quv[keep], muv[keep]
    # 0 个共享点表示该候选图与 query 没有几何共视支持，调用方应直接
    # 删除该候选，而不是用全图掩膜把无关图像送入 PnP。
    if len(quv) == 0:
        return None
    # 1~2 个点不能稳定构造凸包。此时保留候选，但采用全图掩膜，让
    # LightGlue 按原始 HLoc 方式尝试匹配；只有真正的 0 点才会被过滤。
    if len(quv) < 3:
        return np.ones((query_camera.height, query_camera.width), bool), np.ones((map_camera.height, map_camera.width), bool)
    try:
        qmask = hull_mask(quv, query_camera.width, query_camera.height, margin)
        mmask = hull_mask(muv, map_camera.width, map_camera.height, margin)
    except Exception:
        return np.ones((query_camera.height, query_camera.width), bool), np.ones((map_camera.height, map_camera.width), bool)
    return qmask, mmask


@lru_cache(maxsize=2)
def _read_model_cached(model_path):
    """缓存 map 文本 SfM 模型，避免每个 query/map 对重复解析大文件。"""
    model_path = Path(model_path)
    return (read_images_text(model_path / "images.txt"),
            read_points3D_text(model_path / "points3D.txt"),
            read_cameras_text(model_path / "cameras.txt"))


def process_pair(args, query_name, map_name, output_dir, model_data, query_camera):
    """为一个图像对生成可视化中间结果。

    该函数与 pipeline 中的 ``compute_pair_masks`` 使用同一套投影逻辑，
    但额外保存 3D 点 ID、两幅图上的投影坐标、黑底共视图和 JSON 统计。
    它主要用于检查裁剪效果，不参与 HLoc 的 PnP 计算。
    """
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
        # 只有至少三个非共线点时，凸包才有稳定的二维面积。
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
    """批量执行 MegaLoc top-k 检索和 query/map 共视区域可视化。"""
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
    # 归一化后点积即余弦相似度；每张 query 取相似度最高的 top-k map 图。
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
            # 每个 query-map 对单独一个目录，保证两幅输出图严格成对，
            # 也避免同一 query 的 map 结果互相覆盖。
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
