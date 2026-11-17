"""InLoc 共视几何、掩膜和缓存工具。

pipeline_with_warp.py 和 warp.py 共用这里的实现，确保在线匹配和离线可视化
使用完全相同的 XYZcut 采样、坐标变换、query 投影和凸包掩膜规则。
"""

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from scipy.io import loadmat

from ...localize_inloc import get_scan_pose
from ...utils.read_write_model import qvec2rotmat


def read_initial_poses(path):
    """读取 query 的 world-to-camera 初始位姿。

    每行至少包含 ``name qw qx qy qz tx ty tz``，行尾可带额外统计字段。
    返回 ``query basename -> (R, t)``。
    """
    path = Path(path)
    poses = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        if len(fields) < 8:
            raise ValueError(f"Invalid pose at {path}:{line_number}: {line}")
        try:
            qvec = np.asarray([float(value) for value in fields[1:5]], dtype=np.float64)
            tvec = np.asarray([float(value) for value in fields[5:8]], dtype=np.float64)
        except ValueError as error:
            raise ValueError(f"Invalid pose values at {path}:{line_number}: {line}") from error
        name = Path(fields[0].replace("\\", "/")).name
        if name in poses:
            raise ValueError(f"Duplicate query basename {name!r} in {path}")
        poses[name] = (qvec2rotmat(qvec), tvec)
    if not poses:
        raise ValueError(f"No query poses found in {path}")
    return poses


def make_hull_mask(points, width, height, margin):
    """由投影点生成凸包并向外膨胀 ``margin`` 像素。"""
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
    """读取一次 cutout 的 XYZcut，返回全局 XYZ、map 像素和尺寸。"""
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
    """缓存图像尺寸，避免为每个候选重复解码 query 图像。"""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return image.shape[1], image.shape[0]


def compute_pair_geometry(dataset, query_name, map_name, initial_poses, margin=80):
    """计算 InLoc 一对 query/map 的投影点、尺寸和共视掩膜。

    算法与原 pipeline_with_warp.py 相同：读取 XYZcut、变换到全局坐标，
    用 query 初始位姿投影，筛选正深度和 query 画面内点，再由 map 原始
    XYZcut 像素和 query 投影像素分别生成掩膜。
    """
    dataset = Path(dataset)
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
    return {
        "xyz_world": xyz_world[positive][in_frame],
        "map_xy": map_xy,
        "query_xy": query_xy,
        "query_mask": query_mask,
        "map_mask": map_mask,
        "query_shape": (query_height, query_width),
        "map_shape": (map_height, map_width),
    }


def compute_pair_masks(dataset, query_name, map_name, initial_poses, margin=80):
    """返回与在线 pipeline 兼容的 ``(query_mask, map_mask)``。"""
    geometry = compute_pair_geometry(dataset, query_name, map_name, initial_poses, margin)
    if geometry is None:
        return None
    return geometry["query_mask"], geometry["map_mask"]


def mask_cache_signature(dataset, query_name, map_name, poses_path, margin):
    """生成掩膜缓存签名，输入文件或 margin 改变时自动失效。"""
    dataset = Path(dataset)
    query_path = dataset / query_name
    mat_path = Path(str(dataset / map_name) + ".mat")
    floor_name, scan_id, image_name = map_name.split("/")[-3:]
    transform_path = (
        dataset / f"{floor_name}_alignment" / "transformations"
        / f"{image_name[:3]}_trans_{scan_id}.txt"
    )
    stats = []
    for path in (query_path, mat_path, transform_path, Path(poses_path)):
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


def mask_cache_path(output_dir, query_name, map_name):
    """返回一对图像的掩膜缓存路径。"""
    key = hashlib.sha1(f"{query_name}\0{map_name}".encode("utf-8")).hexdigest()
    return Path(output_dir) / "pair_mask_cache" / f"{key}.npz"


def load_cached_masks(cache_path, signature):
    """读取签名匹配的掩膜缓存；过期或损坏时返回 None。"""
    cache_path = Path(cache_path)
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
    """压缩保存一对掩膜或无共视结果。"""
    cache_path = Path(cache_path)
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
