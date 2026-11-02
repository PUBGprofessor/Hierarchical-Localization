"""Indoor 图像对共视区域计算和可视化脚本。

本脚本从 MegaLoc 全局特征中为每张 query 检索 top-k map 图，然后对每个
(query, map) 对执行以下步骤：

1. 读取 map COLMAP 模型以及 reloc3r 提供的 query world-to-camera (w2c) 位姿；
2. 取 map 图像实际观测到的 3D 点，投影到 query 相机并按正深度、图像边界
   筛选，得到同时可能被 query 看见的点；
3. 将这些共享点分别投影到 query 和 map 像素平面；
4. 在两幅图的投影点上分别构造凸包，并向外扩展 ``margin`` 像素；
5. 保存共享点投影、统计 JSON，以及仅保留共视掩膜内像素的两幅图像。

这里的外参只来自 ``--poses`` 指定的估计位姿文件，不读取 query SfM 真值
位姿。凸包是由稀疏 3D 点形成的区域近似，不是逐像素深度/遮挡可见性图。

本文件用于离线批处理和观察裁剪效果；HLoc 定位主流程使用的是
``hloc/covis_warp.py`` 与 ``hloc/match_covis_features.py``。

示例：
  python hloc/pipelines/Indoor/warp.py --query frame00001.png --top-k 10
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
    """从位姿文本中读取指定图像的 COLMAP w2c 旋转和平移。

    每条有效记录格式为 ``图像名 qw qx qy qz tx ty tz``。COLMAP 四元数
    顺序为 wxyz，``qvec2rotmat`` 将其转换为旋转矩阵 R；配合平移 t，
    满足 ``X_camera = R @ X_world + t``。query 的这组外参来自估计位姿文件。
    """
    for line in path.read_text(encoding="utf-8").splitlines():
        row = line.split()
        if row and row[0] == name:
            vals = np.asarray([float(v) for v in row[1:8]])
            return qvec2rotmat(vals[:4]), vals[4:]
    raise ValueError(f"Pose for {name!r} not found in {path}")


def project(xyz, rotation, translation, camera):
    """将世界坐标 3D 点投影到 COLMAP 相机的像素平面。

    参数 ``rotation`` 和 ``translation`` 是 w2c 外参；``camera`` 提供
    相机模型、图像尺寸和内参。

    返回：
      - ``uv``：N 个点的像素坐标，shape 为 (N, 2)；
      - ``z``：点在相机坐标系下的深度，后续用它排除相机后方的点。

    当前实现支持 SIMPLE_PINHOLE、PINHOLE、SIMPLE_RADIAL 和 RADIAL，
    并按 COLMAP 参数顺序计算径向畸变。其他相机模型会显式报错，避免
    在不支持的畸变模型上生成看似正常但位置错误的掩膜。
    """
    # 先将世界点变换到相机坐标系。对行向量点云，等价写法为 X @ R.T + t。
    cam = xyz @ rotation.T + translation
    z = cam[:, 2]
    # 透视除法得到归一化相机平面坐标；深度非正点稍后由调用者过滤。
    xy = cam[:, :2] / np.maximum(z[:, None], 1e-12)
    p, model = camera.params, camera.model
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        # 单焦距模型：参数开头为 f, cx, cy，随后可能带径向畸变系数。
        f, cx, cy = p[:3]
        if model == "SIMPLE_RADIAL":
            r2 = np.sum(xy**2, axis=1)
            xy *= (1 + p[3] * r2)[:, None]
        elif model == "RADIAL":
            r2 = np.sum(xy**2, axis=1)
            xy *= (1 + p[3] * r2 + p[4] * r2**2)[:, None]
        uv = np.column_stack((f * xy[:, 0] + cx, f * xy[:, 1] + cy))
    elif model == "PINHOLE":
        # 双焦距无畸变模型：fx, fy, cx, cy。
        fx, fy, cx, cy = p[:4]
        uv = np.column_stack((fx * xy[:, 0] + cx, fy * xy[:, 1] + cy))
    else:
        raise ValueError(f"Unsupported COLMAP camera model {model}; add projection support first")
    return uv, z


def hull_mask(points, width, height, margin):
    """将一组投影像素点变成外扩后的布尔共视掩膜。

    先计算二维凸包并填充为二值图，再做形态学膨胀。返回数组 shape 是
    (height, width)，True 表示保留该像素，False 表示输出时置黑。
    ``margin`` 是像素半径，用于补偿估计位姿、重建和稀疏采样误差。
    """
    # 优先使用 OpenCV 的 C++ 凸包填充和膨胀；离线批量处理很多图像对时
    # 比 Python 层逐像素操作快。无 OpenCV 时使用 SciPy/PIL 回退路径。
    try:
        import cv2
    except ImportError:
        cv2 = None

    pts = np.unique(np.round(points).astype(int), axis=0)
    if len(pts) < 3:
        # 少于 3 个不同像素点不能可靠构造二维区域，交给调用方处理。
        raise ValueError("Fewer than three valid projected points; cannot form region")
    if cv2 is not None:
        # 求投影点凸包，并把凸多边形栅格化为 0/255 掩膜。
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
            # 椭圆结构元素近似以 margin 为半径向周边膨胀区域。
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
            base = cv2.dilate(base, kernel)
        else:
            from scipy.ndimage import maximum_filter
            base = maximum_filter(base, size=2 * margin + 1)
    return base > 0


def load_descriptors(path):
    """读取 HLoc 全局描述子，返回按图像名排序的 ``name -> vector`` 字典。

    每个 HDF5 图像组应包含 ``global_descriptor``。main() 用它计算 query
    与 map 的点积相似度以选 top-k 候选；它不是局部特征，也不会直接参与
    LightGlue 匹配。
    """
    # HLoc 的 H5 会按图像相对路径建立组。若图像名为 db/1003.jpg，
    # 根层只有 db 组；必须递归取得完整键名，不能把根层键直接当图像组。
    from hloc.utils.io import list_h5_names

    names = sorted(list_h5_names(path))
    descriptors = {}
    with h5py.File(path, "r") as f:
        for name in names:
            group = f[name]
            if "global_descriptor" not in group:
                raise KeyError(
                    f"HDF5 group {name!r} in {path} has no 'global_descriptor'; "
                    "expected one descriptor group per image"
                )
            descriptors[name] = np.asarray(group["global_descriptor"], dtype=np.float32)
    return descriptors


def process_pair(args, query_name, map_name, output_dir, model_data, query_camera):
    """处理单个 query/map 对，保存共视点投影和两张黑底掩膜图。

    ``model_data`` 是 map COLMAP 的 images、points3D、cameras 字典；
    ``query_camera`` 只提供 query 内参，query 外参由 ``args.poses`` 读取。
    返回的 record 用于写入该 map 候选的 JSON 元数据。
    """
    images, points, cameras = model_data
    map_image = next((im for im in images.values() if im.name == map_name), None)
    if map_image is None:
        raise ValueError(f"Map image {map_name} missing from map COLMAP model")
    map_cam = cameras[map_image.camera_id]
    # query 外参使用 reloc3r 估计值；map 外参使用 map SfM 注册位姿。
    qR, qt = parse_pose(args.poses, query_name)
    mR = qvec2rotmat(map_image.qvec)
    # map_image.point3D_ids 与该图像的 2D 特征观测一一对应；-1 表示该
    # 特征没有三角化关联。只保留有效 ID，并对重复 ID 去重。
    ids = np.unique(map_image.point3D_ids[map_image.point3D_ids >= 0])
    xyz = np.asarray([points[int(pid)].xyz for pid in ids], dtype=float)

    # 第一阶段：把 map 图像中的 3D 点投到 query，做 query 视锥筛选。
    quv, qz = project(xyz, qR, qt, query_camera)
    inside = (qz > 0) & (quv[:, 0] >= 0) & (quv[:, 0] < query_camera.width) & (quv[:, 1] >= 0) & (quv[:, 1] < query_camera.height)
    ids, xyz, quv = ids[inside], xyz[inside], quv[inside]

    # 第二阶段：把已进入 query 视锥的点再投回 map 图像，排除 map 相机
    # 后方或 map 图像边界以外的点。剩余点才是本对图像共同有效的 3D 点。
    muv, mz = project(xyz, mR, map_image.tvec, map_cam)
    inside_map = (mz > 0) & (muv[:, 0] >= 0) & (muv[:, 0] < map_cam.width) & (muv[:, 1] >= 0) & (muv[:, 1] < map_cam.height)
    ids, xyz, quv, muv = ids[inside_map], xyz[inside_map], quv[inside_map], muv[inside_map]
    output_dir.mkdir(parents=True, exist_ok=True)
    record = {"query": query_name, "map": map_name, "points_in_map_image": int(len(np.unique(map_image.point3D_ids[map_image.point3D_ids >= 0]))),
              "shared_frustum_points": int(len(ids)), "margin_px": args.margin}
    if len(ids) >= 3:
        # 在两幅图各自的像素坐标中独立生成凸包掩膜。两个掩膜形状可能
        # 因两相机图像尺寸不同而不同，但代表同一组 3D 点的投影区域。
        qmask = hull_mask(quv, query_camera.width, query_camera.height, args.margin)
        mmask = hull_mask(muv, map_cam.width, map_cam.height, args.margin)
        # NPZ 保存共享 3D 点 ID、世界坐标及其 query/map 像素投影，便于
        # 后续检查几何筛选结果，不保存图像特征描述子。
        np.savez_compressed(output_dir / f"{Path(query_name).stem}_{Path(map_name).stem}_points.npz",
                            point3D_ids=ids, xyz=xyz, query_xy=quv, map_xy=muv)
        for label, root, name, mask in (("query", args.query_root, query_name, qmask),
                                        ("map", args.map_root, map_name, mmask)):
            image = PILImage.open(root / "images" / name).convert("RGB")
            if image.size != (mask.shape[1], mask.shape[0]):
                raise ValueError(f"Image size/model mismatch for {name}: {image.size}")
            # 掩膜内像素保留原 RGB 值，区域外全部设为黑色。
            out = np.asarray(image).copy()
            out[~mask] = 0
            # name 可能是 db/1003.jpg；输出文件放在当前 pair 目录，不能
            # 直接把斜杠拼入文件名，否则会被解释成未创建的子目录。
            output_name = f"{label}_{Path(name).name}"
            PILImage.fromarray(out).save(output_dir / output_name)
        record.update(query_mask_pixels=int(qmask.sum()), map_mask_pixels=int(mmask.sum()))
    else:
        # 点数不足时仍保存空/少量点投影和 warning，但不输出掩膜 PNG，
        # 因为无法得到稳定的二维凸包区域。
        record["warning"] = "Fewer than three shared points; projections saved without hull mask"
        np.savez_compressed(output_dir / f"{Path(query_name).stem}_{Path(map_name).stem}_points.npz",
                            point3D_ids=ids, xyz=xyz, query_xy=quv, map_xy=muv)
    (output_dir / f"{Path(map_name).stem}_metadata.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def main():
    """命令行入口：加载模型/特征、检索候选并逐对输出结果。"""
    ap = argparse.ArgumentParser(description=__doc__)
    # 数据集根目录。两者均应包含 images/、sparse/0/ 和 hloc_outputs/。
    ap.add_argument("--map-root", type=Path, default=Path(r"D:\Datasets\Indoor_map_51"))
    ap.add_argument("--query-root", type=Path, default=Path(r"D:\Datasets\Indoor_query_38_38"))
    ap.add_argument("--poses", type=Path, default=ROOT / "output/reloc3r_Indoor127_megaloc/query_poses.txt")
    # 默认处理 query 特征文件里的全部 query；指定 --query 时只处理一张。
    ap.add_argument("--query", default=None, help="process one query; by default process every query descriptor")
    # 与 --query 一起使用时可固定 map 候选，跳过 top-k 检索选择。
    ap.add_argument("--map", dest="map_name", default=None, help="with --query, optionally force one map image")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--margin", type=int, default=80, help="pixel expansion for pose/reconstruction error")
    ap.add_argument("--output", type=Path, default=ROOT / "output/Indoor_127_covis")
    args = ap.parse_args()

    model = args.map_root / "sparse/0"
    # COLMAP 文本模型：images 提供 map 位姿及 2D-3D 关联，points3D 提供
    # 世界坐标，cameras 提供 map 相机内参。
    model_data = (read_images_text(model / "images.txt"), read_points3D_text(model / "points3D.txt"), read_cameras_text(model / "cameras.txt"))
    # 此处只使用 query 相机内参。不要从 query sparse/0/images.txt 取外参，
    # 因为其中可能是真值位姿，不能用于本流程的共视估计。
    # query cameras.txt 只取内参；query 真值外参完全不用。
    query_camera = next(iter(read_cameras_text((args.query_root / "sparse/0/cameras.txt")).values()))
    # query 外参明确来自 --poses 指定的 reloc3r 估计位姿文件，格式遵循
    # COLMAP w2c 约定；默认文件为 reloc3r_Indoor127_megaloc/query_poses.txt。
    map_feats = load_descriptors(args.map_root / "hloc_outputs/global-feats-megaloc.h5")
    query_feats = load_descriptors(args.query_root / "hloc_outputs/global-feats-megaloc.h5")
    map_names = list(map_feats)
    # L2 归一化后，点积等于余弦相似度。map_matrix 的行顺序与 map_names 对齐。
    map_matrix = np.stack([map_feats[n] for n in map_names])
    map_matrix /= np.maximum(np.linalg.norm(map_matrix, axis=1, keepdims=True), 1e-12)
    queries = [args.query] if args.query else list(query_feats)
    args.output.mkdir(parents=True, exist_ok=True)
    all_results = []
    for query_name in queries:
        # 取当前 query 的 MegaLoc 向量并归一化，再与所有 map 向量计算相似度。
        qvec = query_feats[query_name]
        qvec = qvec / max(float(np.linalg.norm(qvec)), 1e-12)
        scores = map_matrix @ qvec
        ranked = np.argsort(-scores)[:args.top_k]
        candidates = [(map_names[i], float(scores[i])) for i in ranked]
        if args.map_name:
            # 手动指定 map 候选时只处理该图，并仍记录它的全局相似度分数。
            candidates = [(args.map_name, float(scores[map_names.index(args.map_name)]))]
        qdir = args.output / Path(query_name).stem
        qdir.mkdir(parents=True, exist_ok=True)
        records = []
        for rank, (map_name, score) in enumerate(candidates, 1):
            # 每个 query-map 对单独建目录，确保两个掩膜图和点投影一一对应，
            # 不会因多个 map 候选共用 query 名而相互覆盖。
            pair_dir = qdir / f"rank{rank:02d}_{Path(map_name).stem}"
            record = process_pair(args, query_name, map_name, pair_dir, model_data, query_camera)
            record.update(rank=rank, cosine_similarity=score)
            # 单对 JSON 保存共视点数、掩膜面积、top-k 排名和检索分数。
            (pair_dir / f"{Path(map_name).stem}_metadata.json").write_text(
                json.dumps(record, indent=2), encoding="utf-8")
            records.append(record)
        (qdir / "retrieval_top10.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        all_results.extend(records)
        print(f"{query_name}: processed {len(records)} map images")
    # 根目录汇总全部 query-map pair 的元数据，便于一次性统计和筛查。
    (args.output / "retrieval_summary.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
