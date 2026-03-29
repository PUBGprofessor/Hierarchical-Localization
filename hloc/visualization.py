import pickle
import random

import numpy as np
import pycolmap
from matplotlib import cm

from .utils.io import read_image
from .utils.viz import add_text, cm_RdGn, plot_images, plot_keypoints, plot_matches


def visualize_sfm_2d(
    reconstruction, image_dir, color_by="visibility", selected=[], n=1, seed=0, dpi=75
):
    assert image_dir.exists()
    if not isinstance(reconstruction, pycolmap.Reconstruction):
        reconstruction = pycolmap.Reconstruction(reconstruction)

    if not selected:
        # 随机选择 n 张图像进行可视化
        image_ids = list(reconstruction.reg_image_ids())
        selected = random.Random(seed).sample(image_ids, min(n, len(image_ids)))

    for i in selected:
        image = reconstruction.images[i]
        # 获取图像中的所有2D特征点坐标
        keypoints = np.array([p.xy for p in image.points2D])
        # 获取每个2D点是否有对应的3D点
        visible = np.array([p.has_point3D() for p in image.points2D])

        if color_by == "visibility":
            # 红色表示不可见（没有对应3D点），蓝色表示可见
            color = [(0, 0, 1) if v else (1, 0, 0) for v in visible]
            text = f"visible: {np.count_nonzero(visible)}/{len(visible)}"
        elif color_by == "track_length":
            tl = np.array(
                [
                    (
                        reconstruction.points3D[p.point3D_id].track.length()
                        if p.has_point3D()
                        else 1
                    )
                    for p in image.points2D
                ]
            )
            max_, med_ = np.max(tl), np.median(tl[tl > 1])
            tl = np.log(tl)
            color = cm.jet(tl / tl.max()).tolist()
            text = f"max/median track length: {max_}/{med_}"
        elif color_by == "depth":
            p3ids = [p.point3D_id for p in image.points2D if p.has_point3D()]
            z = np.array(
                [
                    (image.cam_from_world() * reconstruction.points3D[j].xyz)[-1]
                    for j in p3ids
                ]
            )
            z -= z.min()
            color = cm.jet(z / np.percentile(z, 99.9))
            text = f"visible: {np.count_nonzero(visible)}/{len(visible)}"
            keypoints = keypoints[visible]
        else:
            raise NotImplementedError(f"Coloring not implemented: {color_by}.")

        name = image.name
        plot_images([read_image(image_dir / name)], dpi=dpi)
        plot_keypoints([keypoints], colors=[color], ps=4)
        add_text(0, text)
        add_text(0, name, pos=(0.01, 0.01), fs=5, lcolor=None, va="bottom")


def visualize_loc(
    results,
    image_dir,
    reconstruction=None,
    db_image_dir=None,
    selected=[],
    n=1,
    seed=0,
    prefix=None,
    **kwargs,
):
    assert image_dir.exists()

    with open(str(results) + "_logs.pkl", "rb") as f:
        logs = pickle.load(f)

    if not selected:
        queries = list(logs["loc"].keys())
        if prefix:
            queries = [q for q in queries if q.startswith(prefix)]
        selected = random.Random(seed).sample(queries, min(n, len(queries)))

    if reconstruction is not None:
        if not isinstance(reconstruction, pycolmap.Reconstruction):
            reconstruction = pycolmap.Reconstruction(reconstruction)

    for qname in selected:
        loc = logs["loc"][qname]
        visualize_loc_from_log(
            image_dir, qname, loc, reconstruction, db_image_dir, **kwargs
        )


def visualize_loc_from_log(
    image_dir,        # 查询图像所在的目录
    query_name,       # 查询图像的文件名
    loc,              # 定位日志字典，包含 PnP 结果、匹配点等信息
    reconstruction=None, # pycolmap.Reconstruction 对象，包含 3D 点云和相机位姿
    db_image_dir=None,   # 数据库图像目录（如果与查询图目录不同）
    top_k_db=2,       # 关键参数：指定可视化前几个匹配最好的数据库图像
    dpi=75,           # 绘图分辨率
):
    q_image = read_image(image_dir / query_name)
    
    # 如果使用了共视聚类（Co-visibility Clustering），则定位日志中包含多个簇
    if loc.get("covisibility_clustering", False):
        # 即使定位失败，也选择得分最高（best_cluster）或第一个簇进行可视化
        loc = loc["log_clusters"][loc["best_cluster"] or 0]

    # 获取 PnP 算法筛选后的内点掩码（True 表示该匹配点对定位有贡献）
    inliers = np.array(loc["PnP_ret"]["inlier_mask"])
    mkp_q = loc["keypoints_query"] # 查询图上的关键点坐标
    n = len(loc["db"])              # 涉及到的数据库图像数量
    if reconstruction is not None:
        # kp_to_3D_to_db 记录了：查询图关键点索引 -> 匹配的 3D 点 ID -> 观测到该 3D 点的数据库图索引
        kp_idxs, kp_to_3D_to_db = loc["keypoint_index_to_db"]
        counts = np.zeros(n)              # 用于统计每张数据库图拥有的内点数
        dbs_kp_q_db = [[] for _ in range(n)] # 存储 (查询图点索引, 数据库图点索引) 对
        inliers_dbs = [[] for _ in range(n)] # 存储每个匹配是否为内点
        
        for i, (inl, (p3D_id, db_idxs)) in enumerate(zip(inliers, kp_to_3D_to_db)):
            # 从 3D 重建模型中获取该 3D 点的观测轨迹（Track）
            # track 包含：哪个图像 ID 观察到了这个点，以及在该图上的 2D 点索引
            track = reconstruction.points3D[p3D_id].track
            track = {el.image_id: el.point2D_idx for el in track.elements}
            
            for db_idx in db_idxs:
                counts[db_idx] += inl # 如果是内点，则该数据库图计数加 1
                # 获取该 3D 点在当前数据库图像中对应的 2D 点索引
                kp_db = track[loc["db"][db_idx]]
                dbs_kp_q_db[db_idx].append((i, kp_db))
                inliers_dbs[db_idx].append(inl)
    # 按照内点数量从大到小排序
    db_sort = np.argsort(-counts)
    
    # 遍历前 k 个匹配最好的数据库图像
    for db_idx in db_sort[:top_k_db]:
        if reconstruction is not None:
            # 从重建模型中获取数据库图像的元数据
            db = reconstruction.images[loc["db"][db_idx]]
            db_name = db.name
            db_kp_q_db = np.array(dbs_kp_q_db[db_idx])
            
            # 准备绘图用的关键点坐标对
            kp_q = mkp_q[db_kp_q_db[:, 0]] # 查询图的点
            # 数据库图的点坐标（从 points2D 中提取）
            kp_db = np.array([db.points2D[i].xy for i in db_kp_q_db[:, 1]])
            inliers_db = inliers_dbs[db_idx]
        else:
            # ... (针对 InLoc 等不需要 Reconstruction 模型的特殊处理) ...
            db_name = loc["db"][db_idx]
            kp_q = mkp_q[loc["indices_db"] == db_idx]
            kp_db = loc["keypoints_db"][loc["indices_db"] == db_idx]
            inliers_db = inliers[loc["indices_db"] == db_idx]

        # 读取图片并开始绘图
        db_image = read_image((db_image_dir or image_dir) / db_name)
        # cm_RdGn 会根据 inlier 状态生成颜色：绿色表示内点，红色表示外点
        color = cm_RdGn(inliers_db).tolist()
        text = f"inliers: {sum(inliers_db)}/{len(inliers_db)}"

        # 将查询图和选中的数据库图并排显示
        plot_images([q_image, db_image], dpi=dpi)
        # 绘制特征点之间的匹配连线
        plot_matches(kp_q, kp_db, color, a=0.1)
        # 在图像上添加说明文字
        add_text(0, text)
        opts = dict(pos=(0.01, 0.01), fs=5, lcolor=None, va="bottom")
        add_text(0, query_name, **opts)
        add_text(1, db_name, **opts)
