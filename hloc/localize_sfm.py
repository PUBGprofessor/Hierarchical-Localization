import argparse
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pycolmap
from tqdm import tqdm

from . import logger
from .utils.io import get_keypoints, get_matches, write_poses
from .utils.parsers import parse_image_lists, parse_retrieval


def do_covisibility_clustering(
    frame_ids: List[int], reconstruction: pycolmap.Reconstruction
):
    clusters = []
    visited = set()
    for frame_id in frame_ids:
        # Check if already labeled
        if frame_id in visited:
            continue

        # New component
        clusters.append([])
        queue = {frame_id}
        while len(queue):
            exploration_frame = queue.pop()

            # Already part of the component
            if exploration_frame in visited:
                continue
            visited.add(exploration_frame)
            clusters[-1].append(exploration_frame)

            observed = reconstruction.images[exploration_frame].points2D
            connected_frames = {
                obs.image_id
                for p2D in observed
                if p2D.has_point3D()
                for obs in reconstruction.points3D[p2D.point3D_id].track.elements
            }
            connected_frames &= set(frame_ids)
            connected_frames -= visited
            queue |= connected_frames

    clusters = sorted(clusters, key=len, reverse=True)
    return clusters


class QueryLocalizer:
    def __init__(self, reconstruction, config=None):
        self.reconstruction = reconstruction
        self.config = config or {}

    def localize(self, points2D_all, points2D_idxs, points3D_id, query_camera):
        points2D = points2D_all[points2D_idxs]
        points3D = [self.reconstruction.points3D[j].xyz for j in points3D_id]
        if points2D.shape[0] == 0:
            return None
        ret = pycolmap.estimate_and_refine_absolute_pose(
            points2D,
            points3D,
            query_camera,
            estimation_options=self.config.get("estimation", {}),
            refinement_options=self.config.get("refinement", {}),
        )
        return ret


def pose_from_cluster(
    localizer: QueryLocalizer,       # 定位器对象，持有 3D 重建模型 (reconstruction)
    qname: str,                      # 查询图像的文件名
    query_camera: pycolmap.Camera,   # 查询图像的相机内参 (Intrinsics)
    db_ids: List[int],               # 候选参考图像的 ID 列表 (通常由图像检索步骤提供，如 NetVLAD Top-N)
    features_path: Path,             # 特征文件路径 (.h5)
    matches_path: Path,              # 匹配结果文件路径 (.h5)
    **kwargs,                        # 传递给 PnP 求解器的额外参数 (如 RANSAC 阈值)
):
    # --- 1. 准备查询图的特征点 ---
    # 从 h5 文件读取查询图的 2D 关键点坐标
    kpq = get_keypoints(features_path, qname)
    # 【重要细节】坐标系修正：
    # 深度学习提取的特征点通常以 (0,0) 为左上角像素中心
    # 而 COLMAP 算法内部假定像素中心在 (0.5, 0.5)
    # 加上 0.5 是为了对齐坐标系，防止亚像素级的系统误差
    kpq += 0.5  

    # --- 2. 初始化数据容器 ---
    # kp_idx_to_3D: 记录 [查询图关键点索引] -> [对应的 3D 点 ID 列表]
    # 这是 PnP 求解的核心输入
    kp_idx_to_3D = defaultdict(list)
    
    # kp_idx_to_3D_to_db: 记录 [查询图关键点索引] -> [3D点 ID] -> [来源参考图 ID]
    # 这个字典主要用于后续的可视化 (visualize_loc_from_log)，告诉我们这个匹配是靠哪张参考图建立的
    kp_idx_to_3D_to_db = defaultdict(lambda: defaultdict(list))
    
    num_matches = 0 # 统计总共找到了多少个有效的 2D-3D 匹配

    # --- 3. 遍历每一张参考图，构建 2D-3D 桥梁 ---
    for i, db_id in enumerate(db_ids):
        # 从 3D 模型中获取参考图像对象
        image = localizer.reconstruction.images[db_id]
        
        # 如果这张参考图没有关联任何 3D 点，那它对定位没帮助，直接跳过
        if image.num_points3D == 0:
            logger.debug(f"No 3D points found for {image.name}.")
            continue

        # 制作一张查找表：参考图的第 j 个 2D 点对应的 3D 点 ID 是多少？
        # 如果没有对应 3D 点，则标记为 -1
        points3D_ids = np.array(
            [p.point3D_id if p.has_point3D() else -1 for p in image.points2D]
        )

        # 获取 [查询图] <-> [当前参考图] 的 2D 特征匹配
        # matches 每一行是 [query_idx, db_idx]
        matches, _ = get_matches(matches_path, qname, image.name)
        
        # 【核心过滤逻辑】
        # 我们只保留那些"指向了有效 3D 点"的匹配
        # 逻辑：如果查询点匹配到了参考图的一个点，但那个参考图的点没有 3D 坐标，这个匹配对 PnP 没用，扔掉。
        matches = matches[points3D_ids[matches[:, 1]] != -1]
        
        num_matches += len(matches)

        # 将筛选后的匹配填入字典
        for idx, m in matches:
            # m 是参考图上的特征点索引
            id_3D = points3D_ids[m] 
            
            # 记录用于可视化的反向索引
            kp_idx_to_3D_to_db[idx][id_3D].append(i)
            
            # 记录用于 PnP 的核心映射
            # 避免重复：如果同一个查询点已经通过另一张参考图关联到了同一个 3D 点，不要重复添加
            if id_3D not in kp_idx_to_3D[idx]:
                kp_idx_to_3D[idx].append(id_3D)

    # --- 4. 数据扁平化 (Flattening) ---
    # 将字典转换为列表，因为底层的 C++ PnP 函数需要数组输入
    idxs = list(kp_idx_to_3D.keys()) # 有效的查询图特征点索引
    
    # mkp_idxs: 展平后的查询点索引列表
    mkp_idxs = [i for i in idxs for _ in kp_idx_to_3D[i]]
    # mp3d_ids: 对应位置的 3D 点 ID 列表
    mp3d_ids = [j for i in idxs for j in kp_idx_to_3D[i]]

    # --- 5. 执行定位 (PnP + RANSAC) ---
    # 输入：查询图 2D 点坐标，2D-3D 对应关系索引，相机内参
    # 输出：相机位姿 ret (旋转 R, 位移 t, 内点数量等)
    ret = localizer.localize(kpq, mkp_idxs, mp3d_ids, query_camera, **kwargs)
    
    # 如果定位成功，把相机内参也塞进结果里
    if ret is not None:
        ret["camera"] = query_camera

    # --- 6. 生成日志 (Logging) ---
    # 这部分主要是为了后续 debug 和可视化 (visualize_loc_from_log)
    # 构造一个详细的结构，说明每个匹配是从哪来的
    mkp_to_3D_to_db = [
        (j, kp_idx_to_3D_to_db[i][j]) for i in idxs for j in kp_idx_to_3D[i]
    ]
    log = {
        "db": db_ids,                   # 使用了哪些参考图
        "PnP_ret": ret,                 # PnP 求解的原始结果
        "keypoints_query": kpq[mkp_idxs], # 参与计算的查询图 2D 点
        "points3D_ids": mp3d_ids,       # 对应的 3D 点 ID
        "points3D_xyz": None,           # 为了节省内存，这里不存具体的 XYZ 坐标 (反正可以通过 ID 在模型里查)
        "num_matches": num_matches,     # 2D-2D 匹配的总数
        "keypoint_index_to_db": (mkp_idxs, mkp_to_3D_to_db), # 可视化所需的详细溯源信息
    }
    
    return ret, log

import pickle
import logging
from pathlib import Path
from typing import Dict, Union
from tqdm import tqdm
import pycolmap
# 假设导入了以下辅助类和函数
# from .utils.parsers import parse_image_lists, parse_retrieval
# from .utils.io import write_poses
# from .localize_sfm import QueryLocalizer, pose_from_cluster, do_covisibility_clustering

logger = logging.getLogger(__name__)

def main(
    reference_sfm: Union[Path, pycolmap.Reconstruction], # 已经建好的 3D 模型 (包含 3D 点云和参考图位姿)
    queries: Path,              # 查询图像列表文件 (通常包含内参)
    retrieval: Path,            # 检索结果文件 (pairs-query.txt, 告诉我们要去哪些参考图里找匹配)
    features: Path,             # 特征文件 (query 和 db 的特征)
    matches: Path,              # 匹配文件 (query 和 db 之间的匹配关系)
    results: Path,              # 结果输出路径 (Aachen_v1.1_hloc.txt)
    ransac_thresh: int = 12,    # RANSAC 的像素误差阈值 (PnP 时的容忍度)
    covisibility_clustering: bool = False, # 是否开启共视聚类 (SuperPoint+SuperGlue 通常设为 False)
    prepend_camera_name: bool = False,     # 输出格式选项 (是否在文件名前加相机名)
    config: Dict = None,        # 其他配置参数
):
    # 0. 基础检查
    assert retrieval.exists(), retrieval
    assert features.exists(), features
    assert matches.exists(), matches

    # 解析查询列表 (同时读取内参，因为 PnP 需要内参)
    queries = parse_image_lists(queries, with_intrinsics=True)
    # 解析检索对 (字典: Query -> [DB_Image_1, DB_Image_2...])
    retrieval_dict = parse_retrieval(retrieval)

    logger.info("Reading the 3D model...")
    # 加载 3D 重建模型 (COLMAP 格式)
    if not isinstance(reference_sfm, pycolmap.Reconstruction):
        reference_sfm = pycolmap.Reconstruction(reference_sfm)
    # 建立 "图片名 -> Image ID" 的映射，方便查找
    db_name_to_id = {img.name: i for i, img in reference_sfm.images.items()}

    # 配置 PnP 求解器参数
    config = {"estimation": {"ransac": {"max_error": ransac_thresh}}, **(config or {})}
    # 初始化定位器类 (核心工具，负责把 2D 匹配提升为 2D-3D 对应)
    localizer = QueryLocalizer(reference_sfm, config)

    cam_from_world = {} # 存储最终计算出的位姿结果
    logs = {            # 存储详细日志 (如内点数量等)，用于调试
        "features": features,
        "matches": matches,
        "retrieval": retrieval,
        "loc": {},
    }
    
    logger.info("Starting localization...")
    
    # 遍历每一张查询图像
    for qname, qcam in tqdm(queries):
        # 如果这张图没有检索到任何候选参考图，跳过
        if qname not in retrieval_dict:
            logger.warning(f"No images retrieved for query image {qname}. Skipping...")
            continue
            
        # 获取该查询图对应的候选数据库图片列表
        db_names = retrieval_dict[qname]
        db_ids = []
        for n in db_names:
            if n not in db_name_to_id:
                logger.warning(f"Image {n} was retrieved but not in database")
                continue
            db_ids.append(db_name_to_id[n])

        # 分支 A: 共视聚类 (Covisibility Clustering)
        # 这是一个针对传统特征 (SIFT) 的优化技巧。如果检索结果里混杂了地标 A 和地标 B 的图片，
        # 它会把图片分成几堆，分别尝试定位，防止 RANSAC 被混乱的数据干扰。
        # 对于 SuperPoint+SuperGlue 这种强匹配方法，通常不需要，设为 False 即可。
        if covisibility_clustering:
            clusters = do_covisibility_clustering(db_ids, reference_sfm)
            best_inliers = 0
            best_cluster = None
            logs_clusters = []
            
            # 对每个聚类尝试进行定位
            for i, cluster_ids in enumerate(clusters):
                ret, log = pose_from_cluster(
                    localizer, qname, qcam, cluster_ids, features, matches
                )
                # 记录内点最多的那个聚类结果
                if ret is not None and ret["num_inliers"] > best_inliers:
                    best_cluster = i
                    best_inliers = ret["num_inliers"]
                logs_clusters.append(log)
            
            # 如果找到了有效定位结果，保存位姿
            if best_cluster is not None:
                ret = logs_clusters[best_cluster]["PnP_ret"]
                cam_from_world[qname] = ret["cam_from_world"]
                
            logs["loc"][qname] = { ... } # 记录日志

        # 分支 B: 标准定位流程 (默认路径)
        else:
            # 直接使用所有检索到的候选图进行定位
            # pose_from_cluster 内部会做以下事情：
            # 1. 找到 Query 特征点对应的 Database 特征点
            # 2. 查表找到 Database 特征点对应的 3D 点 ID
            # 3. 建立 Query 2D 点 <-> 3D 点 的对应关系
            # 4. 运行 pycolmap.absolute_pose_estimation (PnP + RANSAC)
            ret, log = pose_from_cluster(
                localizer, qname, qcam, db_ids, features, matches
            )
            
            # 如果 PnP 求解成功，保存结果
            if ret is not None:
                cam_from_world[qname] = ret["cam_from_world"]
            else:
                # 兜底策略 (Fallback): 如果 PnP 失败 (通常是因为匹配太少)
                # 直接把检索到的第 1 张参考图 (Top-1) 的位姿，当作查询图的位姿。
                # 虽然不准，但总比没有好 (至少能拿个及格分)。
                closest = reference_sfm.images[db_ids[0]]
                cam_from_world[qname] = closest.cam_from_world()
            
            log["covisibility_clustering"] = covisibility_clustering
            logs["loc"][qname] = log

    logger.info(f"Localized {len(cam_from_world)} / {len(queries)} images.")
    
    # 将结果写入符合 Benchmark 格式的 txt 文件
    logger.info(f"Writing poses to {results}...")
    write_poses(cam_from_world, results, prepend_camera_name=prepend_camera_name)

    # 保存二进制日志
    logs_path = f"{results}_logs.pkl"
    with open(logs_path, "wb") as f:
        pickle.dump(logs, f)
    logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_sfm", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--ransac_thresh", type=float, default=12.0)
    parser.add_argument("--covisibility_clustering", action="store_true")
    parser.add_argument("--prepend_camera_name", action="store_true")
    args = parser.parse_args()
    main(**args.__dict__)
