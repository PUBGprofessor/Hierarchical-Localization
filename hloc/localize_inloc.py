"""InLoc 定位算法：用数据库图像的激光扫描深度建立 2D-3D 对应并估计查询位姿。

总体输入：
- InLoc 数据集根目录：pairs 引用的查询/数据库图像、数据库扫描的 XYZcut ``.mat``、
  ``database/alignments/<楼层>/transformations`` 中的扫描到全局坐标变换文本。
- 检索 pairs 文本、局部特征 HDF5 和图像匹配 HDF5。

总体输出：调用 ``main`` 时写出指定的位姿结果文本，以及在其路径后追加
``_logs.pkl`` 的定位诊断日志。算法不生成或修改数据集文件、检索 pairs、特征和匹配。
各函数的输入输出细节见对应中文 docstring。
"""

import argparse
import pickle
from pathlib import Path

import cv2
import h5py
import numpy as np
import pycolmap
import torch
from scipy.io import loadmat
from tqdm import tqdm

from . import logger
from .utils.io import write_poses
from .utils.parsers import names_to_pair, parse_retrieval


def interpolate_scan(scan, kp):
    """在数据库扫描的 XYZcut 深度图中插值关键点的三维坐标。

    参数：
    - ``scan``：从 InLoc 数据库图对应的 ``<图像相对路径>.mat`` 中读取的
      ``XYZcut`` 数组，形状通常为 ``(高, 宽, 3)``，每个像素存储扫描坐标系下的 XYZ。
    - ``kp``：关键点二维像素坐标数组，形状为 ``(N, 2)``。

    先做双线性插值；插值结果含 NaN 时使用最近邻结果补足。返回
    ``kp3d``（形状 ``(N, 3)`` 的三维坐标）和 ``valid``（形状 ``(N,)`` 的有效标记）。
    此函数不读写文件。
    """
    h, w, c = scan.shape
    kp = kp / np.array([[w - 1, h - 1]]) * 2 - 1
    assert np.all(kp > -1) and np.all(kp < 1)
    scan = torch.from_numpy(scan).permute(2, 0, 1)[None]
    kp = torch.from_numpy(kp)[None, None]
    grid_sample = torch.nn.functional.grid_sample

    # To maximize the number of points that have depth:
    # do bilinear interpolation first and then nearest for the remaining points
    interp_lin = grid_sample(scan, kp, align_corners=True, mode="bilinear")[0, :, 0]
    interp_nn = torch.nn.functional.grid_sample(
        scan, kp, align_corners=True, mode="nearest"
    )[0, :, 0]
    interp = torch.where(torch.isnan(interp_lin), interp_nn, interp_lin)
    valid = ~torch.any(torch.isnan(interp), 0)

    kp3d = interp.T.numpy()
    valid = valid.numpy()
    return kp3d, valid


def get_scan_pose(dataset_dir, rpath):
    """读取数据库扫描坐标系到 InLoc 全局坐标系的刚体变换。

    ``rpath`` 是检索文件中的数据库图相对路径，路径的倒数第三级和第二级
    分别作为楼层名和扫描 ID，图像名前三字符作为建筑名。函数据此读取：
    ``<dataset_dir>/database/alignments/<楼层名>/transformations/`` 下的
    ``<建筑名>_trans_<扫描ID>.txt``，并将第 8 至 11 行解析为 4x4 变换矩阵。

    返回矩阵 ``T``；调用方用 ``T[:3, :3]`` 和 ``T[:3, 3]`` 将扫描 XYZ 转到全局坐标。
    只读取文件，不产生文件。
    """
    split_image_rpath = rpath.split("/")
    floor_name = split_image_rpath[-3]
    scan_id = split_image_rpath[-2]
    image_name = split_image_rpath[-1]
    building_name = image_name[:3]

    # path = Path(
    #     dataset_dir,
    #     "database/alignments",
    #     floor_name,
    #     f"transformations/{building_name}_trans_{scan_id}.txt",
    # )
    path = Path(
            dataset_dir,
            f"{floor_name}_alignment",
            f"transformations/{building_name}_trans_{scan_id}.txt",
        )
    with open(path) as f:
        raw_lines = f.readlines()

    P_after_GICP = np.array(
        [
            np.fromstring(raw_lines[7], sep=" "),
            np.fromstring(raw_lines[8], sep=" "),
            np.fromstring(raw_lines[9], sep=" "),
            np.fromstring(raw_lines[10], sep=" "),
        ]
    )

    return P_after_GICP


def pose_from_cluster(dataset_dir, q, retrieved, feature_file, match_file, skip=None):
    """汇总单张查询图与其检索数据库图的 2D-3D 对应关系并估计相机位姿。

    参数及所需输入：
    - ``dataset_dir``：InLoc 数据集根目录。
    - ``q``、``retrieved``：查询图相对路径及其检索到的数据库图相对路径列表；
      它们来自检索 pairs 文件。查询图像文件用于读取宽高。
    - ``feature_file``：由 HLoc 特征提取产生的 HDF5 句柄，需含查询图和数据库图
      的 ``keypoints`` 数据集。
    - ``match_file``：由 HLoc 特征匹配产生的 HDF5 句柄，需含每对图像的
      ``matches0`` 数据集。
    - ``skip``：可选匹配数阈值；低于阈值的数据库图会被跳过。

    对每个保留的数据库图，函数读取 ``<dataset_dir>/<数据库图相对路径>.mat``
    中的 ``XYZcut``，将匹配到的数据库关键点插值为 3D 点，再读取相应变换文本
    将这些点转换到全局坐标。最后用查询图像尺寸、固定焦距和 COLMAP PnP/RANSAC
    估计 ``cam_from_world`` 位姿。

    返回位姿估计字典、筛选后的查询/数据库二维关键点、三维点、每个对应点所属的
    检索数据库图索引，以及参与统计的匹配数。函数只读输入文件，不写文件。
    """
    height, width = cv2.imread(str(dataset_dir / q)).shape[:2]
    cx = 0.5 * width
    cy = 0.5 * height
    focal_length = 4032.0 * 28.0 / 36.0

    all_mkpq = []
    all_mkpr = []
    all_mkp3d = []
    all_indices = []
    kpq = feature_file[q]["keypoints"].__array__()
    num_matches = 0

    for i, r in enumerate(retrieved):
        kpr = feature_file[r]["keypoints"].__array__()
        pair = names_to_pair(q, r)
        m = match_file[pair]["matches0"].__array__()
        v = m > -1

        if skip and (np.count_nonzero(v) < skip):
            continue

        mkpq, mkpr = kpq[v], kpr[m[v]]
        num_matches += len(mkpq)

        scan_r = loadmat(Path(dataset_dir, r + ".mat"))["XYZcut"]
        mkp3d, valid = interpolate_scan(scan_r, mkpr)
        Tr = get_scan_pose(dataset_dir, r)
        mkp3d = (Tr[:3, :3] @ mkp3d.T + Tr[:3, -1:]).T

        all_mkpq.append(mkpq[valid])
        all_mkpr.append(mkpr[valid])
        all_mkp3d.append(mkp3d[valid])
        all_indices.append(np.full(np.count_nonzero(valid), i))

    all_mkpq = np.concatenate(all_mkpq, 0)
    all_mkpr = np.concatenate(all_mkpr, 0)
    all_mkp3d = np.concatenate(all_mkp3d, 0)
    all_indices = np.concatenate(all_indices, 0)

    cfg = {
        "model": "SIMPLE_PINHOLE",
        "width": width,
        "height": height,
        "params": [focal_length, cx, cy],
    }
    estimation_options = pycolmap.AbsolutePoseEstimationOptions()
    estimation_options.ransac.max_error = 48
    ret = pycolmap.estimate_and_refine_absolute_pose(
        all_mkpq, all_mkp3d, cfg, estimation_options
    )
    ret["cfg"] = cfg
    return ret, all_mkpq, all_mkpr, all_mkp3d, all_indices, num_matches


def main(dataset_dir, retrieval, features, matches, results, skip_matches=None):
    """对检索文件中的所有查询执行 InLoc 2D-3D 定位并保存结果和日志。

    输入：
    - ``dataset_dir``：InLoc 数据集根目录，包含查询/数据库图、数据库扫描的
      ``.mat`` 文件以及 ``database/alignments`` 变换文本。
    - ``retrieval``：文本格式的 query-db 检索对文件，例如
      ``pairs/inloc/pairs-query-netvlad40.txt``。每个查询对应一组候选数据库图。
    - ``features``：局部特征 HDF5 文件，含 pairs 涉及图像的关键点。
    - ``matches``：图像对匹配 HDF5 文件，含匹配索引。
    - ``results``：位姿结果文本的目标路径。
    - ``skip_matches``：传递给 ``pose_from_cluster`` 的过滤阈值。

    产生：
    - ``results``：逐查询图写出的估计相机位姿文本（不包含相机名字段）。
    - ``<results>_logs.pkl``：pickle 诊断日志，记录输入文件路径、每个查询的
      检索数据库图、PnP 返回值、关键点、3D 点和匹配数量。
    """
    assert retrieval.exists(), retrieval
    assert features.exists(), features
    assert matches.exists(), matches

    retrieval_dict = parse_retrieval(retrieval)
    queries = list(retrieval_dict.keys())

    feature_file = h5py.File(features, "r", libver="latest")
    match_file = h5py.File(matches, "r", libver="latest")

    poses = {}
    logs = {
        "features": features,
        "matches": matches,
        "retrieval": retrieval,
        "loc": {},
    }
    logger.info("Starting localization...")
    for q in tqdm(queries):
        db = retrieval_dict[q]
        # 每张查询图独立完成 2D-3D 汇总和 PnP；保留几何中间量供日志可视化/排查。
        ret, mkpq, mkpr, mkp3d, indices, num_matches = pose_from_cluster(
            dataset_dir, q, db, feature_file, match_file, skip_matches
        )

        poses[q] = ret["cam_from_world"]
        logs["loc"][q] = {
            "db": db,
            "PnP_ret": ret,
            "keypoints_query": mkpq,
            "keypoints_db": mkpr,
            "3d_points": mkp3d,
            "indices_db": indices,
            "num_matches": num_matches,
        }

    # 位姿写为文本；日志另存为 pickle，不改变输入的检索、特征或匹配文件。
    logger.info(f"Writing poses to {results}...")
    write_poses(poses, results, prepend_camera_name=False)

    logs_path = f"{results}_logs.pkl"
    logger.info(f"Writing logs to {logs_path}...")
    with open(logs_path, "wb") as f:
        pickle.dump(logs, f)
    logger.info("Done!")


if __name__ == "__main__":
    # 支持脱离 pipeline 直接运行本模块；参数与 main() 一一对应。
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--skip_matches", type=int)
    args = parser.parse_args()
    main(**args.__dict__)
