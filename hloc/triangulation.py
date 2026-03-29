import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pycolmap
from tqdm import tqdm

from . import logger
from .utils.geometry import compute_epipolar_errors
from .utils.io import get_keypoints, get_matches
from .utils.parsers import parse_retrieval


class OutputCapture:
    def __init__(self, verbose: bool):
        self.verbose = verbose

    def __enter__(self):
        if not self.verbose:
            pycolmap.logging.alsologtostderr = False

    def __exit__(self, exc_type, *args):
        if not self.verbose:
            pycolmap.logging.alsologtostderr = True


def create_db_from_model(
    reconstruction: pycolmap.Reconstruction, database_path: Path
) -> Dict[str, int]:
    if database_path.exists():
        logger.warning("The database already exists, deleting it.")
        database_path.unlink()

    with pycolmap.Database.open(database_path) as db:
        logger.info("Creating database from the reference model...")
        for camera_id, camera in reconstruction.cameras.items():
            db.write_camera(camera, use_camera_id=True)
        for rig_id, rig in reconstruction.rigs.items():
            db.write_rig(rig, use_rig_id=True)
        for frame_id, frame in reconstruction.frames.items():
            db.write_frame(frame, use_frame_id=True)
        for image_id, image in reconstruction.images.items():
            db.write_image(image, use_image_id=True)
    logger.info("Finished creating the database.")
    return {image.name: image_id for image_id, image in reconstruction.images.items()}


def import_features(
    image_ids: Dict[str, int], db: pycolmap.Database, features_path: Path
):
    logger.info("Importing features into the database...")
    for image_name, image_id in tqdm(image_ids.items()):
        keypoints = get_keypoints(features_path, image_name)
        keypoints += 0.5  # COLMAP origin
        db.write_keypoints(image_id, keypoints)


def import_matches(
    image_ids: Dict[str, int],
    db: pycolmap.Database,
    pairs_path: Path,
    matches_path: Path,
    min_match_score: Optional[float] = None,
    skip_geometric_verification: bool = False,
):
    logger.info("Importing matches into the database...")

    with open(str(pairs_path), "r") as f:
        pairs = [p.split() for p in f.readlines()]

    matched = set()
    for name0, name1 in tqdm(pairs):
        id0, id1 = image_ids[name0], image_ids[name1]
        if len({(id0, id1), (id1, id0)} & matched) > 0:
            continue
        matches, scores = get_matches(matches_path, name0, name1)
        if min_match_score:
            matches = matches[scores > min_match_score]
        db.write_matches(id0, id1, matches)
        matched |= {(id0, id1), (id1, id0)}

        if skip_geometric_verification:
            db.write_two_view_geometry(
                id0, id1, pycolmap.TwoViewGeometry(inlier_matches=matches)
            )


def estimation_and_geometric_verification(
    database_path: Path, pairs_path: Path, verbose: bool = False
):
    logger.info("Performing geometric verification of the matches...")
    with OutputCapture(verbose):
        pycolmap.verify_matches(
            database_path,
            pairs_path,
            options=dict(ransac=dict(max_num_trials=20000, min_inlier_ratio=0.1)),
        )


def geometric_verification(
    image_ids: Dict[str, int],
    reference: pycolmap.Reconstruction,
    db: pycolmap.Database,
    features_path: Path,
    pairs_path: Path,
    matches_path: Path,
    max_error: float = 4.0,
):
    logger.info("Performing geometric verification of the matches...")

    pairs = parse_retrieval(pairs_path)

    inlier_ratios = []
    matched = set()
    for name0 in tqdm(pairs):
        id0 = image_ids[name0]
        image0 = reference.images[id0]
        cam0 = reference.cameras[image0.camera_id]
        kps0, noise0 = get_keypoints(features_path, name0, return_uncertainty=True)
        noise0 = 1.0 if noise0 is None else noise0
        if len(kps0) > 0:
            kps0 = np.stack(cam0.cam_from_img(kps0))
        else:
            kps0 = np.zeros((0, 2))

        for name1 in pairs[name0]:
            id1 = image_ids[name1]
            image1 = reference.images[id1]
            cam1 = reference.cameras[image1.camera_id]
            kps1, noise1 = get_keypoints(features_path, name1, return_uncertainty=True)
            noise1 = 1.0 if noise1 is None else noise1
            if len(kps1) > 0:
                kps1 = np.stack(cam1.cam_from_img(kps1))
            else:
                kps1 = np.zeros((0, 2))

            matches = get_matches(matches_path, name0, name1)[0]

            if len({(id0, id1), (id1, id0)} & matched) > 0:
                continue
            matched |= {(id0, id1), (id1, id0)}

            if matches.shape[0] == 0:
                db.write_two_view_geometry(id0, id1, pycolmap.TwoViewGeometry())
                continue

            cam1_from_cam0 = image1.cam_from_world() * image0.cam_from_world().inverse()
            errors0, errors1 = compute_epipolar_errors(
                cam1_from_cam0, kps0[matches[:, 0]], kps1[matches[:, 1]]
            )
            valid_matches = np.logical_and(
                errors0 <= cam0.cam_from_img_threshold(noise0 * max_error),
                errors1 <= cam1.cam_from_img_threshold(noise1 * max_error),
            )
            # TODO: We could also add E to the database, but we need
            # to reverse the transformations if id0 > id1 in utils/database.py.
            db.write_two_view_geometry(
                id0,
                id1,
                pycolmap.TwoViewGeometry(inlier_matches=matches[valid_matches, :]),
            )
            inlier_ratios.append(np.mean(valid_matches))
    logger.info(
        "mean/med/min/max valid matches %.2f/%.2f/%.2f/%.2f%%.",
        np.mean(inlier_ratios) * 100,
        np.median(inlier_ratios) * 100,
        np.min(inlier_ratios) * 100,
        np.max(inlier_ratios) * 100,
    )


def run_triangulation(
    model_path: Path,
    database_path: Path,
    image_dir: Path,
    reference_model: pycolmap.Reconstruction,
    verbose: bool = False,
    options: Optional[Dict[str, Any]] = None,
) -> pycolmap.Reconstruction:
    model_path.mkdir(parents=True, exist_ok=True)
    logger.info("Running 3D triangulation...")
    if options is None:
        options = {}
    with OutputCapture(verbose):
        reconstruction = pycolmap.triangulate_points(
            reference_model, database_path, image_dir, model_path, options=options
        )
    return reconstruction


def main(
    sfm_dir: Path,                    # 输出目录：新的 3D 模型和数据库将存在这里
    reference_model: Path,            # 输入参考模型：提供旧的相机位姿 (COLMAP 格式)
    image_dir: Path,                  # 图片根目录
    pairs: Path,                      # 匹配对列表 (pairs-sfm.txt)
    features: Path,                   # 特征文件 (features.h5)
    matches: Path,                    # 匹配文件 (matches.h5)
    skip_geometric_verification: bool = False, # 是否跳过几何校验 (默认 False，即要进行校验)
    estimate_two_view_geometries: bool = False, # 是否重新估计双视图几何 (通常为 False，直接使用 matches 的结果)
    min_match_score: Optional[float] = None,   # 过滤匹配的最小分数阈值
    verbose: bool = False,            # 是否打印详细日志
    mapper_options: Optional[Dict[str, Any]] = None, # COLMAP 建图器的详细参数
) -> pycolmap.Reconstruction:
    
    # 0. 基础检查：确保输入文件都存在
    assert reference_model.exists(), reference_model
    assert features.exists(), features
    assert pairs.exists(), pairs
    assert matches.exists(), matches

    # 创建输出目录
    sfm_dir.mkdir(parents=True, exist_ok=True)
    # 创建一个空的 SQLite 数据库文件 (COLMAP 的标准数据格式)
    database = sfm_dir / "database.db"
    
    # 读取参考模型 (这一步加载了相机的内参和外参)
    reference = pycolmap.Reconstruction(reference_model)

    # 1. 初始化数据库
    # 将参考模型中的相机参数和图像信息写入新的 database.db
    # 返回一个字典 image_ids，映射 "图片名 -> 数据库中的 Image ID"
    image_ids = create_db_from_model(reference, database)
    logger.info("Created reference_sfm database with %d images.", len(image_ids))
    
    # 2. 导入特征和匹配
    with pycolmap.Database.open(database) as db:
        # 将 .h5 中的特征点 (keypoints) 导入数据库
        import_features(image_ids, db, features)
        # 将 .h5 中的匹配关系 (matches) 导入数据库
        # 这一步会建立 TwoViewGeometry 表，记录哪两张图之间有哪些匹配点
        import_matches(
            image_ids,
            db,
            pairs,
            matches,
            min_match_score,
            skip_geometric_verification,
        )
    
    # 3. 几何校验 (Geometric Verification)
    # 这是一个关键步骤：虽然神经网络 (SuperGlue) 觉得两个点匹配，但它们在几何上合理吗？
    # 必须通过极线约束 (Epipolar Geometry) 剔除错误的外点 (Outliers)。
    if not skip_geometric_verification:
        if estimate_two_view_geometries:
            # 选项 A：如果不信任深度学习的匹配，完全从头计算基础矩阵 F 或单应矩阵 H (较慢)
            estimation_and_geometric_verification(database, pairs, verbose)
        else:
            # 选项 B (默认)：利用深度学习的匹配结果，快速进行 RANSAC 校验
            # 这里的 geometric_verification 是 HLoc 定制的快速校验函数
            with pycolmap.Database.open(database) as db:
                geometric_verification(
                    image_ids, reference, db, features, pairs, matches
                )
    
    # 4. 执行三角化 (Triangulation)
    # 这是最后一步大招：调用 COLMAP 的点云三角化器 (PointTriangulator)
    # 输入：固定的相机位姿 (reference) + 经过校验的 2D 匹配关系 (database)
    # 输出：计算出的 3D 点云坐标
    reconstruction = run_triangulation(
        sfm_dir, database, image_dir, reference, verbose, mapper_options
    )
    
    logger.info(
        "Finished the triangulation with statistics:\n%s", reconstruction.summary()
    )
    return reconstruction

def parse_option_args(args: List[str], default_options) -> Dict[str, Any]:
    options = {}
    for arg in args:
        idx = arg.find("=")
        if idx == -1:
            raise ValueError("Options format: key1=value1 key2=value2 etc.")
        key, value = arg[:idx], arg[idx + 1 :]
        if not hasattr(default_options, key):
            raise ValueError(
                f'Unknown option "{key}", allowed options and default values'
                f" for {default_options.summary()}"
            )
        value = eval(value)
        target_type = type(getattr(default_options, key))
        if not isinstance(value, target_type):
            raise ValueError(
                f'Incorrect type for option "{key}":' f" {type(value)} vs {target_type}"
            )
        options[key] = value
    return options


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sfm_dir", type=Path, required=True)
    parser.add_argument("--reference_sfm_model", type=Path, required=True)
    parser.add_argument("--image_dir", type=Path, required=True)

    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)

    parser.add_argument("--skip_geometric_verification", action="store_true")
    parser.add_argument("--min_match_score", type=float)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args().__dict__

    mapper_options = parse_option_args(
        args.pop("mapper_options"), pycolmap.IncrementalMapperOptions()
    )

    main(**args, mapper_options=mapper_options)
