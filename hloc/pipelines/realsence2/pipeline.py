import argparse
from pathlib import Path
from pprint import pformat
import pycolmap

from ... import (
    extract_features,
    localize_sfm,
    logger,
    match_features,
    pairs_from_covisibility,
    pairs_from_retrieval,
    pairs_from_sequence,
    pairs_from_exhaustive,
    triangulation,
)


def run(args):
    # Setup the paths
    dataset = args.dataset  # 例如: /home/.../KingsCollege/seq1_raw80
    dataset_name = dataset.name  # 自动提取文件夹名，例如 "seq1_raw80"
    
    images = dataset / "images/"  # dataset/room_raw63/images/
    
    # 根据数据集名称自动拼接 3D 模型路径
    sift_sfm = dataset / f"3D-models/{dataset_name}"  # dataset/aachen_v1.1/3D-models/room_raw125，官方提供的 COLMAP SfM 模型

    outputs = args.outputs  # the path where everything will be saved, 这里是outputs/room_raw63_render62
    reference_sfm = outputs / "sfm_superpoint+lightglue"  # the SfM model we will rebuild, 我们们将基于 SuperPoint+SuperGlue 重建的 SfM 模型
    sfm_pairs = (
        outputs / f"pairs-db-covis{args.num_covis}.txt"
    )  # top-k most covisible in SIFT model
    loc_pairs = (
        outputs / f"pairs-query-netvlad{args.num_loc}.txt"
    )  # top-k retrieved by NetVLAD

    # 根据数据集名称自动拼接结果文件名
    results = (
        outputs / f"{dataset_name}_hloc_superpoint+superglue_netvlad{args.num_loc}.txt"
    )

    # list the standard configurations available
    # logger.info("Configs for feature extractors:\n%s", pformat(extract_features.confs))
    # logger.info("Configs for feature matchers:\n%s", pformat(match_features.confs))

    # pick one of the configurations for extraction and matching
    retrieval_conf = extract_features.confs["netvlad"]
    # "netvlad": {
    #     "output": "global-feats-netvlad",
    #     "model": {"name": "netvlad"},
    #     "preprocessing": {"resize_max": 1024},
    # },
    # feature_conf = extract_features.confs["superpoint_max"]
    feature_conf = extract_features.confs["superpoint_my"]
    #  "superpoint_my": {
    #     "output": "feats-superpoint-n2048-rmax1024",
    #     "model": {
    #         "name": "superpoint",
    #         "nms_radius": 3,
    #         "max_keypoints": 2048,
    #     },
    #     "preprocessing": {
    #         "grayscale": True,
    #         "resize_max": 1024,
    #         "resize_force": True,
    #     },
    # },

    # matcher_conf = match_features.confs["superglue"]
    matcher_conf = match_features.confs["superpoint+lightglue"]
    # "superpoint+lightglue": {
    #     "output": "matches-superpoint-lightglue",
    #     "model": {
    #         "name": "lightglue",
    #         "features": "superpoint",
    #     },
    # },

#     （一般使用superpoint）提取局部特征（SuperPoint）。
#     所有图片（包括数据库 Database 图片和查询 Query 图片）。
#     返回生成的.h5 文件的路径，存着几千张图的 Keypoints 和 Descriptors。
    features = extract_features.main(feature_conf, images, outputs)  # 即写入的feats-superpoint-n2048-rmax1024.h5
    # features = outputs / "feats-superpoint-n2048-rmax1024.h5"

#   它读取旧的 sift_sfm 模型（官方提供的 COLMAP 模型）。如果旧模型里两张图看到了足够多的同一个 3D 点（即由共视关系 Covisibility），我们就认为这两张图应该匹配。
#   两两匹配（$N^2$）太慢了，只匹配“邻居”，极大加速。
    # pairs_from_covisibility.main(sift_sfm, sfm_pairs, num_matched=args.num_covis) # 把图片两两共视关系写到 sfm_pairs 文件里

    # -------------------------------------------------------------------
    # 使用详尽匹配生成建图匹配对，替代旧模型共视关系
    # -------------------------------------------------------------------
    logger.info("Reading registered images from the reference SfM model...")
    # 直接读取原有 sift_sfm 模型，获取成功注册的图片列表
    reconstruction = pycolmap.Reconstruction(sift_sfm)
    mapping_image_list = [img.name for img in reconstruction.images.values()]
    
    # 2. 传入 output 和 image_list 进行穷举匹配调用
    pairs_from_exhaustive.main(
        output=sfm_pairs, 
        image_list=mapping_image_list
    )
    # -------------------------------------------------------------------
    # -------------------------------------------------------------------

    # 使用上一步生成的匹配对，进行特征匹配（SuperGlue）。
    sfm_matches = match_features.main(
        matcher_conf, sfm_pairs, feature_conf["output"], outputs
    ) # 返回的是写入的 matches-superglue 文件路径

#   固定相机的位姿不变，根据 SuperPoint 的匹配关系，计算出新的 3D 特征点云坐标，即reference_sfm。（带有特征描述符？）
    triangulation.main(
        reference_sfm, sift_sfm, images, sfm_pairs, features, sfm_matches
    )

#   提取全局特征（通常是 NetVLAD）
    global_descriptors = extract_features.main(retrieval_conf, images, outputs) # global-feats-netvlad.h5文件

#   生成定位用的“匹配对”: 对于每一张 Query 图，计算它和所有 Database 图的全局特征相似度，选出 Top-50（由 args.num_loc 控制）最像的图。
    pairs_from_retrieval.main(
        global_descriptors, # 查询图像的全局特征文件 (.h5)，如 NetVLAD 特征
        loc_pairs,  # 写入的配对文件路径：output/../pairs-query-netvlad50.txt
        args.num_loc, # 10
        query_prefix="query/",  # 疑问在这，是否筛选出了查询图？
        db_model=reference_sfm,
    )

    # 对 Query 图和它找到的 Top-K 个“嫌疑人”进行精细匹配
    loc_matches = match_features.main(
        matcher_conf, loc_pairs, feature_conf["output"], outputs
    ) # 写入的 matches-superglue 文件路径: output/../matches-superglue

    # 定位 (Localization)
    localize_sfm.main(
        reference_sfm,
        dataset / "query_list.txt",
        # dataset / "queries/*_time_queries_with_intrinsics.txt",
        loc_pairs,
        features,
        loc_matches,
        results, # 写入的定位结果文件路径: output/..Aachen-v1.1_hloc_superpoint+superglue_netvlad50.txt
        covisibility_clustering=False,
    )  # not required with SuperPoint+SuperGlue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        # default="datasets/aachen_v1.1",
        default="/home/ljh/workspace/dataset/mapping_fisheye",
        help="Path to the dataset, default: %(default)s",
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default="/home/ljh/workspace/dataset/mapping_fisheye",
        help="Path to the output directory, default: %(default)s",
    )
    parser.add_argument( # 建图阶段的“寻亲”数量
        "--num_covis",
        type=int,
        default=20,
        help="Number of image pairs for SfM, default: %(default)s",
    )
    parser.add_argument( # 用于定位的图像对数量
        "--num_loc",
        type=int,
        default=10,
        # default=50,
        help="Number of image pairs for loc, default: %(default)s",
    )
    args = parser.parse_args()
    
    # 动态将 outputs 默认指向 dataset 目录，除非显式传入了 outputs
    if args.outputs == Path("/home/ljh/workspace/dataset/mapping_fisheye") and args.dataset != Path("/home/ljh/workspace/dataset/mapping_fisheye"):
         args.outputs = args.dataset
         
    run(args)