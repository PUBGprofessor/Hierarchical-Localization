import argparse
from pathlib import Path
from ... import (
    extract_features,
    match_features,
    pairs_from_sequence, # 关键：改用序列配对
    triangulation,
    logger
)

def run(args):
    # 1. 路径设置
    dataset = args.dataset  # 你的数据集根目录
    images = dataset / "images"  # 图片文件夹
    
    # 重点：这是你存放 images.txt, cameras.txt 的文件夹（即你从 Cambridge 转换出来的 W2C 位姿）
    # 即使点云是错的，hloc 也会读取这里的位姿作为“已知位姿”
    input_sfm = dataset / "sparse/0" 
    
    outputs = args.outputs # 输出目录
    outputs.mkdir(parents=True, exist_ok=True)
    
    # 最终输出的 COLMAP 格式模型路径（用于 3DGS）
    reconstructed_sfm = outputs / "sfm_superpoint+lightglue"
    
    # 匹配对文件
    sfm_pairs = outputs / f"pairs-sequence-overlap{args.overlap}.txt"
    
    # 2. 配置项
    feature_conf = extract_features.confs["superpoint_my"]
    matcher_conf = match_features.confs["superpoint+lightglue"]

    # 3. 提取特征 (SuperPoint)
    # 这会生成 feats-superpoint-n2048-rmax1024.h5
    feature_path = extract_features.main(feature_conf, images, outputs)

    # 4. 生成序列匹配对 (降低复杂度的核心)
    # args.overlap=5 意味着每一帧会和它前后的 5 帧图像进行匹配
    logger.info(f"Generating pairs from sequence (overlap={args.overlap})...")
    pairs_from_sequence.main(sfm_pairs, images, num_matched=args.overlap)

    # 5. 特征匹配 (LightGlue)
    # 这会比 SuperGlue 更快、更稳
    match_path = match_features.main(
        matcher_conf, sfm_pairs, feature_conf["output"], outputs
    )

    # 6. 进行三角化 (Re-triangulation)
    # 核心逻辑：它会读取 input_sfm 里的位姿，保持位姿不动，利用新的匹配点计算 3D 坐标
    logger.info("Starting triangulation...")
    triangulation.main(
        reconstructed_sfm, # 输出目录
        input_sfm,         # 输入目录（包含你的 images.txt, cameras.txt）
        images,            # 图片目录
        sfm_pairs,         # 匹配对文件
        feature_path,      # .h5 特征
        match_path         # .h5 匹配结果
    )
    
    logger.info(f"重建完成！COLMAP 模型已保存至: {reconstructed_sfm}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default="F:\DATASETS\KingsCollege\seq1_train40")
    parser.add_argument("--outputs", type=Path, default="F:\DATASETS\KingsCollege\seq1_train40")
    parser.add_argument(
        "--overlap", 
        type=int, 
        default=5, 
        help="每一帧图片向后匹配的邻居数量（建议 5-15）"
    )
    args = parser.parse_args()
    run(args)