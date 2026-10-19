import argparse
from pathlib import Path
from pprint import pformat
import h5py
import numpy as np
import cv2
from tqdm import tqdm

from ... import (
    extract_features,
    localize_sfm,
    logger,
    match_features,
    pairs_from_covisibility,
    pairs_from_retrieval,
    pairs_from_exhaustive,
    triangulation,
)


def compute_pairwise_relative_poses(features_path, matches_path, pairs_path, output_txt, K):
    """
    计算两两图像之间的相对位姿 (R, t) 并保存到 txt（带 tqdm 进度条）。
    """
    results = []
    
    with h5py.File(features_path, 'r') as f_feat, h5py.File(matches_path, 'r') as f_match:
        with open(pairs_path, 'r') as f:
            pairs = [line.strip().split() for line in f if line.strip()]
        
        success_count = 0
        fail_key = 0
        fail_matches = 0
        fail_E = 0
        
        # 2. 使用 tqdm 包装 pairs 列表
        for idx, pair in enumerate(tqdm(pairs, desc="Computing RPE", ncols=100)):
            if len(pair) < 2:
                continue
            img1, img2 = pair[0], pair[1]
            
            matches = None
            reversed_key = False
            
            try:
                if img1 in f_match and img2 in f_match[img1]:
                    item = f_match[img1][img2]
                    matches = item[()] if isinstance(item, h5py.Dataset) else (item['matches0'][()] if 'matches0' in item else item['matches'][()])
                    reversed_key = False
                elif img2 in f_match and img1 in f_match[img2]:
                    item = f_match[img2][img1]
                    matches = item[()] if isinstance(item, h5py.Dataset) else (item['matches0'][()] if 'matches0' in item else item['matches'][()])
                    reversed_key = True
                else:
                    fail_key += 1
                    continue
            except Exception:
                fail_key += 1
                continue
            
            if matches is None:
                fail_matches += 1
                continue
            
            valid = matches > -1
            if np.sum(valid) < 8:  # 匹配点少于 8 对
                fail_matches += 1
                continue
            
            try:
                kp1 = f_feat[img1]['keypoints'][()]
                kp2 = f_feat[img2]['keypoints'][()]
            except KeyError:
                continue
            
            if not reversed_key:
                if len(matches) != len(kp1):
                    fail_matches += 1
                    continue
                pts1 = kp1[valid]
                pts2 = kp2[matches[valid]]
            else:
                if len(matches) != len(kp2):
                    fail_matches += 1
                    continue
                pts1 = kp1[matches[valid]]
                pts2 = kp2[valid]
            
            # 1. 计算本质矩阵 (Essential Matrix)
            E, mask = cv2.findEssentialMat(
                pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
            )
            if E is None:
                fail_E += 1
                continue
                
            # 2. 从本质矩阵恢复相对位姿 (R, t)
            _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
            
            results.append((img1, img2, R, t))
            success_count += 1
            
        print(f"\n[DEBUG] Final Stats -> Success: {success_count}, Fail Key: {fail_key}, Fail Matches: {fail_matches}, Fail E: {fail_E}")
            
    # 按指定格式写入结果到 txt 文件
    with open(output_txt, 'w') as f_out:
        for img1, img2, R, t in results:
            name1 = img1.split('/')[-1]
            name2 = img2.split('/')[-1]
            
            f_out.write(f"--- Pair: {name1} to {name2} ---\n")
            
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = t.flatten()
            
            for row in T:
                f_out.write(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f} {row[3]:.6f}\n")
            f_out.write("\n")
            
    print(f"Successfully computed and saved {len(results)} relative poses to: {output_txt}")

def run(args):
    # Setup the paths
    dataset = args.dataset 
    dataset_name = dataset.name 
    
    images = dataset / "images/" 
    outputs = args.outputs 
    
    sfm_pairs = outputs / f"pairs-db-covis{args.num_covis}.txt"
    relative_pose_output = outputs / f"{dataset_name}_pairwise_relative_poses.txt"

    # 1. 提取局部特征 (SuperPoint)
    feature_conf = extract_features.confs["superpoint_my"]
    features = extract_features.main(feature_conf, images, outputs)  

    # -------------------------------------------------------------------
    # 2. 修改：使用详尽匹配 (Exhaustive Matching) 生成两两匹配对
    # -------------------------------------------------------------------
    sfm_pairs = outputs / f"pairs-exhaustive.txt"
    
    # 核心修改：用 features=features 代替 image_list=mapping_dir
    pairs_from_exhaustive.main(  
        output=sfm_pairs, 
        features=features, # 从 .h5 特征文件中自动获取所有图片名进行两两全排列组合
    )
    # -------------------------------------------------------------------

    # 3. 特征匹配 (LightGlue / SuperGlue)
    matcher_conf = match_features.confs["superglue"]
    sfm_matches = match_features.main(
        matcher_conf, sfm_pairs, feature_conf["output"], outputs
    ) # sfm_matches 直接返回生成的 matches .h5 文件路径！ 

    # -------------------------------------------------------------------
    # 4. 新增：基于已知内参和特征匹配，计算两两相对位姿并保存
    # -------------------------------------------------------------------
    # 根据你提供的 cameras.txt 中的内参构造内参矩阵 K
    fx = 1598.8286579302537
    fy = 1590.7820490996983
    cx = 960.0
    cy = 540.0
    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1]
    ], dtype=np.float64)

    # matches 的实际输出路径（HLoc 的 match_features 会在 outputs 下生成对应的 .h5 文件）
    # 自动定位生成的 matches h5 文件
    relative_pose_output = outputs / f"{dataset_name}_pairwise_relative_poses.txt"

    compute_pairwise_relative_poses(
        features_path=features,
        matches_path=sfm_matches, # 直接传入上面 match_features 返回的路径
        pairs_path=sfm_pairs,
        output_txt=relative_pose_output,
        K=K
    )
    # -------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default="/home/ljh/workspace/dataset/KingsCollege/seq1_train130",
        help="Path to the dataset, default: %(default)s",
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default="/home/ljh/workspace/dataset/KingsCollege/seq1_train130",
        help="Path to the output directory, default: %(default)s",
    )
    parser.add_argument(
        "--num_covis",
        type=int,
        default=20,
        help="Number of image pairs, default: %(default)s",
    )
    args = parser.parse_args()
    
    if args.outputs == Path("/home/ljh/workspace/dataset/KingsCollege/seq1_train130") and args.dataset != Path("/home/ljh/workspace/dataset/KingsCollege/seq1_train130"):
         args.outputs = args.dataset
         
    run(args)