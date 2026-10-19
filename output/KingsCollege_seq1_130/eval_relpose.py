import os
import re
import numpy as np
from scipy.spatial.transform import Rotation as R

def parse_hloc_images_txt(txt_path):
    """
    解析 hloc/COLMAP 的 images.txt 文件 (World-to-Camera 格式)
    返回字典：{image_name: (R_matrix, t_vector)}
    """
    poses = {}
    if not os.path.exists(txt_path):
        print(f"Error: hloc images file not found at {txt_path}")
        return poses
        
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) >= 10:
            # COLMAP format: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            name = parts[9]
            
            # 转为旋转矩阵 (scipy 的 from_quat 接收 [x, y, z, w])
            rot = R.from_quat([qx, qy, qz, qw])
            R_mat = rot.as_matrix()
            t_vec = np.array([tx, ty, tz], dtype=float)
            
            poses[name] = (R_mat, t_vec)
    return poses

def parse_reloc3r_results(txt_path):
    """逐行扫描解析 Reloc3r 输出的 all_poses.txt 文件"""
    results = []
    if not os.path.exists(txt_path):
        print(f"Error: Reloc3r result file not found at {txt_path}")
        return results
        
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        if line.startswith("--- Pair:"):
            parts = line.split()
            if len(parts) >= 5:
                img1 = parts[2]
                img2 = parts[4]
                
                mat_lines = []
                i += 1
                while i < n and len(mat_lines) < 4:
                    m_line = lines[i].strip()
                    if m_line:
                        mat_lines.append(m_line)
                    i += 1
                    
                if len(mat_lines) == 4:
                    try:
                        mat = np.array([[float(x) for x in l.split()] for l in mat_lines])
                        results.append((img1, img2, mat))
                    except ValueError as e:
                        print(f"Warning: Failed to parse matrix for {img1} to {img2}: {e}")
                continue
        i += 1
        
    return results

def compute_rotation_error(R_pred, R_gt):
    """计算旋转角度误差（度）"""
    R_err = np.dot(R_pred.T, R_gt)
    tr = np.trace(R_err)
    val = (tr - 1.0) / 2.0
    val = np.clip(val, -1.0, 1.0)
    angle_rad = np.arccos(val)
    return np.degrees(angle_rad)

def compute_translation_direction_error(t_pred, t_gt):
    """计算平移方向误差（度）"""
    norm_pred = np.linalg.norm(t_pred)
    norm_gt = np.linalg.norm(t_gt)
    if norm_pred < 1e-8 or norm_gt < 1e-8:
        return 0.0
    t_pred_unit = t_pred / norm_pred
    t_gt_unit = t_gt / norm_gt
    dot = np.dot(t_pred_unit, t_gt_unit)
    dot = np.clip(dot, -1.0, 1.0)
    return np.degrees(np.arccos(dot))

def evaluate_relative_poses(reloc3r_txt, hloc_txt):
    print("Parsing hloc absolute poses...")
    hloc_poses = parse_hloc_images_txt(hloc_txt)
    if not hloc_poses:
        return
        
    print("Parsing Reloc3r relative poses...")
    reloc3r_pairs = parse_reloc3r_results(reloc3r_txt)
    if not reloc3r_pairs:
        print("Error: No valid relative pose pairs parsed from Reloc3r output.")
        return

    errors_by_step = {}
    matched_count = 0
    
    for img1, img2, T_pred in reloc3r_pairs:
        if img1 not in hloc_poses or img2 not in hloc_poses:
            continue
            
        matched_count += 1
        # 计算帧号步长
        frame_num1 = int(re.search(r'\d+', img1).group())
        frame_num2 = int(re.search(r'\d+', img2).group())
        step = abs(frame_num2 - frame_num1) // 2
        
        # 获取 hloc 绝对位姿 (World-to-Camera: R1, t1 和 R2, t2)
        R1, t1 = hloc_poses[img1]
        R2, t2 = hloc_poses[img2]
        
        # 【核心修正】根据 W2C 计算真实的 2to1 相对位姿：
        # R_gt_2to1 = R1 * R2^T
        # t_gt_2to1 = t1 - R_gt_2to1 * t2
        R_gt_2to1 = np.dot(R1, R2.T)
        t_gt_2to1 = t1 - np.dot(R_gt_2to1, t2)
        
        # 提取 Reloc3r 预测值 (pose2to1)
        R_pred = T_pred[0:3, 0:3]
        t_pred = T_pred[0:3, 3]
        
        # 计算误差
        rot_err = compute_rotation_error(R_pred, R_gt_2to1)
        trans_err = compute_translation_direction_error(t_pred, t_gt_2to1)
        
        if step not in errors_by_step:
            errors_by_step[step] = {'rot': [], 'trans': []}
        errors_by_step[step]['rot'].append(rot_err)
        errors_by_step[step]['trans'].append(trans_err)

    print(f"Successfully matched and evaluated {matched_count} pairs.")

    if matched_count == 0:
        print("Warning: Matched count is 0. Please check if image names match.")
        return

    # 打印格式化评估报告
    print("\n" + "="*65)
    print(f"{'Step (Gap)':<12} | {'Pairs':<8} | {'Mean Rot Err (°)':<18} | {'Mean Trans Err (°)':<20}")
    print("-" * 65)
    
    for step in sorted(errors_by_step.keys()):
        rots = errors_by_step[step]['rot']
        trans = errors_by_step[step]['trans']
        mean_rot = np.mean(rots)
        mean_trans = np.mean(trans)
        print(f"Step {step:<7} | {len(rots):<8} | {mean_rot:<18.4f} | {mean_trans:<20.4f}")
    print("="*65)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate Reloc3r relative poses against hloc/COLMAP poses')
    parser.add_argument('--reloc3r_txt', type=str, default='seq1_train130_pairwise_relative_poses_sorted.txt', help='Path to Reloc3r output txt')
    parser.add_argument('--hloc_txt', type=str, default='/home/ljh/workspace/dataset/KingsCollege/seq1_train130/sparse/0/images.txt', help='Path to hloc/COLMAP images.txt')
    args = parser.parse_args()
    
    evaluate_relative_poses(args.reloc3r_txt, args.hloc_txt)