import os
import re
import numpy as np
from scipy.spatial.transform import Rotation as R

def parse_hloc_images_txt(txt_path):
    """解析 hloc/COLMAP 的 images.txt 文件 (World-to-Camera 格式)"""
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
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            name = parts[9]
            
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
                        pass
                continue
        i += 1
    return results

def compute_rotation_error(R_pred, R_gt):
    R_err = np.dot(R_pred.T, R_gt)
    tr = np.trace(R_err)
    val = (tr - 1.0) / 2.0
    val = np.clip(val, -1.0, 1.0)
    return np.degrees(np.arccos(val))

def compute_translation_direction_error(t_pred, t_gt):
    norm_pred = np.linalg.norm(t_pred)
    norm_gt = np.linalg.norm(t_gt)
    if norm_pred < 1e-8 or norm_gt < 1e-8:
        return 0.0
    t_pred_unit = t_pred / norm_pred
    t_gt_unit = t_gt / norm_gt
    dot = np.dot(t_pred_unit, t_gt_unit)
    dot = np.clip(dot, -1.0, 1.0)
    return np.degrees(np.arccos(dot))

def process_and_annotate(reloc3r_txt, hloc_txt, output_txt):
    print("Parsing hloc absolute poses...")
    hloc_poses = parse_hloc_images_txt(hloc_txt)
    if not hloc_poses:
        return
        
    print("Parsing Reloc3r relative poses...")
    reloc3r_pairs = parse_reloc3r_results(reloc3r_txt)
    if not reloc3r_pairs:
        print("Error: No valid pairs found.")
        return

    evaluated_data = []

    for img1, img2, T_pred in reloc3r_pairs:
        if img1 not in hloc_poses or img2 not in hloc_poses:
            continue
            
        R1, t1 = hloc_poses[img1]
        R2, t2 = hloc_poses[img2]
        
        # Ground Truth W2C 相对位姿
        R_gt_2to1 = np.dot(R1, R2.T)
        t_gt_2to1 = t1 - np.dot(R_gt_2to1, t2)
        
        R_pred = T_pred[0:3, 0:3]
        t_pred = T_pred[0:3, 3]
        
        rot_err = compute_rotation_error(R_pred, R_gt_2to1)
        trans_err = compute_translation_direction_error(t_pred, t_gt_2to1)
        
        evaluated_data.append({
            'img1': img1,
            'img2': img2,
            'matrix': T_pred,
            'rot_err': rot_err,
            'trans_err': trans_err,
            # 综合评分用于排序（以旋转误差为主，可根据需要调整）
            'score': rot_err + trans_err / 10.0 
        })

    if not evaluated_data:
        print("Error: No matching pairs evaluated.")
        return

    # 按综合评分或旋转误差排序
    evaluated_data.sort(key=lambda x: x['rot_err']) # 按旋转误差从小到大排序
    
    n_total = len(evaluated_data)
    smallest_20_set = {id(item) for item in evaluated_data[:20]}
    largest_20_set = {id(item) for item in evaluated_data[-20:]}

    print(f"Writing annotated results to {output_txt}...")
    with open(output_txt, 'w') as f:
        for item in evaluated_data:
            tag = ""
            if id(item) in smallest_20_set:
                tag = " [TOP_20_SMALLEST]"
            elif id(item) in largest_20_set:
                tag = " [TOP_20_LARGEST]"
                
            header = f"--- Pair: {item['img1']} to {item['img2']} | Rot_Err: {item['rot_err']:.4f}° | Trans_Err: {item['trans_err']:.4f}°{tag} ---"
            f.write(header + "\n")
            
            # 写入 4x4 矩阵
            for row in item['matrix']:
                f.write(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f} {row[3]:.6f}\n")
            f.write("\n")

    # 打印 Top 20 极值结果到控制台
    print("\n" + "="*80)
    print(" TOP 20 SMALLEST ROTATION ERROR PAIRS:")
    print(f"{'Index':<6} | {'Image 1':<16} | {'Image 2':<16} | {'Rot Err (°)':<12} | {'Trans Err (°)'}")
    print("-" * 80)
    for idx, item in enumerate(evaluated_data[:20]):
        print(f"{idx+1:<6} | {item['img1']:<16} | {item['img2']:<16} | {item['rot_err']:<12.4f} | {item['trans_err']:.4f}")

    print("\n" + "="*80)
    print(" TOP 20 LARGEST ROTATION ERROR PAIRS:")
    print(f"{'Index':<6} | {'Image 1':<16} | {'Image 2':<16} | {'Rot Err (°)':<12} | {'Trans Err (°)'}")
    print("-" * 80)
    for idx, item in enumerate(evaluated_data[-20:][::-1]): # 从大到小倒序打出
        print(f"{idx+1:<6} | {item['img1']:<16} | {item['img2']:<16} | {item['rot_err']:<12.4f} | {item['trans_err']:.4f}")
    print("="*80)
    print(f"Done! Annotated file saved as: {output_txt}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Annotate all_poses.txt with errors and highlight top 20 extreme pairs')
    parser.add_argument('--reloc3r_txt', type=str, default='all_poses.txt', help='Path to original Reloc3r output txt')
    parser.add_argument('--hloc_txt', type=str, default='/home/ljh/workspace/dataset/KingsCollege/seq1_train130/sparse/0/images.txt', help='Path to hloc/COLMAP images.txt')
    parser.add_argument('--output_txt', type=str, default='all_poses_with_errors.txt', help='Path to output annotated txt')
    args = parser.parse_args()
    
    process_and_annotate(args.reloc3r_txt, args.hloc_txt, args.output_txt)