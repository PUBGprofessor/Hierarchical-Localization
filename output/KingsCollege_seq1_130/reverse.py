import re
from pathlib import Path
import numpy as np

def parse_frame_num(name):
    """从文件名中提取帧编号的整数，例如 'frame00002.png' -> 2"""
    match = re.search(r'frame(\d+)\.png', name)
    return int(match.group(1)) if match else 0

def invert_transformation_matrix(T):
    """求 4x4 齐次变换矩阵的逆矩阵 (T_{A->B} -> T_{B->A})"""
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv

def sort_and_format_poses(input_txt, output_txt):
    input_path = Path(input_txt)
    output_path = Path(output_txt)
    
    if not input_path.exists():
        print(f"[Error] 输入文件不存在: {input_path}")
        return

    # 1. 生成所有期望的完整图片列表（130张，从 2 到 260，步长 2）
    all_frames = [f"frame{i:05d}.png" for i in range(2, 261, 2)]
    
    # 生成所有理论上的期望配对 (f1 < f2)
    expected_pairs = set()
    for i in range(len(all_frames)):
        for j in range(i + 1, len(all_frames)):
            expected_pairs.add((all_frames[i], all_frames[j]))
            
    # 2. 解析现有的结果文件并规范化矩阵
    found_poses = {} # 键: (nameA, name2), 值: 4x4 矩阵
    content = input_path.read_text(encoding='utf-8')
    blocks = content.strip().split('\n\n')
    
    for block in blocks:
        lines = [line.strip() for line in block.strip().split('\n') if line.strip()]
        if not lines:
            continue
        header_line = lines[0]
        
        match = re.search(r'--- Pair: (\S+) to (\S+) ---', header_line)
        if match:
            name1, name2 = match.group(1), match.group(2)
            
            # 读取 4x4 矩阵
            T = np.zeros((4, 4), dtype=np.float64)
            for r_idx, r_line in enumerate(lines[1:5]):
                T[r_idx] = list(map(float, r_line.split()))
            
            num1 = parse_frame_num(name1)
            num2 = parse_frame_num(name2)
            
            if num1 == num2:
                continue
            
            # 统一转换成：编号小的在前，大的在后 (nameA -> nameB)
            if num1 < num2:
                canonical_pair = (name1, name2)
                canonical_T = T
            else:
                canonical_pair = (name2, name1)
                canonical_T = invert_transformation_matrix(T) # 几何求逆！
                
            found_poses[canonical_pair] = canonical_T

    # 3. 统计缺失的图片对
    found_keys = set(found_poses.keys())
    missing_pairs = expected_pairs - found_keys
    sorted_missing = sorted(list(missing_pairs), key=lambda x: (parse_frame_num(x[0]), parse_frame_num(x[1])))
    
    print("=" * 50)
    print(f"📊 统计报告:")
    print(f"  - 理论总配对数 (Expected): {len(expected_pairs)}")
    print(f"  - 成功计算配对数 (Found):   {len(found_keys)}")
    print(f"  - 缺失配对数 (Missing):     {len(sorted_missing)}")
    print("=" * 50)
    
    if sorted_missing:
        print(f"\n⚠️ 提示: 有 {len(sorted_missing)} 个图片对因匹配点不足或 RANSAC 失败而缺失。")

    # 4. 按照帧序号顺序对成功获取的位姿进行全局排序
    sorted_found_keys = sorted(list(found_keys), key=lambda x: (parse_frame_num(x[0]), parse_frame_num(x[1])))
    
    # 5. 写入到排版后的新文件中
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for nameA, nameB in sorted_found_keys:
            f_out.write(f"--- Pair: {nameA} to {nameB} ---\n")
            T = found_poses[(nameA, nameB)]
            for row in T:
                f_out.write(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f} {row[3]:.6f}\n")
            f_out.write("\n")
            
    print(f"✨ 转换、求逆与重排后的文件已成功保存至:\n  📁 {output_path}")

if __name__ == "__main__":
    input_file = "./seq1_train130_pairwise_relative_poses.txt"
    output_file = "./seq1_train130_pairwise_relative_poses_sorted.txt"

    sort_and_format_poses(input_file, output_file)