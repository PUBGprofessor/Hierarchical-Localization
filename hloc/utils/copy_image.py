import os
import shutil
from pathlib import Path

def process_image_list(list_file_path, dataset_root, output_dir_name="result_images"):
    """
    Args:
        list_file_path: 包含图片对的txt文件路径
        dataset_root: 图片实际存储的根目录 (因为txt里通常是相对路径)
        output_dir_name: 结果保存的文件夹名称
    """
    
    # 1. 创建输出目录
    output_dir = Path(r"datasets\aachen_v1_1", output_dir_name)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        print(f"已创建输出目录: {output_dir.resolve()}")
    else:
        print(f"输出目录已存在: {output_dir.resolve()}")

    # 用于记录已经复制过的图片，防止重复操作（因为查询图每行都有）
    processed_files = set()

    # 2. 读取文件并处理
    with open(list_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"共找到 {len(lines)} 行数据，开始处理...")

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        # 分割每一行，获取 query路径 和 db路径
        try:
            # 假设是用空格分隔的
            query_rel_path, db_rel_path = line.split()
        except ValueError:
            print(f"[跳过] 格式错误的行: {line}")
            continue

        # 拼接完整的源文件路径
        src_query_path = Path(dataset_root) / query_rel_path
        src_db_path = Path(dataset_root) / db_rel_path

        # --- 处理 Query 图片 ---
        # 只复制一次 Query 图片
        if src_query_path not in processed_files:
            if src_query_path.exists():
                 # 添加前缀 (例如 query_1860.jpg)
                new_query_name = f"query_{src_query_path.name}"
                dest_query_path = output_dir / new_query_name
                shutil.copy2(src_query_path, dest_query_path)
                processed_files.add(src_query_path)
                print(f"[Query] 复制: {src_query_path.name}")
            else:
                print(f"[错误] Query 图片不存在: {src_query_path}")

        # --- 处理 DB 图片 ---
        if src_db_path.exists():
            # 获取原文件名 (例如 1860.jpg)
            original_db_name = src_db_path.name
            # db图片保持原名
            new_db_name = f"{original_db_name}"
            
            dest_db_path = output_dir / new_db_name
            shutil.copy2(src_db_path, dest_db_path)
            print(f"  -> [DB] 复制并重命名: {original_db_name} -> {new_db_name}")
        else:
            print(f"  -> [错误] DB 图片不存在: {src_db_path}")

    print("\n所有操作完成！")

# ================= 配置区域 =================
if __name__ == "__main__":
    # 1. 在这里填入你的 txt 文件路径
    # 假设你的数据保存在 pairs.txt 中
    MY_LIST_FILE = Path(r"outputs\aachen_v1.1\pairs-query-netvlad30_singleImage.txt") 

    # 2. 在这里填入图片所在的根目录路径
    # 如果 txt 里的路径是相对于当前文件夹的，就写 "."
    # 如果图片在 D:/datasets/Aachen/ 下，就写那个路径
    MY_DATASET_ROOT = r"datasets\aachen_v1_1\images_upright" 

    # 3. 想要保存的文件夹名字
    MY_OUTPUT_DIR = "visualize_result"
    # 展示当前路径
    print(f"当前工作目录: {Path.cwd()}")

    # 创建一个示例 txt 文件用于测试 (如果你已经有文件了，这一段可以删除)
    if not os.path.exists(MY_LIST_FILE):
        print("未找到列表文件，正在生成示例文件...")
        sample_data = """query/night/nexus5x/IMG_20161227_172439.jpg db/1860.jpg
query/night/nexus5x/IMG_20161227_172439.jpg db/1286.jpg"""
        with open(MY_LIST_FILE, "w") as f:
            f.write(sample_data)
        # 注意：因为只是示例，没有真实的 jpg 图片，运行会提示“图片不存在”

    # 运行函数
    process_image_list(MY_LIST_FILE, MY_DATASET_ROOT, MY_OUTPUT_DIR)