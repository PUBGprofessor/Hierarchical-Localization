import argparse
from pathlib import Path
from . import logger

def main(output, image_dir, num_matched):
    """
    不再读取 COLMAP 模型，直接读取图片文件夹进行序列配对
    output: pairs.txt 输出路径
    image_dir: 图片文件夹路径 (datasets/seq1/images)
    num_matched: overlap 数量
    """
    logger.info(f"正在从目录 {image_dir} 读取图片列表...")
    
    # 1. 获取所有图片并严格排序
    # 只抓取常见的图片格式
    extensions = ['.png', '.jpg', '.jpeg', '.JPG', '.PNG']
    image_paths = [p for p in Path(image_dir).iterdir() if p.suffix in extensions]
    
    # 关键：按文件名排序，确保 frame0001 在 frame0002 前面
    image_names = sorted([p.name for p in image_paths])
    
    num_images = len(image_names)
    if num_images == 0:
        logger.error(f"❌ 错误：在目录 {image_dir} 中没有找到任何图片！")
        return

    logger.info(f"统计：共有 {num_images} 张图片，正在生成序列配对 (Overlap={num_matched})...")
    
    pairs = []
    # 2. 生成滑动窗口配对
    for i in range(num_images):
        for j in range(1, num_matched + 1):
            if i + j < num_images:
                pairs.append((image_names[i], image_names[i + j]))

    # 3. 写入文件
    with open(output, "w") as f:
        f.write("\n".join(" ".join([i, j]) for i, j in pairs))
    
    logger.info(f"✅ 成功！Found {len(pairs)} pairs. 结果已写入 {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image_dir", required=True, type=Path)
    parser.add_argument("--num_matched", required=True, type=int)
    args = parser.parse_args()
    main(**args.__dict__)