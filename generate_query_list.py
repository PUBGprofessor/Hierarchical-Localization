import os
from pathlib import Path

def generate_query_list_with_intrinsics():
    # ================= 配置区域 =================
    # 你的查询图片所在的具体文件夹
    query_images_dir = Path("datasets/room_raw63/images/query")
    
    # 最终输出的 query_list.txt 路径 (放在数据集根目录)
    output_txt_path = Path("datasets/room_raw63/query_list.txt")
    
    # 你从 COLMAP 复制出来的精确相机参数 (去掉前面的 ID "1"，只保留模型和数字)
    # 格式: MODEL WIDTH HEIGHT PARAMS[]
    camera_params = "PINHOLE 3114 2075 3172.5294374200544 3173.9530257001675 1557 1037.5"
    
    # 在 HLoc 中，图片路径需要相对于 images/ 文件夹
    # 因为你的图放在 images/query/ 下，所以前缀是 "query/"
    prefix = "query/"
    # ============================================

    if not query_images_dir.exists():
        print(f"❌ 错误: 找不到查询图片文件夹 '{query_images_dir}'")
        return

    # 支持常见的图片格式后缀
    valid_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    
    print(f"📂 正在扫描文件夹: {query_images_dir} ...")
    
    # 获取所有图片并按字母顺序排序
    image_files = [f for f in query_images_dir.iterdir() if f.is_file() and f.suffix in valid_extensions]
    image_files.sort(key=lambda x: x.name)
    
    if len(image_files) == 0:
        print("⚠️ 警告: 文件夹中没有找到图片文件。")
        return

    # 写入文件
    with open(output_txt_path, "w", encoding="utf-8") as f:
        for img in image_files:
            # 拼装完美的一行: "query/图片名.JPG PINHOLE 宽 高 参数..."
            line = f"{prefix}{img.name} {camera_params}\n"
            f.write(line)

    print(f"🎉 成功生成高级查询列表！共写入 {len(image_files)} 条记录。")
    print(f"📄 文件已保存至: {output_txt_path.absolute()}")
    
    # 打印前两行作为预览检查
    print("\n👀 文件内容预览 (前两行):")
    with open(output_txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i in range(min(2, len(lines))):
            print(lines[i].strip())

if __name__ == "__main__":
    generate_query_list_with_intrinsics()