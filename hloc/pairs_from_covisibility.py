import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pycolmap
from tqdm import tqdm

from . import logger
from .utils.read_write_model import read_model


def _main_pycolmap(model, output, num_matched):
    """使用 pycolmap 轨迹计算共视 pairs，避免 legacy Python 模型副本。"""
    logger.info("Reading the COLMAP model with pycolmap...")
    reconstruction = pycolmap.Reconstruction(model)
    images = reconstruction.images
    image_names = {image_id: image.name for image_id, image in images.items()}
    covis_by_image = {image_id: defaultdict(int) for image_id in images}
    logger.info("Extracting image pairs from covisibility info...")
    for point in tqdm(reconstruction.points3D.values()):
        track_ids = [element.image_id for element in point.track.elements]
        for image_id in track_ids:
            covis = covis_by_image[image_id]
            for other_id in track_ids:
                if other_id != image_id:
                    covis[other_id] += 1

    pairs = []
    for image_id, covis in tqdm(covis_by_image.items()):
        if not covis:
            continue
        covis_ids = np.fromiter(covis.keys(), dtype=np.int64)
        covis_num = np.fromiter(covis.values(), dtype=np.int64)
        if len(covis_ids) <= num_matched:
            order = np.argsort(-covis_num)
        else:
            order = np.argpartition(covis_num, -num_matched)[-num_matched:]
            order = order[np.argsort(-covis_num[order])]
        pairs.extend(
            (image_names[image_id], image_names[int(covis_ids[i])]) for i in order
        )

    logger.info("Found %d pairs.", len(pairs))
    with open(output, "w") as f:
        f.write("\n".join(" ".join(pair) for pair in pairs))
    # Drop the C++ reconstruction before any later pycolmap model is loaded.
    del covis_by_image, reconstruction
    return


def main(model, output, num_matched):
    # 大型 map 优先使用 pycolmap 的 C++ 重建对象，避免 legacy reader
    # 为每个 3D 点创建大量 Python 对象。失败时保留下方旧实现兼容性。
    try:
        return _main_pycolmap(model, output, num_matched)
    except Exception as exc:
        logger.warning("Memory-efficient covisibility pass failed (%s); using legacy reader.", exc)
    logger.info("Reading the COLMAP model...")
    # 读取 COLMAP 的稀疏重建模型（cameras, images, points3D）
    # model: 包含 cameras.bin, images.bin, points3D.bin 的文件夹路径
    cameras, images, points3D = read_model(model)

    logger.info("Extracting image pairs from covisibility info...")
    pairs = [] # 存放最终生成的图片匹配对列表 [(image1, image2), ...]
    
    # 遍历模型中的每一张图片 (image_id: 图片ID, image: 图片对象)
    # 这里的 images 是数据库中已经建图成功的图片
    for image_id, image in tqdm(images.items()):
        
        # 1. 找出当前图片观测到的所有有效的 3D 点
        # image.point3D_ids 记录了该图片每个特征点对应的 3D 点 ID
        # -1 表示该特征点没有三角化成功（没有对应的 3D 点）
        matched = image.point3D_ids != -1
        points3D_covis = image.point3D_ids[matched]

        # 2. 统计共视邻居 (Covisibility Search)
        # covis: 字典，格式为 {邻居图片ID: 共有的3D点数量}
        covis = defaultdict(int)
        
        # 遍历当前图片看到的所有 3D 点
        for point_id in points3D_covis:
            # points3D[point_id].image_ids: 记录了这个 3D 点被哪些图片看到了
            for image_covis_id in points3D[point_id].image_ids:
                # 如果这个邻居不是自己，就给它投一票
                if image_covis_id != image_id:
                    covis[image_covis_id] += 1

        # 如果没有找到任何共视邻居，说明这张图是孤立的，跳过
        if len(covis) == 0:
            logger.info(f"Image {image_id} does not have any covisibility.")
            continue

        # 将邻居 ID 和对应的共视点数量转为 numpy 数组，方便排序
        covis_ids = np.array(list(covis.keys()))
        covis_num = np.array([covis[i] for i in covis_ids])

        # 3. 筛选 Top-K 邻居
        # 如果邻居数量少于要求的 num_matched（比如默认 20 个），就全部保留并按共视数量降序排列
        if len(covis_ids) <= num_matched:
            top_covis_ids = covis_ids[np.argsort(-covis_num)]
        else:
            # 如果邻居太多，只取共视点数量最多的 Top-K 个
            # np.argpartition 是部分排序算法，比全排序快，找出最大的 num_matched 个元素的索引
            ind_top = np.argpartition(covis_num, -num_matched)
            ind_top = ind_top[-num_matched:]  # 取出这 Top-K 个索引（此时内部可能是乱序的）
            
            # 对这 Top-K 个索引再进行一次真正的排序，保证输出是从最相似到较不相似
            ind_top = ind_top[np.argsort(-covis_num[ind_top])]
            
            # 根据索引取出对应的图片 ID
            top_covis_ids = [covis_ids[i] for i in ind_top]
            
            # 这是一个断言检查：确保排在第一位的确实是共视点最多的
            assert covis_num[ind_top[0]] == np.max(covis_num)

        # 4. 生成匹配对
        # 将当前图片与筛选出的 Top-K 邻居组成配对
        for i in top_covis_ids:
            # 保存的是图片名字（如 "db/123.jpg"），因为后续特征匹配脚本是用名字索引的
            pair = (image.name, images[i].name)
            pairs.append(pair)

    logger.info(f"Found {len(pairs)} pairs.")
    
    # 5. 写入输出文件 (txt)
    # 格式为每一行两个图片名，中间空格分隔： "img1.jpg img2.jpg"
    with open(output, "w") as f:
        f.write("\n".join(" ".join([i, j]) for i, j in pairs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num_matched", required=True, type=int)
    args = parser.parse_args()
    main(**args.__dict__)
