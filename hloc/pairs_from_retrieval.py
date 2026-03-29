import argparse
import collections.abc as collections
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch

from . import logger
from .utils.io import list_h5_names
from .utils.parsers import parse_image_lists
from .utils.read_write_model import read_images_binary


def parse_names(prefix, names, names_all):
    if prefix is not None:
        if not isinstance(prefix, str):
            prefix = tuple(prefix)
        names = [n for n in names_all if n.startswith(prefix)]
        if len(names) == 0:
            raise ValueError(f"Could not find any image with the prefix `{prefix}`.")
    elif names is not None:
        if isinstance(names, (str, Path)):
            names = parse_image_lists(names)
        elif isinstance(names, collections.Iterable):
            names = list(names)
        else:
            raise ValueError(
                f"Unknown type of image list: {names}."
                "Provide either a list or a path to a list file."
            )
    else:
        names = names_all
    return names


def get_descriptors(names, path, name2idx=None, key="global_descriptor"):
    if name2idx is None:
        with h5py.File(str(path), "r", libver="latest") as fd:
            desc = [fd[n][key].__array__() for n in names]
    else:
        desc = []
        for n in names:
            with h5py.File(str(path[name2idx[n]]), "r", libver="latest") as fd:
                desc.append(fd[n][key].__array__())
    return torch.from_numpy(np.stack(desc, 0)).float()


def pairs_from_score_matrix(
    scores: torch.Tensor,
    invalid: np.array,
    num_select: int,
    min_score: Optional[float] = None,
):
    assert scores.shape == invalid.shape
    if isinstance(scores, np.ndarray):
        scores = torch.from_numpy(scores)
    invalid = torch.from_numpy(invalid).to(scores.device)
    if min_score is not None:
        invalid |= scores < min_score
    scores.masked_fill_(invalid, float("-inf"))

    topk = torch.topk(scores, num_select, dim=1)
    indices = topk.indices.cpu().numpy()
    valid = topk.values.isfinite().cpu().numpy()

    pairs = []
    for i, j in zip(*np.where(valid)):
        pairs.append((i, indices[i, j]))
    return pairs


def main(
    descriptors,          # 查询图像的全局特征文件 (.h5)，如 NetVLAD 特征
    output,               # 输出的配对文件路径 (.txt)
    num_matched,          # Top-K: 每张查询图找多少个最相似的邻居 (如 50)
    query_prefix=None,    # (可选) 查询图片名前缀过滤器
    query_list=None,      # (可选) 指定只处理哪些查询图片
    db_prefix=None,       # (可选) 数据库图片名前缀过滤器
    db_list=None,         # (可选) 指定数据库图片列表
    db_model=None,        # (可选) COLMAP 参考模型路径。如果提供，只检索模型里存在的图片
    db_descriptors=None,  # (可选) 数据库图像的全局特征文件 (如果与 descriptors 分开存储)
):
    logger.info("Extracting image pairs from a retrieval database.")

    # 1. 处理数据库特征文件路径
    # HLoc 允许数据库特征分散在多个 .h5 文件中
    if db_descriptors is None:
        db_descriptors = descriptors # 如果没单独传，就默认查询和数据库在同一个文件里
    if isinstance(db_descriptors, (Path, str)):
        db_descriptors = [db_descriptors] # 统一转为列表处理
    
    # 建立映射表：图片名 -> 它在哪个 .h5 文件里
    # 这为了后续读取特征时知道去哪个文件找
    name2db = {n: i for i, p in enumerate(db_descriptors) for n in list_h5_names(p)} # ？
    db_names_h5 = list(name2db.keys())
    
    # 获取查询特征文件里所有的图片名
    query_names_h5 = list_h5_names(descriptors)

    # 2. 确定数据库图片列表 (Search Space)
    if db_model:
        # 黄金标准：如果提供了 COLMAP 模型 (db_model)，直接从 images.bin 读取
        # 这确保了我们检索到的图片一定是已经建图成功的、有 3D 位姿的图片
        images = read_images_binary(db_model / "images.bin")
        db_names = [i.name for i in images.values()]
    else:
        # 否则根据前缀或列表进行筛选
        db_names = parse_names(db_prefix, db_list, db_names_h5)
    
    if len(db_names) == 0:
        raise ValueError("Could not find any database image.")
    
    # 3. 确定查询图片列表
    query_names = parse_names(query_prefix, query_list, query_names_h5)

    # 4. 加载特征向量到内存 (Load Descriptors)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 读取数据库图片的所有全局特征 (N_db, D)
    db_desc = get_descriptors(db_names, db_descriptors, name2db)
    # 读取查询图片的所有全局特征 (N_q, D)
    query_desc = get_descriptors(query_names, descriptors)
    
    # 5. 计算相似度矩阵 (Similarity Calculation)
    # 使用爱因斯坦求和约定 (einsum) 进行矩阵乘法
    # id (query), jd (db) -> ij (score matrix)
    # 等价于 query_desc @ db_desc.T
    # 因为全局特征通常是 L2 归一化的，所以点积 = 余弦相似度
    sim = torch.einsum("id,jd->ij", query_desc.to(device), db_desc.to(device))

    # 6. 排除自匹配 (Avoid Self-Matching)
    # 如果查询集和数据库集有重叠（比如在做建图时的回环检测），
    # 我们不希望图片 A 匹配到图片 A 自己（虽然相似度是 1.0，但没意义）
    # 创建一个布尔掩码，如果名字相同则为 True
    self = np.array(query_names)[:, None] == np.array(db_names)[None]
    
    # 7. 选取 Top-K (Pairs Selection)
    # 根据相似度矩阵，为每个查询选出分数最高的 num_matched 个数据库图片
    # min_score=0 排除负相关的匹配
    pairs = pairs_from_score_matrix(sim, self, num_matched, min_score=0)
    
    # 将索引转换为真实的文件名
    pairs = [(query_names[i], db_names[j]) for i, j in pairs]

    logger.info(f"Found {len(pairs)} pairs.")
    
    # 8. 写入结果
    # 格式："query.jpg db_image.jpg"
    with open(output, "w") as f:
        f.write("\n".join(" ".join([i, j]) for i, j in pairs))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num_matched", type=int, required=True)
    parser.add_argument("--query_prefix", type=str, nargs="+")
    parser.add_argument("--query_list", type=Path)
    parser.add_argument("--db_prefix", type=str, nargs="+")
    parser.add_argument("--db_list", type=Path)
    parser.add_argument("--db_model", type=Path)
    parser.add_argument("--db_descriptors", type=Path)
    args = parser.parse_args()
    main(**args.__dict__)
