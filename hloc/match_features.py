import argparse
import pprint
from functools import partial
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Dict, List, Optional, Tuple, Union

import h5py
import torch
from tqdm import tqdm

from . import logger, matchers
from .utils.base_model import dynamic_load
from .utils.parsers import names_to_pair, names_to_pair_old, parse_retrieval

"""
A set of standard configurations that can be directly selected from the command
line using their name. Each is a dictionary with the following entries:
    - output: the name of the match file that will be generated.
    - model: the model configuration, as passed to a feature matcher.
"""
confs = {
    "superpoint+lightglue": {
        "output": "matches-superpoint-lightglue",
        "model": {
            "name": "lightglue",
            "features": "superpoint",
        },
    },
    "disk+lightglue": {
        "output": "matches-disk-lightglue",
        "model": {
            "name": "lightglue",
            "features": "disk",
        },
    },
    "aliked+lightglue": {
        "output": "matches-aliked-lightglue",
        "model": {
            "name": "lightglue",
            "features": "aliked",
        },
    },
    "superglue": {
        "output": "matches-superglue",
        "model": {
            "name": "superglue",
            "weights": "outdoor",
            "sinkhorn_iterations": 50,
        },
    },
    "superglue-fast": {
        "output": "matches-superglue-it5",
        "model": {
            "name": "superglue",
            "weights": "outdoor",
            "sinkhorn_iterations": 5,
        },
    },
    "NN-superpoint": {
        "output": "matches-NN-mutual-dist.7",
        "model": {
            "name": "nearest_neighbor",
            "do_mutual_check": True,
            "distance_threshold": 0.7,
        },
    },
    "NN-ratio": {
        "output": "matches-NN-mutual-ratio.8",
        "model": {
            "name": "nearest_neighbor",
            "do_mutual_check": True,
            "ratio_threshold": 0.8,
        },
    },
    "NN-mutual": {
        "output": "matches-NN-mutual",
        "model": {
            "name": "nearest_neighbor",
            "do_mutual_check": True,
        },
    },
    "adalam": {
        "output": "matches-adalam",
        "model": {"name": "adalam"},
    },
}


class WorkQueue:
    def __init__(self, work_fn, num_threads=1):
        self.queue = Queue(num_threads)
        self.threads = [
            Thread(target=self.thread_fn, args=(work_fn,)) for _ in range(num_threads)
        ]
        for thread in self.threads:
            thread.start()

    def join(self):
        for thread in self.threads:
            self.queue.put(None)
        for thread in self.threads:
            thread.join()

    def thread_fn(self, work_fn):
        item = self.queue.get()
        while item is not None:
            work_fn(item)
            item = self.queue.get()

    def put(self, data):
        self.queue.put(data)


class FeaturePairsDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, feature_path_q, feature_path_r):
        """
        初始化数据集
        :param pairs: 列表，包含要匹配的图像对名字，格式如 [('img1', 'img2'), ...]
        :param feature_path_q: 查询图像特征文件 (.h5) 的路径
        :param feature_path_r: 参考图像特征文件 (.h5) 的路径
        """
        self.pairs = pairs
        self.feature_path_q = feature_path_q
        self.feature_path_r = feature_path_r

    def __len__(self):
        # 返回总的图像对数量
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        根据索引读取一对图像的特征
        """
        name0, name1 = self.pairs[idx] # 获取第 idx 个图像对的文件名
        data = {}

        # --- 1. 读取第一张图 (Query) 的特征 ---
        with h5py.File(self.feature_path_q, "r") as fd:
            grp = fd[name0]
            for k, v in grp.items():
                # 将 H5 中的数据转为 NumPy 数组，再转为 PyTorch FloatTensor
                # 并在键名后加 "0" 以区分（例如 keypoints -> keypoints0）
                data[k + "0"] = torch.from_numpy(v.__array__()).float()
            
            # 伪造一个空的图像张量，某些匹配模型需要图像占位符来获取尺寸
            # grp["image_size"] 通常存的是 [宽, 高]，这里通过 [::-1] 转为 [高, 宽]
            data["image0"] = torch.empty((1,) + tuple(grp["image_size"])[::-1])

        # --- 2. 读取第二张图 (Reference) 的特征 ---
        with h5py.File(self.feature_path_r, "r") as fd:
            grp = fd[name1]
            for k, v in grp.items():
                # 键名后加 "1" 以区分（例如 keypoints -> keypoints1）
                data[k + "1"] = torch.from_numpy(v.__array__()).float()
            
            data["image1"] = torch.empty((1,) + tuple(grp["image_size"])[::-1])

        return data

    def __len__(self):
        return len(self.pairs)


def writer_fn(inp, match_path):
    pair, pred = inp
    with h5py.File(str(match_path), "a", libver="latest") as fd:
        if pair in fd:
            del fd[pair]
        grp = fd.create_group(pair)
        matches = pred["matches0"][0].cpu().short().numpy()
        grp.create_dataset("matches0", data=matches)
        if "matching_scores0" in pred:
            scores = pred["matching_scores0"][0].cpu().half().numpy()
            grp.create_dataset("matching_scores0", data=scores)


def main(
    conf: Dict,                        # 匹配器的配置字典 (如 LightGlue/SuperGlue 的参数)
    pairs: Path,                       # 包含待匹配图像对列表的文本文件路径 (pairs.txt)
    features: Union[Path, str],        # 特征文件的路径 (.h5) 或者特征的名字 (字符串)
    export_dir: Optional[Path] = None, # 导出结果的目录 (如果 features 只是名字，必须提供此项)
    matches: Optional[Path] = None,    # 输出匹配结果的文件路径 (.h5)
    features_ref: Optional[Path] = None, # 参考图像的特征文件路径 (如果为 None，则默认与 features 相同)
    overwrite: bool = False,           # 是否覆盖已存在的匹配文件
) -> Path:
    
    # 逻辑分支 A：如果用户直接提供了具体的特征文件路径
    # isinstance(features, Path): 用户传的是 Path 对象
    # Path(features).exists(): 用户传的是字符串路径且文件存在
    if isinstance(features, Path) or Path(features).exists():
        features_q = features # 将其作为查询特征路径
        
        # 这种模式下，用户必须显式提供输出路径 matches，否则报错
        if matches is None:
            raise ValueError(
                "Either provide both features and matches as Path" " or both as names."
            )
            
    # 逻辑分支 B：如果用户只提供了特征的名字 (如 "feats-superpoint-n4096") 走这里
    # 这种模式通常用于 pipeline 自动生成的流程
    else:
        # 必须提供 export_dir 来拼接完整路径
        if export_dir is None:
            raise ValueError(
                "Provide an export_dir if features is not" f" a file path: {features}."
            )
        
        # 自动拼接出特征文件路径：export_dir/名字.h5
        features_q = Path(export_dir, features + ".h5")
        
        # 如果没有指定输出文件名，则自动生成一个很长的名字
        # 格式通常是：特征名_匹配器名_配对文件名.h5
        # 例如: feats-superpoint_lightglue_pairs-query.h5
        if matches is None:
            matches = Path(export_dir, f'{features}_{conf["output"]}_{pairs.stem}.h5')

    # 处理参考特征文件 (features_ref)
    # 场景 1：建图 (Mapping) -> 查询图和参考图都在同一个文件里 -> features_ref 为 None -> 设为 features_q
    # 场景 2：定位 (Localization) -> 查询图在 query.h5，参考图在 db.h5 -> features_ref 是 db.h5
    if features_ref is None:
        features_ref = features_q
    
    # 调用真正的干活函数，开始跑网络进行匹配
    match_from_paths(conf, pairs, matches, features_q, features_ref, overwrite)

    return matches


def find_unique_new_pairs(pairs_all: List[Tuple[str]], match_path: Path = None):
    """Avoid to recompute duplicates to save time."""
    pairs = set()
    for i, j in pairs_all:
        if (j, i) not in pairs:
            pairs.add((i, j))
    pairs = list(pairs)
    if match_path is not None and match_path.exists():
        with h5py.File(str(match_path), "r", libver="latest") as fd:
            pairs_filtered = []
            for i, j in pairs:
                if (
                    names_to_pair(i, j) in fd
                    or names_to_pair(j, i) in fd
                    or names_to_pair_old(i, j) in fd
                    or names_to_pair_old(j, i) in fd
                ):
                    continue
                pairs_filtered.append((i, j))
        return pairs_filtered
    return pairs


@torch.no_grad() # 禁用梯度计算，显存占用更少，推理更快
def match_from_paths(
    conf: Dict,
    pairs_path: Path,         # 定义了哪些图片需要配对的文本文件 (pairs.txt)
    match_path: Path,         # 匹配结果的输出路径 (.h5)
    feature_path_q: Path,     # 查询图像的特征文件 (.h5)
    feature_path_ref: Path,   # 参考图像的特征文件 (.h5)
    overwrite: bool = False,  # 是否覆盖已有结果
) -> Path:
    
    # 打印当前使用的匹配配置（如 SuperGlue 的 weights, sinkhorn_iterations 等）
    logger.info(
        "Matching local features with configuration:" f"\n{pprint.pformat(conf)}"
    )

    # 1. 基础文件校验
    if not feature_path_q.exists():
        raise FileNotFoundError(f"Query feature file {feature_path_q}.")
    if not feature_path_ref.exists():
        raise FileNotFoundError(f"Reference feature file {feature_path_ref}.")
    
    # 创建输出文件的父目录
    match_path.parent.mkdir(exist_ok=True, parents=True)

    # 2. 解析配对列表
    assert pairs_path.exists(), pairs_path
    # parse_retrieval 读取文本文件，格式通常是：query_img db_img1 db_img2 ...
    pairs = parse_retrieval(pairs_path)
    # 将字典展平为列表：[(q, db1), (q, db2), ...]
    pairs = [(q, r) for q, rs in pairs.items() for r in rs]
    
    # 3. 断点续传逻辑 (核心优化)
    # 检查 match_path 中是否已经存在某些配对结果
    # 如果存在且不强制覆盖 (overwrite=False)，则从任务列表中剔除这些配对
    pairs = find_unique_new_pairs(pairs, None if overwrite else match_path)
    
    # 如果所有任务都做完了，直接返回
    if len(pairs) == 0:
        logger.info("Skipping the matching.")
        return

    # 4. 模型初始化
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 动态加载匹配模型类 (如 SuperGlue, LightGlue)
    Model = dynamic_load(matchers, conf["model"]["name"])
    # 实例化模型并移动到 GPU
    model = Model(conf["model"]).eval().to(device)

    # 5. 数据加载器
    # FeaturePairsDataset 负责从两个 .h5 特征文件中读取对应的 keypoints 和 descriptors
    dataset = FeaturePairsDataset(pairs, feature_path_q, feature_path_ref)
    loader = torch.utils.data.DataLoader(
        dataset, num_workers=5, batch_size=1, shuffle=False, pin_memory=True
    )
    
    # 6. 异步写入队列 (核心优化)
    # 磁盘 I/O (写入 H5) 远比 GPU 推理慢。为了不让 GPU 等硬盘，
    # 这里开启了一个后台线程队列 (WorkQueue)，专门负责把结果写入磁盘。
    # writer_fn 是实际执行写入的函数，使用 partial 预先固定 match_path 参数
    writer_queue = WorkQueue(partial(writer_fn, match_path=match_path), 5)

    # 7. 推理循环
    for idx, data in enumerate(tqdm(loader, smoothing=0.1)):
        # 将数据移动到 GPU (非阻塞模式加速传输)
        # 注意：不移动以 "image" 开头的字段（通常是文件名字符串等元数据）
        data = {
            k: v if k.startswith("image") else v.to(device, non_blocking=True)
            for k, v in data.items()
        }
        
        # 执行模型前向传播
        pred = model(data) # 返回数据的格式是什么？
        
        # 将当前处理的配对名转换为存储用的 key (如 "img1.jpg/img2.jpg")
        pair = names_to_pair(*pairs[idx])
        
        # 将结果放入队列，由后台线程写入磁盘
        # 主线程立即继续处理下一个 batch，不需要等待写入完成
        writer_queue.put((pair, pred))
    
    # 等待队列中所有剩余的任务写入完成
    writer_queue.join()
    logger.info("Finished exporting matches.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--export_dir", type=Path)
    parser.add_argument("--features", type=str, default="feats-superpoint-n4096-r1024")
    parser.add_argument("--matches", type=Path)
    parser.add_argument(
        "--conf", type=str, default="superglue", choices=list(confs.keys())
    )
    args = parser.parse_args()
    main(confs[args.conf], args.pairs, args.features, args.export_dir)
