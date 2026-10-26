"""只用两幅图像共视区域内的局部特征运行 HLoc 匹配器。

输入 H5 仍然是原始 SuperPoint 特征文件，本文件不会修改它们。对每个
query/map 对，先按照共视掩膜选出关键点，再把对应的描述子送入 LightGlue。
匹配完成后，``matches0`` 会扩展回原始 query 特征索引，匹配值也映射回
原始 map 特征索引。因此 ``localize_sfm`` 以及已有的 SfM 模型仍然可以
使用未裁剪的特征索引体系。
"""

from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

from . import matchers
from .utils.base_model import dynamic_load
from .utils.parsers import names_to_pair, parse_retrieval


def main(conf, pairs, features_query, features_map, matches, mask_provider, overwrite=False):
    """执行掩膜局部匹配并写入 HLoc 兼容的 matches HDF5。

    参数：
      - ``pairs``：每行一个 ``query map`` 候选对；
      - ``features_query/features_map``：原始局部特征 H5；
      - ``mask_provider``：给定图像名，返回 query/map 两个布尔掩膜，
        若返回 ``None`` 表示零共视点，该对不应出现在 pairs 中；
      - ``overwrite``：是否重算已经存在的匹配组。
    """
    pairs = Path(pairs)
    matches = Path(matches)
    matches.parent.mkdir(parents=True, exist_ok=True)
    # parse_retrieval 将两列配对文件转换成 query -> [map, ...] 字典。
    retrieval = parse_retrieval(pairs)
    image_pairs = [(q, r) for q, refs in retrieval.items() for r in refs]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Model = dynamic_load(matchers, conf["model"]["name"])
    model = Model(conf["model"]).eval().to(device)
    with h5py.File(features_query, "r") as fq, h5py.File(features_map, "r") as fm, h5py.File(matches, "a", libver="latest") as out:
        for qname, mname in tqdm(image_pairs, desc="masked feature matching"):
            pair_key = names_to_pair(qname, mname)
            if not overwrite and pair_key in out:
                continue
            qgroup, mgroup = fq[qname], fm[mname]
            # mask_provider 的返回值已经由主流程的共视投影算法计算，
            # 这里仅负责将像素掩膜应用到关键点坐标。
            qmask, mmask = mask_provider(qname, mname)
            qxy = np.asarray(qgroup["keypoints"])
            mxy = np.asarray(mgroup["keypoints"])
            # SuperPoint 坐标是 (x, y)，NumPy 图像索引是 [y, x]。
            # 先检查边界，再查掩膜，避免负坐标从数组尾部误取像素。
            qi = np.flatnonzero((qxy[:, 0] >= 0) & (qxy[:, 0] < qmask.shape[1]) & (qxy[:, 1] >= 0) & (qxy[:, 1] < qmask.shape[0]) & qmask[np.clip(qxy[:, 1].astype(int), 0, qmask.shape[0]-1), np.clip(qxy[:, 0].astype(int), 0, qmask.shape[1]-1)])
            mi = np.flatnonzero((mxy[:, 0] >= 0) & (mxy[:, 0] < mmask.shape[1]) & (mxy[:, 1] >= 0) & (mxy[:, 1] < mmask.shape[0]) & mmask[np.clip(mxy[:, 1].astype(int), 0, mmask.shape[0]-1), np.clip(mxy[:, 0].astype(int), 0, mmask.shape[1]-1)])
            if not len(qi) or not len(mi):
                # 没有可送入匹配器的点时，写入全 -1，保持 HLoc 文件格式。
                expanded = np.full(len(qxy), -1, dtype=np.int16)
                scores = np.zeros(len(qxy), dtype=np.float16)
            else:
                data = {}
                for suffix, group, indices in (("0", qgroup, qi), ("1", mgroup, mi)):
                    data[f"keypoints{suffix}"] = torch.from_numpy(np.asarray(group["keypoints"])[indices][None].astype(np.float32)).to(device)
                    desc = np.asarray(group["descriptors"])
                    # HLoc SuperPoint 通常存储为 [D, N]；LightGlue 的 HLoc
                    # 封装会再转置最后两个轴，因此这里要保持原始布局。
                    if desc.shape[0] == 256:
                        desc = desc[:, indices]
                    elif desc.shape[1] == 256:
                        desc = desc[indices].T
                    else:
                        raise ValueError(f"Unexpected descriptor shape {desc.shape} in {suffix} for {qname}, {mname}")
                    data[f"descriptors{suffix}"] = torch.from_numpy(desc[None].astype(np.float32)).to(device)
                    size = np.asarray(group["image_size"], dtype=np.float32)
                    data[f"image{suffix}"] = torch.empty((1, 1, int(size[1]), int(size[0])), device=device)
                with torch.no_grad():
                    pred = model(data)
                m0 = pred["matches0"][0].detach().cpu().numpy().astype(np.int64)
                # m0 的索引只针对裁剪后的 mi 子数组，必须映射回原始
                # map 特征索引，否则后续 SfM 3D 点查表会错位。
                expanded = np.full(len(qxy), -1, dtype=np.int16)
                valid = m0 >= 0
                expanded[qi[valid]] = mi[m0[valid]].astype(np.int16)
                scores = np.zeros(len(qxy), dtype=np.float16)
                if "matching_scores0" in pred:
                    scores[qi] = pred["matching_scores0"][0].detach().cpu().numpy().astype(np.float16)
            if pair_key in out:
                del out[pair_key]
            grp = out.create_group(pair_key)
            grp.create_dataset("matches0", data=expanded)
            grp.create_dataset("matching_scores0", data=scores)
    return matches
