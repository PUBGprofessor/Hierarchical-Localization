from pathlib import Path
from typing import Mapping, Tuple

import cv2
import h5py
import numpy as np
import pycolmap

from .parsers import names_to_pair, names_to_pair_old


def read_image(path, grayscale=False):
    if grayscale:
        mode = cv2.IMREAD_GRAYSCALE
    else:
        mode = cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError(f"Cannot read image {path}.")
    if not grayscale and len(image.shape) == 3:
        image = image[:, :, ::-1]  # BGR to RGB
    return image


def list_h5_names(path):
    names = []
    # 以只读模式打开 .h5 文件，libver="latest" 是为了提高读取性能
    with h5py.File(str(path), "r", libver="latest") as fd:

        # 定义一个内部回调函数，用于遍历 HDF5 的树状结构
        def visit_fn(_, obj):
            # 如果当前对象是一个数据集（Dataset），说明它存储了具体的特征数据（如 keypoints 或 descriptors）
            if isinstance(obj, h5py.Dataset):
                # 获取该数据集的父节点名称（即图片路径），并去掉前导和尾随的斜杠
                # 例如：从 "/mapping/001.jpg/keypoints" 中提取出 "mapping/001.jpg"
                names.append(obj.parent.name.strip("/"))

        # 使用 visititems 深度优先遍历文件中的所有对象
        fd.visititems(visit_fn)
        
    # 使用 set 去重（因为一张图片可能有多个 Dataset，如 keypoints 和 descriptors），最后转回列表
    return list(set(names))

def get_keypoints(
    path: Path, name: str, return_uncertainty: bool = False
) -> np.ndarray:
    with h5py.File(str(path), "r", libver="latest") as hfile:
        dset = hfile[name]["keypoints"]
        p = dset.__array__()
        uncertainty = dset.attrs.get("uncertainty")
    if return_uncertainty:
        return p, uncertainty
    return p


def find_pair(hfile: h5py.File, name0: str, name1: str):
    pair = names_to_pair(name0, name1)
    if pair in hfile:
        return pair, False
    pair = names_to_pair(name1, name0)
    if pair in hfile:
        return pair, True
    # older, less efficient format
    pair = names_to_pair_old(name0, name1)
    if pair in hfile:
        return pair, False
    pair = names_to_pair_old(name1, name0)
    if pair in hfile:
        return pair, True
    raise ValueError(
        f"Could not find pair {(name0, name1)}... "
        "Maybe you matched with a different list of pairs? "
    )


def get_matches(path: Path, name0: str, name1: str) -> Tuple[np.ndarray]:
    with h5py.File(str(path), "r", libver="latest") as hfile:
        pair, reverse = find_pair(hfile, name0, name1)
        matches = hfile[pair]["matches0"].__array__()
        scores = hfile[pair]["matching_scores0"].__array__()
    idx = np.where(matches != -1)[0]
    matches = np.stack([idx, matches[idx]], -1)
    if reverse:
        matches = np.flip(matches, -1)
    scores = scores[idx]
    return matches, scores


def write_poses(
    poses: Mapping[str, pycolmap.Rigid3d], path: str, prepend_camera_name: bool
):
    with open(path, "w") as f:
        for query, t in poses.items():
            qvec = " ".join(map(str, t.rotation.quat[[3, 0, 1, 2]]))
            tvec = " ".join(map(str, t.translation))
            name = query.split("/")[-1]
            if prepend_camera_name:
                name = query.split("/")[-2] + "/" + name
            f.write(f"{name} {qvec} {tvec}\n")
