import argparse
import collections.abc as collections
import glob
import pprint
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Union

import cv2
import h5py
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm

from . import extractors, logger
from .utils.base_model import dynamic_load
from .utils.io import list_h5_names, read_image
from .utils.parsers import parse_image_lists

"""
A set of standard configurations that can be directly selected from the command
line using their name. Each is a dictionary with the following entries:
    - output: the name of the feature file that will be generated.
    - model: the model configuration, as passed to a feature extractor.
    - preprocessing: how to preprocess the images read from disk.
"""
confs = {
    "superpoint_aachen": {
        "output": "feats-superpoint-n4096-r1024",
        "model": {
            "name": "superpoint",
            "nms_radius": 3,
            "max_keypoints": 4096,
        },
        "preprocessing": {
            "grayscale": True,
            "resize_max": 1024,
        },
    },
    # Resize images to 1600px even if they are originally smaller.
    # Improves the keypoint localization if the images are of good quality.
    "superpoint_max": {
        "output": "feats-superpoint-n4096-rmax1600",
        "model": {
            "name": "superpoint",
            "nms_radius": 3,
            "max_keypoints": 4096,
        },
        "preprocessing": {
            "grayscale": True,
            "resize_max": 1600,
            "resize_force": True,
        },
    },
    # 我的修改：轻量 SuperPoint 配置
     "superpoint_my": {
        "output": "feats-superpoint-n2048-rmax1024",
        "model": {
            "name": "superpoint",
            "nms_radius": 3,
            "max_keypoints": 2048,
        },
        "preprocessing": {
            "grayscale": True,
            "resize_max": 1024,
            "resize_force": True,
        },
    },
    "superpoint_inloc": {
        "output": "feats-superpoint-n4096-r1600",
        "model": {
            "name": "superpoint",
            "nms_radius": 4,
            "max_keypoints": 4096,
        },
        "preprocessing": {
            "grayscale": True,
            "resize_max": 1600,
        },
    },
    "r2d2": {
        "output": "feats-r2d2-n5000-r1024",
        "model": {
            "name": "r2d2",
            "max_keypoints": 5000,
        },
        "preprocessing": {
            "grayscale": False,
            "resize_max": 1024,
        },
    },
    "d2net-ss": {
        "output": "feats-d2net-ss",
        "model": {
            "name": "d2net",
            "multiscale": False,
        },
        "preprocessing": {
            "grayscale": False,
            "resize_max": 1600,
        },
    },
    "sift": {
        "output": "feats-sift",
        "model": {"name": "dog"},
        "preprocessing": {
            "grayscale": True,
            "resize_max": 1600,
        },
    },
    "sosnet": {
        "output": "feats-sosnet",
        "model": {"name": "dog", "descriptor": "sosnet"},
        "preprocessing": {
            "grayscale": True,
            "resize_max": 1600,
        },
    },
    "disk": {
        "output": "feats-disk",
        "model": {
            "name": "disk",
            "max_keypoints": 5000,
        },
        "preprocessing": {
            "grayscale": False,
            "resize_max": 1600,
        },
    },
    "aliked-n16": {
        "output": "feats-aliked-n16",
        "model": {
            "name": "aliked",
            "model_name": "aliked-n16",
        },
        "preprocessing": {
            "grayscale": False,
            "resize_max": 1024,
        },
    },
    # Global descriptors
    "dir": {
        "output": "global-feats-dir",
        "model": {"name": "dir"},
        "preprocessing": {"resize_max": 1024},
    },
    "netvlad": {
        "output": "global-feats-netvlad",
        "model": {"name": "netvlad"},
        "preprocessing": {"resize_max": 1024},
    },
    "openibl": {
        "output": "global-feats-openibl",
        "model": {"name": "openibl"},
        "preprocessing": {"resize_max": 1024},
    },
    "megaloc": {
        "output": "global-feats-megaloc",
        "model": {"name": "megaloc"},
        "preprocessing": {"resize_max": 1024},
    },
}


def resize_image(image, size, interp):
    if interp.startswith("cv2_"):
        interp = getattr(cv2, "INTER_" + interp[len("cv2_") :].upper())
        h, w = image.shape[:2]
        if interp == cv2.INTER_AREA and (w < size[0] or h < size[1]):
            interp = cv2.INTER_LINEAR
        resized = cv2.resize(image, size, interpolation=interp)
    elif interp.startswith("pil_"):
        interp = getattr(PIL.Image, interp[len("pil_") :].upper())
        resized = PIL.Image.fromarray(image.astype(np.uint8))
        resized = resized.resize(size, resample=interp)
        resized = np.asarray(resized, dtype=image.dtype)
    else:
        raise ValueError(f"Unknown interpolation {interp}.")
    return resized


class ImageDataset(torch.utils.data.Dataset):
    # 默认配置字典：如果没有传入对应的参数，就使用这里的默认值
    default_conf = {
        "globs": ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG"], # 默认扫描的文件后缀
        "grayscale": False,          # 是否转为灰度图（SuperPoint 等网络需要 True）
        "resize_max": None,          # 限制最大边长（如 1024 或 1600），防止显存爆炸
        "resize_force": False,       # 是否强制缩放，即使原图比 resize_max 小
        "interpolation": "cv2_area", # 缩放插值方法，cv2_area 适合缩小图像，不易产生混叠
    }

    def __init__(self, root, conf, paths=None):
        """
        初始化数据集。
        root: 图片根目录
        conf: 用户提供的配置字典（会覆盖 default_conf）
        paths: 可选。指定要读取的图片列表（文件路径、列表或文本文件）。如果不传，则自动扫描 root 下的所有图片。
        """
        # 配置合并技巧：
        # {**dict1, **dict2} 是 Python 3.5+ 的字典合并语法，conf 会覆盖 default_conf 中相同的 key
        # SimpleNamespace 将字典转换为对象，允许使用 obj.key 的方式访问，比 obj['key'] 更方便
        self.conf = conf = SimpleNamespace(**{**self.default_conf, **conf})
        self.root = root

        # 模式一：自动发现模式 (如果未指定 paths)
        if paths is None:
            paths = [] # 存放所有找到的图片路径列表
            # 遍历所有定义的后缀名 (globs)，递归查找 root 目录下的匹配文件
            for g in conf.globs: # ? query也拿来了吗
                paths += glob.glob((Path(root) / "**" / g).as_posix(), recursive=True)
            
            # 错误检查：如果目录下没找到任何图片，报错
            if len(paths) == 0:
                raise ValueError(f"Could not find any image in root: {root}.")
            
            # 去重并排序，保证每次运行的顺序一致（这对特征提取和匹配的一致性很重要）
            paths = sorted(set(paths))
            
            # 将绝对路径转换为相对于 root 的相对路径（作为后续 H5 文件的 key）
            self.names = [Path(p).relative_to(root).as_posix() for p in paths]
            logger.info(f"Found {len(self.names)} images in root {root}.")
            
        # 模式二：指定列表模式 (如果指定了 paths)
        else:
            # 如果 paths 是一个文件路径（如 "queries.txt"），则解析该文件获取图片列表
            if isinstance(paths, (Path, str)):
                self.names = parse_image_lists(paths)
            # 如果 paths 已经是一个列表/迭代器
            elif isinstance(paths, collections.Iterable):
                self.names = [p.as_posix() if isinstance(p, Path) else p for p in paths]
            else:
                raise ValueError(f"Unknown format for path argument {paths}.")

            # 完整性检查：确保列表里的每一张图在硬盘上都真实存在
            for name in self.names:
                if not (root / name).exists():
                    raise ValueError(f"Image {name} does not exists in root: {root}.")

    def __getitem__(self, idx):
        """
        读取并预处理单张图片。
        """
        name = self.names[idx]
        
        # 1. 读取图像
        # read_image 是 HLoc 封装的函数，通常使用 OpenCV 读取
        image = read_image(self.root / name, self.conf.grayscale)
        
        # 转换数据类型为 float32，这是深度学习模型的标准输入类型
        image = image.astype(np.float32)
        
        # 获取原始尺寸 (宽, 高)。注意 image.shape 是 (H, W, C)，所以切片 [:2] 拿到 (H, W)，[::-1] 翻转为 (W, H)
        size = image.shape[:2][::-1]

        # 2. 图像缩放逻辑
        if self.conf.resize_max and (
            self.conf.resize_force or max(size) > self.conf.resize_max
        ):
            # 计算缩放比例：最大边长 / 当前最大边长
            scale = self.conf.resize_max / max(size)
            
            # 计算新的尺寸，四舍五入取整
            size_new = tuple(int(round(x * scale)) for x in size)
            
            # 执行缩放。cv2_area 插值在降采样时能保留更多细节
            image = resize_image(image, size_new, self.conf.interpolation)

        # 3. 维度变换与归一化
        if self.conf.grayscale:
            # 如果是灰度图，shape 是 (H, W)，需要增加一个维度变成 (1, H, W)
            image = image[None]
        else:
            # 如果是彩图，shape 是 (H, W, 3)，需要转置为 PyTorch 格式 (3, H, W)
            image = image.transpose((2, 0, 1)) 
            
        # 归一化：将像素值从 [0, 255] 映射到 [0.0, 1.0]
        image = image / 255.0

        # 返回字典：包含处理后的图像张量和原始尺寸（用于后续坐标恢复）
        data = {
            "image": image,
            "original_size": np.array(size),
        }
        return data

    def __len__(self):
        return len(self.names)


@torch.no_grad()
def main(
    conf: Dict,                                    # 配置字典（如模型名称、参数等）
    image_dir: Path,                               # 图片所在的根目录
    export_dir: Optional[Path] = None,             # 结果导出的目录
    as_half: bool = True,                          # 是否将 float32 转为 float16 以节省存储空间
    image_list: Optional[Union[Path, List[str]]] = None, # 指定要处理的图片列表（可选）
    feature_path: Optional[Path] = None,           # 直接指定输出文件路径（覆盖 export_dir 逻辑）
    overwrite: bool = False,                       # 是否覆盖已存在的特征文件
) -> Path:
   # 打印当前的提取配置信息
    logger.info(
        "Extracting local features with configuration:" f"\n{pprint.pformat(conf)}"
    )

    # 初始化数据集：负责读取图片、预处理（如缩放、灰度化）
    dataset = ImageDataset(image_dir, conf["preprocessing"], image_list)
    
    # 确定输出文件路径：如果没有指定 feature_path，则根据配置生成，例如 "outputs/feats-superpoint-n4096-r1024.h5"(局部特征提取)或"global-feats-netvlad.h5"(全局特征提取)
    if feature_path is None:
        feature_path = Path(export_dir, conf["output"] + ".h5")
    
    # 创建父目录
    feature_path.parent.mkdir(exist_ok=True, parents=True)

    # 断点续传逻辑：
    # 如果文件存在且不强制覆盖，则读取文件中已有的图片名，并在本次提取中跳过它们
    # 返回已存在的图片名列表
    skip_names = set(
        list_h5_names(feature_path) if feature_path.exists() and not overwrite else ()
    )
    
    # 过滤掉已经处理过的图片
    dataset.names = [n for n in dataset.names if n not in skip_names]
    
    # 如果所有图片都处理过了，直接返回
    if len(dataset.names) == 0:
        logger.info("Skipping the extraction.")
        return feature_path

    # 选择计算设备：优先使用 GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 动态加载模型类：根据 conf["model"]["name"] (这里是 'superpoint') 从 extractors 字典中加载对应的类
    Model = dynamic_load(extractors, conf["model"]["name"])
    
    # 实例化模型，设置为评估模式 (eval)，并移动到 GPU
    model = Model(conf["model"]).eval().to(device)

    # 创建数据加载器：单进程加载，pin_memory 加速数据传输到 GPU
    loader = torch.utils.data.DataLoader(
        dataset, num_workers=1, shuffle=False, pin_memory=True
    )

    # 开始遍历图片进行推理
    for idx, data in enumerate(tqdm(loader)):
        # 获取当前图片的相对路径名称（作为 HDF5 中的 group key）
        name = dataset.names[idx]
        
        # 1. 模型推理：将图片送入网络
        # non_blocking=True 允许数据传输和计算重叠
        pred = model({"image": data["image"].to(device, non_blocking=True)})
        
        # 将预测结果从 GPU 转回 CPU 并转为 numpy 数组
        # k: 键名 (如 'keypoints', 'descriptors', 'scores')
        # v[0]: 取 batch 中的第一个元素（因为 batch_size=1）
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

        # 获取图片的原始尺寸 (宽, 高)
        pred["image_size"] = original_size = data["original_size"][0].numpy()
        
        # 2. 坐标还原逻辑 (非常重要)：
        # 如果模型输出了关键点，需要将其坐标从“预处理后的大小”映射回“原始图片大小”
        if "keypoints" in pred:
            # 获取预处理后的图片尺寸（即输入进网络的尺寸）
            size = np.array(data["image"].shape[-2:][::-1])
            
            # 计算缩放比例：原始尺寸 / 网络输入尺寸
            scales = (original_size / size).astype(np.float32)
            
            # 坐标变换公式：(x + 0.5) * scale - 0.5
            # 这是为了保证像素中心对齐，减少精度误差
            pred["keypoints"] = (pred["keypoints"] + 0.5) * scales[None] - 0.5
            
            # 如果预测结果中包含尺度信息（如 SIFT 的 scale），也要相应缩放
            if "scales" in pred:
                pred["scales"] *= scales.mean()
            
            # 添加关键点的不确定性 (uncertainty)，通常由检测噪声决定
            # 这在后续的建图优化中可能用到
            uncertainty = getattr(model, "detection_noise", 1) * scales.mean()

        # 3. 存储优化：半精度转换
        # 如果开启 as_half，将 float32 数据转为 float16，文件体积减半
        if as_half:
            for k in pred:
                dt = pred[k].dtype
                if (dt == np.float32) and (dt != np.float16):
                    pred[k] = pred[k].astype(np.float16)

        # 4. 写入 HDF5 文件
        # 使用 "a" (append) 模式，libver="latest" 使用最新格式以提高性能
        with h5py.File(str(feature_path), "a", libver="latest") as fd:
            try:
                # 如果该图片的数据已存在（在 overwrite=True 时可能发生），先删除旧数据
                if name in fd:
                    del fd[name]
                
                # 创建一个 Group，名字是图片路径
                grp = fd.create_group(name)
                
                # 将预测结果的每一项 (keypoints, descriptors 等) 存为 Dataset
                for k, v in pred.items():
                    grp.create_dataset(k, data=v)
                
                # 将不确定性写入 keypoints 的属性 (Attribute) 中
                if "keypoints" in pred:
                    grp["keypoints"].attrs["uncertainty"] = uncertainty
            
            except OSError as error:
                # 错误处理：通常是磁盘空间不足
                if "No space left on device" in error.args[0]:
                    logger.error(
                        "Out of disk space: storing features on disk can take "
                        "significant space, did you enable the as_half flag?"
                    )
                    # 发生错误时清理部分写入的数据
                    del grp, fd[name]
                raise error

        # 手动删除预测字典，释放内存
        del pred

    logger.info("Finished exporting features.")
    return feature_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--export_dir", type=Path, required=True)
    parser.add_argument(
        "--conf", type=str, default="superpoint_aachen", choices=list(confs.keys())
    )
    parser.add_argument("--as_half", action="store_true")
    parser.add_argument("--image_list", type=Path)
    parser.add_argument("--feature_path", type=Path)
    args = parser.parse_args()
    main(
        confs[args.conf],
        args.image_dir,
        args.export_dir,
        args.as_half,
        args.image_list,
        args.feature_path,
    )
