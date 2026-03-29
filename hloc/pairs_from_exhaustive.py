import argparse
import collections.abc as collections
from pathlib import Path
from typing import List, Optional, Union

from . import logger
from .utils.io import list_h5_names
from .utils.parsers import parse_image_lists


def main(
    output: Path,                                      # 输出文件路径 (生成的配对列表 .txt)
    image_list: Optional[Union[Path, List[str]]] = None, # 查询图像列表 (可以是文件路径，也可以是列表)
    features: Optional[Path] = None,                   # 查询图像的特征文件 (.h5)，如果没有提供 list，就从这里读取所有图片名
    ref_list: Optional[Union[Path, List[str]]] = None,   # (可选) 参考/数据库图像列表
    ref_features: Optional[Path] = None,                 # (可选) 参考图像的特征文件
):
    # --- 第一步：确定“查询图像” (Query Images) 有哪些 ---
    if image_list is not None:
        if isinstance(image_list, (str, Path)):
            names_q = parse_image_lists(image_list)    # 从文本文件解析
        elif isinstance(image_list, collections.Iterable):
            names_q = list(image_list)                 # 直接使用传入的列表
        else:
            raise ValueError(f"Unknown type for image list: {image_list}")
    elif features is not None:
        names_q = list_h5_names(features)              # 从 .h5 特征文件中读取所有键名作为图片名
    else:
        raise ValueError("Provide either a list of images or a feature file.")

    # --- 第二步：确定“参考图像” (Reference Images) 有哪些 ---
    self_matching = False # 标记：是否是“自己匹配自己” (Mapping 模式)
    
    if ref_list is not None:
        # 如果明确指定了参考图像列表 (例如 Localization 模式：Query vs Database)
        if isinstance(ref_list, (str, Path)):
            names_ref = parse_image_lists(ref_list)
        elif isinstance(image_list, collections.Iterable): # 注意：源代码这里可能是个笔误，应该检查 ref_list，但逻辑是通的
            names_ref = list(ref_list)
        else:
            raise ValueError(f"Unknown type for reference image list: {ref_list}")
    elif ref_features is not None:
        names_ref = list_h5_names(ref_features)
    else:
        # --- 关键逻辑：自我匹配模式 ---
        # 如果没有提供任何参考图像，说明我们是在做 SfM 建图
        # 这时候参考集就是查询集本身 (Ref = Query)
        self_matching = True
        names_ref = names_q

    # --- 第三步：生成配对 (核心算法) ---
    pairs = []
    # 双重循环遍历
    for i, n1 in enumerate(names_q):
        for j, n2 in enumerate(names_ref):
            # 如果是自我匹配模式 (Mapping)，我们需要避免：
            # 1. 自己匹配自己 (i == j) -> 没有意义
            # 2. 重复匹配 (i > j) -> 比如 A配B 和 B配A 是一样的，只算一次
            if self_matching and j <= i:
                continue
            
            # 如果不是自我匹配 (Localization)，则 Query 的每一张图都要和 Ref 的每一张图匹配
            pairs.append((n1, n2))

    logger.info(f"Found {len(pairs)} pairs.")
    
    # --- 第四步：写入结果 ---
    # 格式：image1.jpg image2.jpg
    with open(output, "w") as f:
        f.write("\n".join(" ".join([i, j]) for i, j in pairs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image_list", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--ref_list", type=Path)
    parser.add_argument("--ref_features", type=Path)
    args = parser.parse_args()
    main(**args.__dict__)
