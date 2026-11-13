"""InLoc 数据集端到端定位 pipeline。

总体输入：InLoc 数据目录、预计算的 query-db 检索 pairs 文件，以及 HLoc 配置对应
的模型权重。数据目录需包含 pairs 引用的图像、数据库图对应的 XYZcut ``.mat`` 文件、
扫描变换文本。总体输出：``outputs`` 下的局部特征/匹配 HDF5 中间文件、逐查询位姿
结果文本和定位诊断 pickle 日志。详细目录和参数见下方 ``run`` 与 ``main``。
"""

import argparse
from pathlib import Path

from ... import extract_features, localize_inloc, logger, match_features


def run(args):
    """按 InLoc notebook 的顺序执行特征提取、匹配和位姿定位。

    需要的输入文件：
    - ``args.dataset`` 下的 InLoc 数据集。查询图和数据库图都由检索文件中的
      相对路径引用；数据库图还需要对应的 ``.mat`` 点云深度图，以及
      ``database/alignments/<楼层>/transformations/`` 下的扫描变换文本。
    - ``args.pairs``：预计算的查询图到数据库图检索列表，默认使用
      ``pairs/inloc/pairs-query-netvlad20-local.txt``（已移除数据集 split 前缀，
      与当前特征 HDF5 中的图像组名一致）。
    - 特征提取器、匹配器配置及其所需模型权重，由 HLoc 配置和运行环境提供。

    产生的文件：
    - ``args.outputs`` 下的局部特征 HDF5 文件和匹配结果 HDF5 文件。
    - ``InLoc_hloc_superpoint+superglue_netvlad<num_loc>.txt``：查询位姿结果。
    - 与位姿文件同名并带 ``_logs.pkl`` 后缀的定位诊断日志。
    """
    dataset = args.dataset
    outputs = args.outputs
    loc_pairs = args.pairs
    results = outputs / f"InLoc_hloc_superpoint+superglue_netvlad{args.num_loc}.txt"
    # results = outputs / f"InLoc_hloc_superpoint+lightglue_megaloc{args.num_loc}.txt"

    # 这两个默认配置与原 pipeline_InLoc.ipynb 相同；CLI 选项可替换配置名。
    feature_conf = extract_features.confs[args.feature]
    matcher_conf = match_features.confs[args.matcher]

    # 对 dataset 中检索涉及的图像提取局部特征，返回特征 HDF5 文件路径。
    features = extract_features.main(feature_conf, dataset, outputs)
    # 根据 pairs 文件列出的 query-db 图像对匹配局部特征，返回匹配 HDF5 路径。
    matches = match_features.main(
        matcher_conf, loc_pairs, feature_conf["output"], outputs
    )
    # 利用数据库扫描的 XYZcut 将数据库关键点反投影到 3D，再估计每张查询图位姿。
    localize_inloc.main(
        dataset,
        loc_pairs,
        features,
        matches,
        results,
        skip_matches=args.skip_matches,
    )
    logger.info("Localization results saved to %s", results)


def main():
    """解析命令行参数并启动 ``run``。

    默认输入为 ``D:\\Datasets\\InLoc`` 和
    ``pairs/inloc/pairs-query-netvlad20-local.txt``；
    默认中间产物及定位输出写入 ``outputs/inloc``。输出目录由各 HLoc 步骤按需创建。
    """
    parser = argparse.ArgumentParser(
        description="Run the HLoc localization pipeline on InLoc."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        # default=Path("datasets/inloc"),
        default=Path(r"D:\Datasets\InLoc"),
        help="Path to the InLoc dataset, default: %(default)s",
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        # 原始 pairs 使用 query/ 和 database/ 前缀，与当前特征 HDF5 的组名不匹配。
        default=Path("pairs/inloc/pairs-query-netvlad20-local.txt"),
        help="Precomputed query-to-database retrieval pairs, default: %(default)s",
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path("outputs/inloc"),
        help="Path to the output directory, default: %(default)s",
    )
    parser.add_argument(
        "--feature",
        # default="superpoint_inloc",
        default="superpoint_my",
        choices=extract_features.confs.keys(),
        help="Feature extractor configuration, default: %(default)s",
    )
    parser.add_argument(
        "--matcher",
        # default="superglue",
        default="superpoint+lightglue",
        choices=match_features.confs.keys(),
        help="Feature matcher configuration, default: %(default)s",
    )
    parser.add_argument(
        "--num-loc",
        type=int,
        # default=40,
        default=20,
        help="Number of retrieved database images encoded in the pairs filename, default: %(default)s",
    )
    parser.add_argument(
        "--skip-matches",
        type=int,
        default=20,
        help="Skip database images with fewer matches, default: %(default)s",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
