"""将 InLoc 检索 pairs 中的标准目录前缀转换为本地特征文件使用的图像名。

此仓库当前 InLoc 特征 HDF5 的组名相对 ``D:\\Datasets\\InLoc`` 保存，形如
``iphone7/...`` 和 ``cutouts/...``；部分预计算 pairs 则使用
``query/iphone7/...`` 和 ``database/cutouts/...``。本脚本移除开头的
``query/``、``database/``，保留图像名及一行一对的文件格式。

示例：
    python -m hloc.pipelines.InLoc.prepare_pairs
    python -m hloc.pipelines.InLoc.prepare_pairs --input pairs/inloc/pairs-query-netvlad20.txt --output pairs/inloc/pairs-query-netvlad20-local.txt
"""

import argparse
from pathlib import Path


PREFIXES = ("query/", "database/")


def normalize_name(name):
    """移除图像相对路径开头的 InLoc split 前缀，其余部分保持原样。"""
    name = name.replace("\\", "/")
    for prefix in PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def convert_pairs(input_path, output_path):
    """转换 pairs 文本中的两列图像名并写入新文件。

    输入每个非空行应包含两个空白分隔字段：query 名和 database 名。
    输出保持相同的行数、字段顺序和配对关系；空行会忽略。源文件不会被覆盖。
    返回转换的图像对数量。
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("输入和输出路径相同；请使用新输出文件名以保留原始 pairs。")

    converted = []
    with input_path.open("r", encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 2:
                raise ValueError(
                    f"第 {line_number} 行应有两个图像名，实际得到 {len(fields)} 列。"
                )
            converted.append(" ".join(normalize_name(name) for name in fields))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(converted) + "\n", encoding="utf-8")
    return len(converted)


def main():
    """解析输入输出路径，执行转换并报告转换的图像对数。"""
    parser = argparse.ArgumentParser(
        description="Remove query/ and database/ prefixes from InLoc pairs names."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("pairs/inloc/pairs-query-netvlad20.txt"),
        help="Input retrieval pairs file, default: %(default)s",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pairs/inloc/pairs-query-netvlad20-local.txt"),
        help="Output converted pairs file, default: %(default)s",
    )
    args = parser.parse_args()
    count = convert_pairs(args.input, args.output)
    print(f"Converted {count} image pairs: {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
