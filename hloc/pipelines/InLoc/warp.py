"""InLoc 共视裁剪的离线可视化和中间数据导出。

本脚本复用 ``covis.py`` 中与 pipeline_with_warp 完全相同的 XYZcut 采样、
坐标变换、query 投影和凸包掩膜逻辑。它不执行特征匹配或 PnP，只保存：

* 每个 query/map pair 的共享点投影 ``.npz``；
* 共视统计和输入信息 ``*_metadata.json``；
* 掩膜后的 query/map 图像，以及标出投影点的叠加图；
* 每个 query 的候选汇总和全局汇总 JSON。

运行（建议先只看一张 query）：

    python -m hloc.pipelines.InLoc.warp --query iphone7/IMG_0764.JPG

默认使用已转换的 InLoc retrieval pairs 和 reloc3r 初始位姿。pairs 文件中的
候选顺序被视为检索排序，因此 ``--top-k`` 直接截取前 k 个候选。
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image as PILImage, ImageDraw
from tqdm import tqdm

from ... import logger
from ...utils.parsers import parse_retrieval
from .covis import compute_pair_geometry, read_initial_poses


def masked_image(image_path, mask):
    """保留掩膜内 RGB 像素，掩膜外置黑。"""
    image = PILImage.open(image_path).convert("RGB")
    if image.size != (mask.shape[1], mask.shape[0]):
        raise ValueError(
            f"Image/mask size mismatch for {image_path}: "
            f"image={image.size}, mask={(mask.shape[1], mask.shape[0])}"
        )
    output = np.asarray(image).copy()
    output[~mask] = 0
    return PILImage.fromarray(output)


def overlay_points(image_path, points, color=(255, 48, 48), radius=4):
    """在原图上绘制共视点投影，返回用于保存的 RGB PIL 图像。"""
    image = PILImage.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for x, y in np.asarray(points):
        x, y = float(x), float(y)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    return image


def save_empty_pair(pair_dir, query_name, map_name, margin, warning):
    """保存无几何结果的 NPZ 和 JSON，保持与有几何 pair 相同的目录结构。"""
    pair_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        pair_dir / f"{Path(query_name).stem}_{Path(map_name).stem}_points.npz",
        xyz_world=np.empty((0, 3), dtype=np.float32),
        query_xy=np.empty((0, 2), dtype=np.float32),
        map_xy=np.empty((0, 2), dtype=np.float32),
    )
    record = {
        "query": query_name,
        "map": map_name,
        "shared_points": 0,
        "margin_px": margin,
        "status": "no_geometry",
        "warning": warning,
    }
    metadata = pair_dir / f"{Path(map_name).stem}_metadata.json"
    metadata.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return record


def process_pair(dataset, query_name, map_name, initial_poses, pair_dir, margin):
    """处理一个 InLoc pair，保存点、掩膜、叠加图和 JSON 统计。"""
    pair_dir.mkdir(parents=True, exist_ok=True)
    try:
        geometry = compute_pair_geometry(
            dataset, query_name, map_name, initial_poses, margin=margin
        )
    except KeyError:
        # 与 pipeline 的普通 HLoc 回退一致：没有初始位姿时不做共视裁剪，
        # 可视化中保存全图版本，明确标记该 pair 没有几何依据。
        query_path = dataset / query_name
        map_path = dataset / map_name
        query = PILImage.open(query_path).convert("RGB")
        map_image = PILImage.open(map_path).convert("RGB")
        query.save(pair_dir / f"query_{Path(query_name).name}")
        map_image.save(pair_dir / f"map_{Path(map_name).name}")
        record = save_empty_pair(
            pair_dir,
            query_name,
            map_name,
            margin,
            "initial_pose_missing; full-image HLoc fallback",
        )
        record.update(status="full_image_fallback", query_mask_pixels=query.width * query.height,
                      map_mask_pixels=map_image.width * map_image.height)
        (pair_dir / f"{Path(map_name).stem}_metadata.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return record

    if geometry is None:
        return save_empty_pair(
            pair_dir,
            query_name,
            map_name,
            margin,
            "no valid XYZcut points projected into the query view",
        )

    query_mask = geometry["query_mask"]
    map_mask = geometry["map_mask"]
    query_path = dataset / query_name
    map_path = dataset / map_name
    masked_image(query_path, query_mask).save(pair_dir / f"query_{Path(query_name).name}")
    masked_image(map_path, map_mask).save(pair_dir / f"map_{Path(map_name).name}")
    overlay_points(query_path, geometry["query_xy"]).save(
        pair_dir / f"query_{Path(query_name).stem}_points.png"
    )
    overlay_points(map_path, geometry["map_xy"], color=(48, 220, 80)).save(
        pair_dir / f"map_{Path(map_name).stem}_points.png"
    )
    np.savez_compressed(
        pair_dir / f"{Path(query_name).stem}_{Path(map_name).stem}_points.npz",
        xyz_world=geometry["xyz_world"],
        query_xy=geometry["query_xy"],
        map_xy=geometry["map_xy"],
    )
    record = {
        "query": query_name,
        "map": map_name,
        "status": "covisible",
        "shared_points": int(len(geometry["query_xy"])),
        "margin_px": margin,
        "query_mask_pixels": int(query_mask.sum()),
        "map_mask_pixels": int(map_mask.sum()),
        "query_image_size": [int(query_mask.shape[1]), int(query_mask.shape[0])],
        "map_image_size": [int(map_mask.shape[1]), int(map_mask.shape[0])],
    }
    (pair_dir / f"{Path(map_name).stem}_metadata.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(r"D:\Datasets\InLoc"))
    parser.add_argument(
        "--pairs",
        type=Path,
        default=Path("pairs/inloc/pairs-query-netvlad20-local.txt"),
        help="Retrieval pairs in local HDF5 naming format",
    )
    parser.add_argument(
        "--initial-poses",
        type=Path,
        default=Path(r"D:\project\reloc3r\output\InLoc\query_poses.txt"),
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/inloc_warp"))
    parser.add_argument("--query", default=None, help="Only process this query image")
    parser.add_argument("--map", dest="map_name", default=None, help="Process one map candidate")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--margin", type=int, default=80)
    args = parser.parse_args()
    if args.top_k < 1 or args.margin < 0:
        parser.error("--top-k must be positive and --margin must be non-negative")

    dataset = args.dataset.resolve()
    pairs_path = args.pairs.resolve()
    poses_path = args.initial_poses.resolve()
    if not dataset.is_dir() or not pairs_path.is_file():
        raise FileNotFoundError(f"Dataset or pairs file missing: {dataset}, {pairs_path}")
    retrieval = parse_retrieval(pairs_path)
    initial_poses = read_initial_poses(poses_path) if poses_path.is_file() else {}
    query_names = [args.query] if args.query else list(retrieval)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    all_results = []
    for query_name in tqdm(query_names, desc="Visualizing InLoc queries", unit="query"):
        if query_name not in retrieval:
            raise KeyError(f"Query {query_name!r} is not present in {pairs_path}")
        candidates = retrieval[query_name][:args.top_k]
        if args.map_name:
            if args.map_name not in retrieval[query_name]:
                raise KeyError(f"Map {args.map_name!r} is not a candidate for {query_name!r}")
            candidates = [args.map_name]
        query_dir = output / Path(query_name).stem
        query_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for rank, map_name in enumerate(candidates, 1):
            pair_dir = query_dir / f"rank{rank:02d}_{Path(map_name).stem}"
            record = process_pair(
                dataset, query_name, map_name, initial_poses, pair_dir, args.margin
            )
            record.update(rank=rank)
            (pair_dir / f"{Path(map_name).stem}_metadata.json").write_text(
                json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            records.append(record)
            all_results.append(record)
        (query_dir / f"retrieval_top{len(candidates)}.json").write_text(
            json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        logger.info("InLoc warp visualization: %s, processed %d candidates", query_name, len(records))

    (output / "retrieval_summary.json").write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
