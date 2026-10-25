# Hierarchical-Localization 代码流程与关键组件说明

本文结合仓库现有代码和 `代码结构.md`、`笔记.md`，说明 HLoc 的核心数据流、关键函数、输入输出格式和典型使用方法。

## 1. 总体定位流程

HLoc 使用“先粗后精”的分层定位策略：

```text
图像
  ├─ 全局特征提取（NetVLAD）
  ├─ 全局相似度检索（Top-K）
  ├─ 局部特征提取（SuperPoint）
  ├─ 局部特征匹配（LightGlue / SuperGlue / NN）
  ├─ 参考 SfM 模型中的 2D 点查找对应 3D 点
  └─ PnP + RANSAC 求查询图相机位姿
```

其中，全局检索减少需要精确匹配的候选图数量；局部特征匹配提供像素级对应；SfM 模型提供参考图像上的 3D 点和相机信息；最后由 `pycolmap` 完成绝对位姿估计。

## 2. 主要目录职责

| 目录或文件 | 作用 |
| --- | --- |
| `hloc/extract_features.py` | 读取图像、预处理、动态加载特征模型并写入 `.h5` |
| `hloc/extractors/` | 局部或全局特征提取器，如 SuperPoint、NetVLAD、D2-Net |
| `hloc/pairs_from_retrieval.py` | 根据全局描述子计算相似度并生成 Top-K 配对 |
| `hloc/match_features.py` | 读取局部特征，调用匹配器并写入匹配 `.h5` |
| `hloc/matchers/` | LightGlue、SuperGlue、最近邻、AdaLAM 等匹配器封装 |
| `hloc/triangulation.py` | 将特征和匹配导入 COLMAP，几何验证并三角化 3D 点 |
| `hloc/reconstruction.py` | 从无模型开始进行 COLMAP 增量式三维重建 |
| `hloc/localize_sfm.py` | 将 2D-2D 匹配转换为 2D-3D 对应，并执行 PnP 定位 |
| `hloc/utils/io.py` | 读写图像、HDF5 特征、HDF5 匹配和位姿结果 |
| `hloc/utils/base_model.py` | 统一模型接口和动态加载机制 |
| `hloc/pipelines/` | 不同数据集的场景化脚本；核心算法仍复用上述公共模块 |

`pipelines` 下有许多数据集专用脚本，它们主要负责路径、数据集文件名和参数配置。因此阅读或添加注释时，重点放在公共模块即可。本文以 `hloc/pipelines/Aachen_v1_1/pipeline.py` 作为代表性 pipeline。

## 3. 特征提取

### 3.1 `extract_features.main`

文件：`hloc/extract_features.py`

```python
features = extract_features.main(
    extract_features.confs["superpoint_my"],
    image_dir,
    output_dir,
)
```

主要步骤：

1. `ImageDataset` 搜索图像并转换为相对于 `image_dir` 的图片名。
2. 根据 preprocessing 配置进行灰度化、缩放、转置和 `[0, 255] -> [0, 1]` 归一化。
3. 通过 `dynamic_load(extractors, conf["model"]["name"])` 动态加载模型。
4. 使用 `DataLoader` 批量推理。
5. 将关键点坐标从缩放后的图像坐标恢复到原始图像坐标。
6. 以图片相对路径为 HDF5 group 名，写入提取结果。

### 3.2 SuperPoint 输出

`hloc/extractors/superpoint.py` 封装 SuperPoint，通常输出：

- `keypoints`：`N x 2`，关键点坐标；
- `descriptors`：局部描述子；
- `scores`：关键点检测分数；
- `image_size`：原始图像尺寸。

常用配置：

```python
feature_conf = extract_features.confs["superpoint_my"]
```

`max_keypoints` 控制最多保留多少个点，`nms_radius` 控制非极大值抑制的邻域半径，`resize_max` 控制输入图像最大边长。

### 3.3 NetVLAD 输出

```python
retrieval_conf = extract_features.confs["netvlad"]
global_features = extract_features.main(
    retrieval_conf, image_dir, output_dir
)
```

`hloc/extractors/netvlad.py` 先用 VGG16 提取 CNN 特征，再由 `NetVLADLayer` 聚合为固定长度的 `global_descriptor`。该描述子用于图像级相似度检索，不直接参与最终的像素级 PnP 计算。

## 4. 全局检索

文件：`hloc/pairs_from_retrieval.py`

```python
pairs_from_retrieval.main(
    global_features,
    output_dir / "pairs-query-netvlad50.txt",
    num_matched=50,
    query_prefix="query",
    db_model=reference_sfm,
)
```

执行过程：

1. 从 HDF5 读取查询图和数据库图的 `global_descriptor`。
2. 计算 `query_desc @ db_desc.T`。由于描述子通常已 L2 归一化，点积可视作余弦相似度。
3. 屏蔽查询图与自身的配对。
4. 对每张查询图选取相似度最高的 `num_matched` 张参考图。
5. 输出文本，每行格式为：

```text
query/image1.jpg database/image27.jpg
```

如果传入 `db_model`，数据库候选会限制为 COLMAP 模型中已注册的图像，避免检索到没有 3D 位姿的图片。

## 5. 局部匹配

文件：`hloc/match_features.py`

```python
matcher_conf = match_features.confs["superpoint+lightglue"]
matches = match_features.main(
    matcher_conf,
    pairs_path,
    feature_conf["output"],
    output_dir,
)
```

`FeaturePairsDataset` 从特征 HDF5 中读取两张图的关键点和描述子，并整理成带有 `0`、`1` 后缀的输入键：

```text
keypoints0, descriptors0, image0
keypoints1, descriptors1, image1
```

匹配器通常输出 `matches0`：长度等于图 0 关键点数量的一维数组。数组第 `i` 项是图 1 中与图 0 第 `i` 个点对应的索引；没有匹配时为 `-1`。`get_matches` 会把它转换为 `N x 2` 的 `[query_index, reference_index]` 数组。

可用匹配方式包括：

- `superpoint+lightglue`：深度学习匹配器，通常是推荐的 SuperPoint 组合；
- `superglue`：基于注意力和最优传输的匹配器；
- `NN-superpoint`、`NN-ratio`、`NN-mutual`：最近邻匹配及其过滤策略；
- `adalam`：带局部几何一致性验证的匹配器。

## 6. 参考模型与三角化

### 6.1 `triangulation.main`

文件：`hloc/triangulation.py`

该模块适用于已经有参考相机位姿的情况：

1. 从参考 COLMAP 模型建立 `database.db`。
2. 将 HDF5 关键点和匹配导入数据库。
3. 进行双视图几何验证，过滤不符合极线约束的外点。
4. 调用 `pycolmap.triangulate_points`，在固定相机位姿下生成新的 3D 点。

```python
reference_sfm = triangulation.main(
    sfm_dir,
    reference_model,
    image_dir,
    pairs,
    features,
    matches,
)
```

如果没有任何相机位姿，需要使用 `reconstruction.main` 执行从零开始的增量式 SfM，而不是直接调用三角化。

## 7. PnP 定位核心

文件：`hloc/localize_sfm.py`

### 7.1 `QueryLocalizer.localize`

该函数调用：

```python
pycolmap.estimate_and_refine_absolute_pose(
    points2D,
    points3D,
    query_camera,
    estimation_options=...,
    refinement_options=...,
)
```

输入是查询图上的 2D 像素点、参考模型中的 3D 世界坐标以及查询相机内参。内部使用 RANSAC 估计初始绝对位姿，并进行位姿细化。

### 7.2 `pose_from_cluster`

这是定位中最关键的桥接函数：

1. 读取查询图关键点。
2. 遍历 NetVLAD 检索得到的候选参考图。
3. 读取查询图与参考图之间的局部匹配。
4. 通过参考图的 COLMAP 轨迹，把参考图 2D 点索引映射到 3D 点 ID。
5. 丢弃没有关联 3D 点的匹配。
6. 合并得到查询图 2D 点到 3D 点的对应关系。
7. 调用 `QueryLocalizer.localize` 执行 PnP + RANSAC。
8. 返回位姿结果和调试日志。

注意，代码中对查询关键点加 `0.5` 是为了对齐 COLMAP 的像素中心坐标约定。特征提取和 COLMAP 使用的坐标约定必须保持一致，否则会引入像素级系统误差。

### 7.3 共视聚类

`do_covisibility_clustering` 根据候选参考图是否共享 3D 点，把候选图拆成多个连通分量。传统局部特征在候选混入多个不同地点时可能受益于该步骤；对于匹配质量较高的 SuperPoint + LightGlue，通常可以关闭：

```python
covisibility_clustering=False
```

## 8. Aachen 代表性 pipeline

文件：`hloc/pipelines/Aachen_v1_1/pipeline.py`

该脚本的实际定位链为：

```text
读取数据集路径
  -> 读取局部特征配置
  -> 读取匹配配置
  -> 使用 NetVLAD 提取全局特征
  -> 生成 query/database Top-K 配对
  -> 使用 LightGlue 做局部匹配
  -> 读取 reference_sfm
  -> localize_sfm.main 执行 PnP 定位
  -> 写出位姿 txt 和 logs.pkl
```

脚本中部分建库或三角化代码处于注释状态，因此运行前必须确认：

- `reference_sfm` 已存在并包含 `images.bin`、`cameras.bin`、`points3D.bin` 等模型文件；
- `sfm_pairs` 已经生成，或者已有可用的数据库匹配文件；
- 查询列表文件包含正确的相机模型、图像宽高和内参；
- 特征、匹配、配对文件中的图像相对路径保持一致。

## 9. HDF5 文件结构

局部特征文件通常类似：

```text
feats-superpoint-*.h5
└── image_name.jpg
    ├── keypoints
    ├── descriptors
    ├── scores
    └── image_size
```

匹配文件通常类似：

```text
matches-*.h5
└── image_a.jpg/image_b.jpg
    ├── matches0
    └── matching_scores0
```

图片名必须使用相对于数据集图像根目录的路径。不要在 `pairs.txt` 中混用绝对路径，否则 HDF5 查询和 COLMAP 模型中的图片名可能无法对应。

## 10. 输出结果

`localize_sfm.main` 主要产生两个结果：

1. 位姿文本：每行通常为

```text
image_name qw qx qy qz tx ty tz
```

2. `results_logs.pkl`：保存每个查询图的候选参考图、匹配数量、PnP 返回值和调试信息，适合定位失败分析和可视化。

## 11. 推荐阅读顺序

1. `hloc/pipelines/Aachen_v1_1/pipeline.py`：先看完整调用关系。
2. `hloc/extract_features.py`：理解图像预处理和 HDF5 特征输出。
3. `hloc/pairs_from_retrieval.py`：理解全局检索和 Top-K 配对。
4. `hloc/match_features.py`：理解局部匹配输出格式。
5. `hloc/utils/io.py`：理解 HDF5 数据如何被读取。
6. `hloc/localize_sfm.py`：重点阅读 `pose_from_cluster` 和 `QueryLocalizer.localize`。
7. `hloc/triangulation.py`、`hloc/reconstruction.py`：最后理解建图与 3D 点来源。

