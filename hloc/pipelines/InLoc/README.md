# InLoc localization

This pipeline follows the steps in the repository's `pipeline_InLoc.ipynb`:
extract SuperPoint features, match the precomputed NetVLAD retrieval pairs with
SuperGlue, and estimate query poses from the InLoc 3D lidar scans.

Place the InLoc dataset in `datasets/inloc/` and use the precomputed retrieval
pairs in `pairs/inloc/pairs-query-netvlad40.txt`. Run from the repository root:

```bash
python -m hloc.pipelines.InLoc.pipeline
```

The defaults use `superpoint_inloc`, `superglue`, NetVLAD top-40 pairs, and
skip database images with fewer than 20 matches. Override paths or settings with
the command-line options, for example:

```bash
python -m hloc.pipelines.InLoc.pipeline --dataset /path/to/inloc --outputs outputs/inloc
```

The localization results and diagnostic logs are written under `outputs/inloc/`.
