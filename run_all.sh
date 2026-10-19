#!/bin/bash

# ================= 配置区域 =================
# 1. 基础工作目录 (存放数据集和你的辅助 Python 脚本)
BASE_DIR="/home/ljh/workspace/dataset/KingsCollege"

# 2. Ground Truth 真实位姿文件路径
GT_FILE="$BASE_DIR/seq1_test131/sparse/0/images.txt"

# 3. 想要批量运行的数据集名称列表 (用空格换行分隔)
DATASETS=(
    # "seq1_raw60"
    # "seq1_raw60_render70"
    # "seq1_raw40"
    # "seq1_raw40_render90"
    # "seq1_raw20"
    # "seq1_raw20_render110"
    # "seq1_raw10"
    # "seq1_raw10_render120"
    "seq1_raw130"
    # 在这里随时添加新的数据集...
)
# ============================================

# 设置：遇到任何报错立刻终止脚本
set -e

echo "🚀 正在从 Hierarchical-Localization 目录启动批量自动化流程..."

for DATASET_NAME in "${DATASETS[@]}"; do
    echo ""
    echo "=========================================================="
    echo " 🎯 正在处理数据集: ${DATASET_NAME} "
    echo "=========================================================="
    
    # 动态拼接当前数据集的各项路径
    DATASET_DIR="$BASE_DIR/$DATASET_NAME"
    IMAGES_TXT="$DATASET_DIR/3D-models/$DATASET_NAME/images.txt"
    EST_FILE="$DATASET_DIR/${DATASET_NAME}_hloc_superpoint+superglue_netvlad10.txt"

    # 1. 修复 images.txt 前缀 (精确指向 BASE_DIR 下的脚本)
    echo "🔧 [1/5] 执行 add_prefix.py..."
    python "$BASE_DIR/add_prefix.py" --images_txt "$IMAGES_TXT" --prefix "mapping/a"

    # 2. 强力清理二进制文件 
    echo "🧹 [2/5] 清理残留的 .bin 模型文件..."
    python "$BASE_DIR/remove_bins.py" --dataset_dir "$DATASET_DIR/3D-models/$DATASET_NAME" --target "*.bin"

    # 3. 生成带内参的 query 列表
    echo "📝 [3/5] 执行 add_query_list.py..."
    python "$BASE_DIR/add_query_list.py" --dataset_dir "$DATASET_DIR"

    # 4. 运行 Hloc 定位流水线 (当前目录就是 Hloc 根目录，直接调用包)
    echo "📍 [4/5] 开始执行 Hloc Pipeline 定位..."
    python -m hloc.pipelines.kingscollege_seq1.pipeline --dataset "$DATASET_DIR"

    # 5. 评估精度并生成报告
    echo "📊 [5/5] 开始执行位姿评估 evaluate_poses.py..."
    python "$BASE_DIR/evaluate_poses.py" \
        --est_file "$EST_FILE" \
        --gt_file "$GT_FILE" \
        --out_dir "$DATASET_DIR"

    echo "✅ 数据集 ${DATASET_NAME} 的所有流程已顺利跑完！报告已生成。"
done

echo ""
echo "🎉🏆 所有数据集自动化处理完毕！可以前往各个文件夹查看 evaluation_report.txt。"