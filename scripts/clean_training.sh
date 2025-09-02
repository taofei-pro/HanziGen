#!/bin/bash

# 训练产物清理脚本
# 在重新训练前执行，清理之前的训练产物

echo "🧹 开始清理训练产物..."

# 获取目标字体名称
TARGET_FONT_PATH="fonts/M8.ttf"
TARGET_FONT_NAME=$(basename "$TARGET_FONT_PATH" | sed -E 's/\.(ttf|otf)$//')

echo "📁 目标字体: $TARGET_FONT_NAME"

# 清理模型权重文件
echo "🗑️  清理模型权重..."
if [ -d "checkpoints" ]; then
    rm -f checkpoints/vqvae_*.pth
    rm -f checkpoints/ldm_*.pth
    echo "   ✅ 已清理 checkpoints/ 中的模型权重"
else
    echo "   ℹ️  checkpoints/ 目录不存在"
fi

# 清理训练日志
echo "🗑️  清理训练日志..."
if [ -d "runs" ]; then
    rm -rf runs/VQVAE/*
    rm -rf runs/LDM/*
    echo "   ✅ 已清理 runs/ 中的训练日志"
else
    echo "   ℹ️  runs/ 目录不存在"
fi

# 清理生成样本
echo "🗑️  清理生成样本..."
if [ -d "samples_${TARGET_FONT_NAME}" ]; then
    rm -rf samples_${TARGET_FONT_NAME}/*
    echo "   ✅ 已清理 samples_${TARGET_FONT_NAME}/ 中的生成样本"
else
    echo "   ℹ️  samples_${TARGET_FONT_NAME}/ 目录不存在"
fi

# 清理SVG输出（如果存在）
echo "🗑️  清理SVG输出..."
if [ -d "svgs_${TARGET_FONT_NAME}" ]; then
    rm -rf svgs_${TARGET_FONT_NAME}/*
    echo "   ✅ 已清理 svgs_${TARGET_FONT_NAME}/ 中的SVG文件"
else
    echo "   ℹ️  svgs_${TARGET_FONT_NAME}/ 目录不存在"
fi

# 清理字集划分（需要重新生成）
echo "🗑️  清理字集划分..."
if [ -d "charsets/splits/${TARGET_FONT_NAME}" ]; then
    rm -f charsets/splits/${TARGET_FONT_NAME}/train.txt
    rm -f charsets/splits/${TARGET_FONT_NAME}/val.txt
    echo "   ✅ 已清理字集划分文件"
else
    echo "   ℹ️  字集划分目录不存在"
fi

# 保留覆盖率分析结果（不需要重新生成）
echo "ℹ️  保留覆盖率分析结果（charsets/jf7000_coverage/ 和 charsets/unihan_coverage/）"

# 保留数据集（不需要重新生成）
echo "ℹ️  保留数据集（data/ 目录）"

echo ""
echo "🎉 清理完成！"
echo ""
echo "📋 已清理的内容："
echo "   - 模型权重文件"
echo "   - 训练日志"
echo "   - 生成样本"
echo "   - SVG输出"
echo "   - 字集划分"
echo ""
echo "📋 保留的内容："
echo "   - 覆盖率分析结果"
echo "   - 数据集图像"
echo ""
echo "🚀 现在可以开始重新训练了！"
echo "   执行顺序："
echo "   1. bash scripts/extract_charset.sh"
echo "   2. bash scripts/train_vqvae.sh"
echo "   3. bash scripts/train_ldm.sh"
