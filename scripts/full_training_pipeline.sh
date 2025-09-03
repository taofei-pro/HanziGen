#!/bin/bash

# 完整训练流程脚本
# 一键执行：清理 → 字集划分 → VQ-VAE训练 → LDM训练 → 评估

# 记录训练开始时间
TRAINING_START_TIME=$(date +%s)
echo "🚀 开始完整训练流程..."
echo "⏰ 开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================="

# 获取目标字体名称
TARGET_FONT_PATH="fonts/M8.ttf"
TARGET_FONT_NAME=$(basename "$TARGET_FONT_PATH" | sed -E 's/\.(ttf|otf)$//')

echo "📁 目标字体: $TARGET_FONT_NAME"
echo ""

# 第一步：清理训练产物
echo "🧹 第一步：清理训练产物..."
CLEAN_START_TIME=$(date +%s)
bash scripts/clean_training.sh
CLEAN_END_TIME=$(date +%s)
CLEAN_DURATION=$((CLEAN_END_TIME - CLEAN_START_TIME))

# 检查清理是否成功
if [ $? -ne 0 ]; then
    echo "❌ 清理失败，停止执行"
    exit 1
fi
echo "✅ 清理完成 (耗时: ${CLEAN_DURATION}秒)"
echo ""

# 第二步：重新划分字集
echo "📊 第二步：重新划分字集..."
echo "   训练/验证集比例: 0.9/0.1"
CHARSET_START_TIME=$(date +%s)
bash scripts/extract_charset.sh
CHARSET_END_TIME=$(date +%s)
CHARSET_DURATION=$((CHARSET_END_TIME - CHARSET_START_TIME))

# 检查字集划分是否成功
if [ $? -ne 0 ]; then
    echo "❌ 字集划分失败，停止执行"
    exit 1
fi
echo "✅ 字集划分完成 (耗时: ${CHARSET_DURATION}秒)"
echo ""

# 显示字集统计
echo "📈 字集统计："
if [ -f "charsets/splits/${TARGET_FONT_NAME}/train.txt" ] && [ -f "charsets/splits/${TARGET_FONT_NAME}/val.txt" ]; then
    TRAIN_COUNT=$(wc -l < "charsets/splits/${TARGET_FONT_NAME}/train.txt")
    VAL_COUNT=$(wc -l < "charsets/splits/${TARGET_FONT_NAME}/val.txt")
    echo "   训练集: $TRAIN_COUNT 个汉字"
    echo "   验证集: $VAL_COUNT 个汉字"
    echo "   总计: $((TRAIN_COUNT + VAL_COUNT)) 个汉字"
else
    echo "   ⚠️  无法读取字集文件"
fi
echo ""

# 第三步：训练VQ-VAE
echo "🏋️  第三步：训练VQ-VAE模型..."
echo "   参数: batch_size=16, epochs=200, lr=6e-4"
echo "   预计时间: 4-5小时"
echo ""

# 自动开始VQ-VAE训练
echo "🚀 自动开始VQ-VAE训练..."
echo "⏰ VQ-VAE开始时间: $(date '+%H:%M:%S')"
VQVAE_START_TIME=$(date +%s)
bash scripts/train_vqvae.sh
VQVAE_END_TIME=$(date +%s)
VQVAE_DURATION=$((VQVAE_END_TIME - VQVAE_START_TIME))

# 检查VQ-VAE训练是否成功
if [ $? -ne 0 ]; then
    echo "❌ VQ-VAE训练失败，停止执行"
    exit 1
fi
echo "✅ VQ-VAE训练完成 (耗时: ${VQVAE_DURATION}秒)"
echo "⏰ VQ-VAE完成时间: $(date '+%H:%M:%S')"
echo ""

# 检查模型文件是否存在
if [ ! -f "checkpoints/vqvae_${TARGET_FONT_NAME}.pth" ]; then
    echo "❌ VQ-VAE模型文件未找到，停止执行"
    exit 1
fi
echo "📁 VQ-VAE模型已保存: checkpoints/vqvae_${TARGET_FONT_NAME}.pth"
echo ""

# 第四步：训练LDM
echo "🏋️  第四步：训练LDM模型..."
echo "   参数: batch_size=32, epochs=500, lr=2e-4, sample_steps=120"
echo "   预计时间: 7-9小时"
echo ""

# 自动开始LDM训练
echo "🚀 自动开始LDM训练..."
echo "⏰ LDM开始时间: $(date '+%H:%M:%S')"
LDM_START_TIME=$(date +%s)
bash scripts/train_ldm.sh
LDM_END_TIME=$(date +%s)
LDM_DURATION=$((LDM_END_TIME - LDM_START_TIME))

# 检查LDM训练是否成功
if [ $? -ne 0 ]; then
    echo "❌ LDM训练失败，停止执行"
    exit 1
fi
echo "✅ LDM训练完成 (耗时: ${LDM_DURATION}秒)"
echo "⏰ LDM完成时间: $(date '+%H:%M:%S')"
echo ""

# 检查模型文件是否存在
if [ ! -f "checkpoints/ldm_${TARGET_FONT_NAME}.pth" ]; then
    echo "❌ LDM模型文件未找到，停止执行"
    exit 1
fi
echo "📁 LDM模型已保存: checkpoints/ldm_${TARGET_FONT_NAME}.pth"
echo ""

# 第五步：评估模型性能
echo "📊 第五步：评估模型性能..."
echo "   计算PSNR、SSIM、LPIPS、FID指标"
echo ""

# 自动开始性能评估
echo "🚀 自动开始性能评估..."
echo "⏰ 评估开始时间: $(date '+%H:%M:%S')"
EVAL_START_TIME=$(date +%s)
bash scripts/compute_metrics.sh
EVAL_END_TIME=$(date +%s)
EVAL_DURATION=$((EVAL_END_TIME - EVAL_START_TIME))

# 检查评估是否成功
if [ $? -ne 0 ]; then
    echo "❌ 性能评估失败"
    exit 1
fi
echo "✅ 性能评估完成 (耗时: ${EVAL_DURATION}秒)"
echo "⏰ 评估完成时间: $(date '+%H:%M:%S')"
echo ""

# 显示评估结果
echo "📈 评估结果："
if [ -f "samples_${TARGET_FONT_NAME}/inference/gen" ]; then
    GEN_COUNT=$(ls "samples_${TARGET_FONT_NAME}/inference/gen"/*.png 2>/dev/null | wc -l)
    echo "   生成样本数量: $GEN_COUNT 个"
else
    echo "   ⚠️  无法读取生成样本"
fi
echo ""

# 计算总训练时长
TRAINING_END_TIME=$(date +%s)
TOTAL_DURATION=$((TRAINING_END_TIME - TRAINING_START_TIME))

# 转换时长为可读格式
format_duration() {
    local seconds=$1
    local hours=$((seconds / 3600))
    local minutes=$(((seconds % 3600) / 60))
    local secs=$((seconds % 60))
    
    if [ $hours -gt 0 ]; then
        echo "${hours}小时${minutes}分钟${secs}秒"
    elif [ $minutes -gt 0 ]; then
        echo "${minutes}分钟${secs}秒"
    else
        echo "${secs}秒"
    fi
}

# 完成提示
echo "🎉 完整训练流程执行完成！"
echo "=================================="
echo ""
echo "⏰ 训练时间统计："
echo "   - 清理阶段: $(format_duration $CLEAN_DURATION)"
echo "   - 字集划分: $(format_duration $CHARSET_DURATION)"
echo "   - VQ-VAE训练: $(format_duration $VQVAE_DURATION)"
echo "   - LDM训练: $(format_duration $LDM_DURATION)"
echo "   - 性能评估: $(format_duration $EVAL_DURATION)"
echo "   - 总训练时长: $(format_duration $TOTAL_DURATION)"
echo ""
echo "📋 训练产物位置："
echo "   - 模型权重: checkpoints/"
echo "   - 训练日志: runs/"
echo "   - 生成样本: samples_${TARGET_FONT_NAME}/"
echo "   - 评估结果: 见上方输出"
echo ""
echo "🚀 下一步可选操作："
echo "   1. 生成缺失汉字: bash scripts/inference.sh"
echo "   2. 转换为SVG: bash scripts/convert_to_svg.sh"
echo "   3. 查看训练日志: tensorboard --logdir=runs/"
echo ""
echo "📝 建议："
echo "   - 检查生成的样本质量"
echo "   - 对比评估指标与目标分数"
echo "   - 根据结果决定是否需要进一步调优"
echo ""
echo "✨ 训练完成！总耗时: $(format_duration $TOTAL_DURATION)"
