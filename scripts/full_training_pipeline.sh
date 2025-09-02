#!/bin/bash

# 完整训练流程脚本
# 一键执行：清理 → 字集划分 → VQ-VAE训练 → LDM训练 → 评估

echo "🚀 开始完整训练流程..."
echo "=================================="

# 获取目标字体名称
TARGET_FONT_PATH="fonts/M8.ttf"
TARGET_FONT_NAME=$(basename "$TARGET_FONT_PATH" | sed -E 's/\.(ttf|otf)$//')

echo "📁 目标字体: $TARGET_FONT_NAME"
echo ""

# 第一步：清理训练产物
echo "🧹 第一步：清理训练产物..."
bash scripts/clean_training.sh
echo ""

# 检查清理是否成功
if [ $? -ne 0 ]; then
    echo "❌ 清理失败，停止执行"
    exit 1
fi
echo "✅ 清理完成"
echo ""

# 第二步：重新划分字集
echo "📊 第二步：重新划分字集..."
echo "   训练/验证集比例: 0.9/0.1"
bash scripts/extract_charset.sh

# 检查字集划分是否成功
if [ $? -ne 0 ]; then
    echo "❌ 字集划分失败，停止执行"
    exit 1
fi
echo "✅ 字集划分完成"
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
echo "   参数: batch_size=12, epochs=120, lr=8e-4"
echo "   预计时间: 2-4小时"
echo ""

# 询问是否继续
read -p "是否开始训练VQ-VAE？(y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "⏸️  训练暂停，您可以稍后手动执行"
    exit 0
fi

echo "🚀 开始VQ-VAE训练..."
bash scripts/train_vqvae.sh

# 检查VQ-VAE训练是否成功
if [ $? -ne 0 ]; then
    echo "❌ VQ-VAE训练失败，停止执行"
    exit 1
fi
echo "✅ VQ-VAE训练完成"
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
echo "   参数: batch_size=24, epochs=300, lr=3e-4, sample_steps=80"
echo "   预计时间: 4-8小时"
echo ""

# 询问是否继续
read -p "是否开始训练LDM？(y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "⏸️  训练暂停，您可以稍后手动执行"
    echo "   手动执行命令: bash scripts/train_ldm.sh"
    exit 0
fi

echo "🚀 开始LDM训练..."
bash scripts/train_ldm.sh

# 检查LDM训练是否成功
if [ $? -ne 0 ]; then
    echo "❌ LDM训练失败，停止执行"
    exit 1
fi
echo "✅ LDM训练完成"
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

# 询问是否继续
read -p "是否开始评估？(y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "⏸️  评估暂停，您可以稍后手动执行"
    echo "   手动执行命令: bash scripts/compute_metrics.sh"
    exit 0
fi

echo "🚀 开始性能评估..."
bash scripts/compute_metrics.sh

# 检查评估是否成功
if [ $? -ne 0 ]; then
    echo "❌ 性能评估失败"
    exit 1
fi
echo "✅ 性能评估完成"
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

# 完成提示
echo "🎉 完整训练流程执行完成！"
echo "=================================="
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
echo "✨ 训练完成！"
