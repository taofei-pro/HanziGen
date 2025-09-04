#!/bin/bash

# 完整训练流程脚本 - 针对小数据集（749字符）优化版
# 策略：增加训练轮数 + 优化学习率 + 增强正则化

echo "🚀 开始针对小数据集的完整训练流程..."
echo "📊 数据集大小: 749个字符"
echo "🎯 优化策略: 增加训练轮数 + 优化学习率 + 增强正则化"
echo ""

# 设置参数
TARGET_FONT_PATH="fonts/M8.ttf"
TARGET_FONT_NAME=$(basename "$TARGET_FONT_PATH" | sed -E 's/\.(ttf|otf)$//')

echo "📁 目标字体: $TARGET_FONT_NAME"
echo ""

# 第一步：清理历史训练数据
echo "🧹 第一步：清理历史训练数据..."
echo "   清理模型权重、训练日志、生成样本等，防止干扰"
echo ""

bash scripts/clean_training.sh

if [ $? -ne 0 ]; then
    echo "❌ 清理失败，停止执行"
    exit 1
fi
echo "✅ 历史训练数据清理完成"
echo ""

# 第二步：字体分析
echo "🔍 第二步：分析字体覆盖率..."
echo "   分析目标字体对jf7000和Unihan字符集的覆盖情况"
echo ""

bash scripts/analyze_font.sh

if [ $? -ne 0 ]; then
    echo "❌ 字体分析失败，停止执行"
    exit 1
fi
echo "✅ 字体分析完成"
echo ""

# 第三步：准备数据集
echo "🖨️  第三步：准备字形图像数据集..."
echo "   生成目标字体和参考字体的字形图像"
echo ""

bash scripts/prepare_dataset.sh

if [ $? -ne 0 ]; then
    echo "❌ 数据集准备失败，停止执行"
    exit 1
fi
echo "✅ 数据集准备完成"
echo ""

# 第四步：提取训练/验证字符集
echo "📝 第四步：提取训练/验证字符集..."
echo "   按9:1比例划分训练集和验证集"
echo ""

bash scripts/extract_charset.sh

if [ $? -ne 0 ]; then
    echo "❌ 字符集提取失败，停止执行"
    exit 1
fi
echo "✅ 字符集提取完成"
echo ""

# 第五步：训练VQ-VAE（小数据集优化版）
echo "🏋️  第五步：训练VQ-VAE模型（小数据集优化版）..."
echo "   参数: batch_size=12, epochs=400, lr=8e-4"
echo "   预计时间: 6-8小时"
echo "   优化策略: 增加训练轮数 + 优化学习率 + 增强正则化"
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

# 第六步：训练LDM（小数据集优化版）
echo "🏋️  第六步：训练LDM模型（小数据集优化版）..."
echo "   参数: batch_size=24, epochs=800, lr=3e-4, sample_steps=150"
echo "   预计时间: 12-16小时"
echo "   优化策略: 增加训练轮数 + 优化学习率 + 增强正则化"
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

# 第七步：评估模型性能
echo "📊 第七步：评估模型性能..."
echo "   计算PSNR、SSIM、LPIPS、FID指标"
echo ""

# 自动开始性能评估
echo "🚀 自动开始性能评估..."
echo "⏰ 评估开始时间: $(date '+%H:%M:%S')"
EVAL_START_TIME=$(date +%s)

# 使用测试字符集进行评估
CHARSET_PATH="charsets/test/qianziwen.txt"
echo "📝 使用测试字符集: $CHARSET_PATH"

python inference.py \
    --target_font_path "$TARGET_FONT_PATH" \
    --reference_fonts_dir "fonts/jigmo/" \
    --charset_path "$CHARSET_PATH" \
    --pretrained_ldm_path "checkpoints/ldm_${TARGET_FONT_NAME}.pth" \
    --batch_size 16 \
    --sample_root "samples_${TARGET_FONT_NAME}_eval/" \
    --sample_steps 150 \
    --img_size 512 512 \
    --device "cuda"

if [ $? -ne 0 ]; then
    echo "❌ 模型评估失败"
else
    echo "✅ 模型评估完成"
fi

EVAL_END_TIME=$(date +%s)
EVAL_DURATION=$((EVAL_END_TIME - EVAL_START_TIME))
echo "⏰ 评估完成时间: $(date '+%H:%M:%S')"
echo ""

# 第八步：计算评估指标
echo "📈 第八步：计算评估指标..."
echo "   计算PSNR、SSIM、LPIPS、FID等指标"
echo ""

# 自动开始指标计算
echo "🚀 自动开始指标计算..."
echo "⏰ 指标计算开始时间: $(date '+%H:%M:%S')"
METRICS_START_TIME=$(date +%s)

# 设置评估参数
GENERATED_IMG_DIR="samples_${TARGET_FONT_NAME}_eval/gen"
GROUND_TRUTH_IMG_DIR="samples_${TARGET_FONT_NAME}_eval/gt"
EVAL_BATCH_SIZE=4

echo "📝 评估参数:"
echo "   - 生成图像目录: $GENERATED_IMG_DIR"
echo "   - 真实图像目录: $GROUND_TRUTH_IMG_DIR"
echo "   - 评估批次大小: $EVAL_BATCH_SIZE"
echo ""

# 检查目录是否存在
if [ ! -d "$GENERATED_IMG_DIR" ] || [ ! -d "$GROUND_TRUTH_IMG_DIR" ]; then
    echo "⚠️  警告: 评估目录不存在，跳过指标计算"
    echo "   生成图像目录: $GENERATED_IMG_DIR"
    echo "   真实图像目录: $GROUND_TRUTH_IMG_DIR"
    echo "   请先运行推理步骤生成评估样本"
    echo ""
else
    # 计算评估指标
    echo "🧮 开始计算评估指标..."
    python compute_metrics.py \
        --generated_img_dir "$GENERATED_IMG_DIR" \
        --ground_truth_img_dir "$GROUND_TRUTH_IMG_DIR" \
        --eval_batch_size "$EVAL_BATCH_SIZE" \
        --device "$DEVICE"
    
    if [ $? -ne 0 ]; then
        echo "❌ 指标计算失败"
    else
        echo "✅ 指标计算完成"
    fi
fi

METRICS_END_TIME=$(date +%s)
METRICS_DURATION=$((METRICS_END_TIME - METRICS_START_TIME))
echo "⏰ 指标计算完成时间: $(date '+%H:%M:%S')"
echo "⏰ 指标计算耗时: ${METRICS_DURATION}秒"
echo ""

# 总结
TOTAL_DURATION=$((VQVAE_DURATION + LDM_DURATION + EVAL_DURATION + METRICS_DURATION))
echo "🎉 完整训练流程完成！"
echo ""
echo "📊 训练总结:"
echo "   - VQ-VAE训练: ${VQVAE_DURATION}秒"
echo "   - LDM训练: ${LDM_DURATION}秒"
echo "   - 模型评估: ${EVAL_DURATION}秒"
echo "   - 指标计算: ${METRICS_DURATION}秒"
echo "   - 总耗时: ${TOTAL_DURATION}秒"
echo ""
echo "📁 输出文件:"
echo "   - VQ-VAE模型: checkpoints/vqvae_${TARGET_FONT_NAME}.pth"
echo "   - LDM模型: checkpoints/ldm_${TARGET_FONT_NAME}.pth"
echo "   - 训练样本: samples_${TARGET_FONT_NAME}/"
echo "   - 评估样本: samples_${TARGET_FONT_NAME}_eval/"
echo ""
echo "🎯 针对小数据集的优化策略:"
echo "   ✅ 增加训练轮数: VQ-VAE(400轮) + LDM(800轮)"
echo "   ✅ 优化学习率: VQ-VAE(8e-4) + LDM(3e-4)"
echo "   ✅ 增强正则化: 降低批次大小，减少过拟合风险"
echo "   ✅ 数据增强: 充分利用有限的749个字符"
echo "   ✅ 清理历史数据: 防止干扰，确保训练纯净"
echo ""
echo "📈 评估指标说明:"
echo "   - PSNR (峰值信噪比): 越高越好，目标 > 20"
echo "   - SSIM (结构相似性): 越高越好，目标 > 0.95"
echo "   - LPIPS (感知相似性): 越低越好，目标 < 0.05"
echo "   - FID (Fréchet Inception距离): 越低越好，目标 < 5"
echo ""
echo "💡 建议:"
echo "   - 如果效果仍不理想，可以考虑进一步增加训练轮数"
echo "   - 可以尝试调整学习率调度策略"
echo "   - 考虑使用预训练模型进行迁移学习"
echo "   - 对于小数据集，增加训练轮数比增加模型复杂度更有效"
echo "   - 每次重新训练前都要清理历史数据，避免干扰"
echo ""
echo "⏰ 完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
