#!/bin/bash

TARGET_FONT_PATH="fonts/M8.ttf"
TRAIN_SPLIT_RATIO=0.9           # 保持0.9/0.1比例
VAL_SPLIT_RATIO=0.1             # 保持0.9/0.1比例
RANDOM_SEED=2025
BATCH_SIZE=24                    # 从32降到24，减少过拟合风险
LEARNING_RATE=3e-4               # 从2e-4增加到3e-4，加快收敛
NUM_EPOCHS=800                   # 从500增加到800，充分学习小数据集
SAMPLE_STEPS=150                 # 从120增加到150，提升生成质量
IMG_SAVE_INTERVAL=5              # 保持5
LPIPS_EVAL_INTERVAL=10           # 保持10
EVAL_BATCH_SIZE=4                # 保持4
DEVICE="cuda"

echo "🚀 开始针对小数据集优化的LDM训练..."
echo "📊 数据集大小: 749个字符"
echo "🎯 优化策略: 增加训练轮数 + 优化学习率 + 增强正则化"
echo ""

TARGET_FONT_NAME=$(basename "$TARGET_FONT_PATH" | sed -E 's/\.(ttf|otf)$//')

PRETRAINED_VQVAE_PATH="checkpoints/vqvae_${TARGET_FONT_NAME}.pth"
MODEL_SAVE_PATH="checkpoints/ldm_${TARGET_FONT_NAME}.pth"
SAMPLE_ROOT="samples_${TARGET_FONT_NAME}/"

echo "📁 目标字体: $TARGET_FONT_NAME"
echo "💾 VQ-VAE模型: $PRETRAINED_VQVAE_PATH"
echo "💾 LDM模型保存路径: $MODEL_SAVE_PATH"
echo "⚙️  训练参数:"
echo "   - 批次大小: $BATCH_SIZE"
echo "   - 学习率: $LEARNING_RATE"
echo "   - 训练轮数: $NUM_EPOCHS"
echo "   - 采样步数: $SAMPLE_STEPS"
echo "   - 训练/验证比例: $TRAIN_SPLIT_RATIO/$VAL_SPLIT_RATIO"
echo ""

# 检查VQ-VAE模型是否存在
if [ ! -f "$PRETRAINED_VQVAE_PATH" ]; then
    echo "❌ 错误: VQ-VAE模型文件不存在: $PRETRAINED_VQVAE_PATH"
    echo "请先运行 VQ-VAE 训练脚本"
    exit 1
fi

echo "✅ VQ-VAE模型检查通过，开始LDM训练..."
echo ""

python train_ldm.py \
    --split_ratios "$TRAIN_SPLIT_RATIO" "$VAL_SPLIT_RATIO" \
    --random_seed "$RANDOM_SEED" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --num_epochs "$NUM_EPOCHS" \
    --pretrained_vqvae_path "$PRETRAINED_VQVAE_PATH" \
    --model_save_path "$MODEL_SAVE_PATH" \
    --sample_root "$SAMPLE_ROOT" \
    --sample_steps "$SAMPLE_STEPS" \
    --img_save_interval "$IMG_SAVE_INTERVAL" \
    --lpips_eval_interval "$LPIPS_EVAL_INTERVAL" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --device "$DEVICE"

echo ""
echo "✅ LDM训练完成！"
echo "📁 模型已保存到: $MODEL_SAVE_PATH"
echo "🖼️  生成样本保存在: $SAMPLE_ROOT"
