#!/bin/bash

TARGET_FONT_PATH="fonts/M8.ttf"
TRAIN_SPLIT_RATIO=0.9           # 保持0.9/0.1比例
VAL_SPLIT_RATIO=0.1             # 保持0.9/0.1比例
RANDOM_SEED=2025
BATCH_SIZE=10                    # 从12降到10，进一步减少过拟合风险
LEARNING_RATE=1e-3               # 从8e-4增加到1e-3，更激进的学习率
NUM_EPOCHS=600                   # 从400增加到600，进一步充分学习小数据集
DEVICE="cuda"

echo "🚀 开始针对小数据集优化的VQ-VAE训练（进阶版）..."
echo "📊 数据集大小: 749个字符"
echo "🎯 优化策略: 增加训练轮数 + 优化学习率 + 增强正则化"
echo ""

TARGET_FONT_NAME=$(basename "$TARGET_FONT_PATH" | sed -E 's/\.(ttf|otf)$//')

MODEL_SAVE_PATH="checkpoints/vqvae_${TARGET_FONT_NAME}.pth"

echo "📁 目标字体: $TARGET_FONT_NAME"
echo "💾 模型保存路径: $MODEL_SAVE_PATH"
echo "⚙️  训练参数:"
echo "   - 批次大小: $BATCH_SIZE"
echo "   - 学习率: $LEARNING_RATE"
echo "   - 训练轮数: $NUM_EPOCHS"
echo "   - 训练/验证比例: $TRAIN_SPLIT_RATIO/$VAL_SPLIT_RATIO"
echo ""

python train_vqvae.py \
    --split_ratios "$TRAIN_SPLIT_RATIO" "$VAL_SPLIT_RATIO" \
    --random_seed "$RANDOM_SEED" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --num_epochs "$NUM_EPOCHS" \
    --model_save_path "$MODEL_SAVE_PATH" \
    --device "$DEVICE"

echo ""
echo "✅ VQ-VAE训练完成！"
echo "📁 模型已保存到: $MODEL_SAVE_PATH"
