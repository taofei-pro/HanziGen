# Z1字体训练指南

## 🎯 快速开始

### 方式1：一键完整训练（推荐）

```bash
cd /home/zihun/workspace/Font/HanziGen
conda activate hanzigen
bash scripts/full_training_pipeline.sh
```

**预计时间：** 40-50分钟  
**包含步骤：** 数据准备 + VQ-GAN训练 + LDM训练 + 评估

---

### 方式2：分步执行（更好控制）

```bash
cd /home/zihun/workspace/Font/HanziGen
conda activate hanzigen

# 步骤1：分析字体覆盖率（~1分钟）
bash scripts/analyze_font.sh

# 步骤2：准备字形图像数据集（~2-3分钟）
bash scripts/prepare_dataset.sh

# 步骤3：提取训练/验证字符集（~1分钟）
bash scripts/extract_charset.sh

# 步骤4：训练VQ-GAN模型（~12-15分钟）
bash scripts/train_vqgan.sh

# 步骤5：训练LDM模型（~20-30分钟）
bash scripts/train_ldm.sh

# 步骤6：计算评估指标（~2-3分钟）
bash scripts/compute_metrics.sh
```

---

## 📊 训练配置说明

### VQ-GAN配置
- **批次大小：** 4
- **学习率：** 8e-4
- **训练轮数：** 400
- **模型参数：** 17.36M
- **判别器学习率：** 2e-4
- **感知损失权重：** 0.4
- **对抗损失权重：** 0.1

### LDM配置
- **批次大小：** 16
- **学习率：** 4e-4
- **训练轮数：** 600
- **采样步数：** 200
- **使用Stable Diffusion：** 是

---

## 📁 输出文件位置

训练完成后，你会得到以下文件：

```
checkpoints/
├── vqgan_Z1.pth          # VQ-GAN模型（~66MB）
└── ldm_Z1.pth            # LDM模型（~4.6GB）

samples_Z1/
├── train/                # 训练过程样本
├── val/                  # 验证过程样本
└── eval_outputs/         # 最终评估样本

runs/
├── VQGAN/                # VQ-GAN训练日志（TensorBoard）
└── LDM/                  # LDM训练日志（TensorBoard）

data/glyph_images/Z1/     # 字形图像数据集
```

---

## 🔍 监控训练进度

### 方式1：查看终端输出
训练过程中会实时显示：
- 进度条
- 损失值（Loss）
- 评估指标（PSNR, SSIM, LPIPS, FID）

### 方式2：使用TensorBoard（可选）

```bash
# 在另一个终端窗口运行
cd /home/zihun/workspace/Font/HanziGen
conda activate hanzigen

# 查看VQ-GAN训练
tensorboard --logdir runs/VQGAN --port 6006

# 查看LDM训练
tensorboard --logdir runs/LDM --port 6007
```

然后在浏览器打开：
- VQ-GAN: http://localhost:6006
- LDM: http://localhost:6007

---

## ⚠️ 注意事项

### 1. 训练速度预期
- **VQ-GAN第1批次：** ~1-2秒（初始化慢）
- **VQ-GAN后续批次：** ~30ms/批次
- **每轮VQ-GAN：** ~2秒
- **400轮VQ-GAN：** ~12-15分钟

如果速度明显偏慢，请检查：
- GPU是否正常工作：`nvidia-smi`
- 是否在正确的conda环境：`conda activate hanzigen`

### 2. 显存使用
- **VQ-GAN训练：** ~2-3GB
- **LDM训练：** ~4-6GB
- **RTX 5090D（24GB）：** 完全足够

### 3. 中断恢复
如果训练中断：
- VQ-GAN会自动保存最佳模型到 `checkpoints/vqgan_Z1.pth`
- LDM会自动保存最佳模型到 `checkpoints/ldm_Z1.pth`
- 可以从上次中断的步骤继续执行

---

## 📈 评估指标说明

训练完成后会显示以下指标：

| 指标 | 说明 | 目标值 |
|------|------|--------|
| **PSNR** | 峰值信噪比，越高越好 | > 12 |
| **SSIM** | 结构相似性，越高越好 | > 0.85 |
| **LPIPS** | 感知相似性，越低越好 | < 0.12 |
| **FID** | Fréchet距离，越低越好 | < 40 |

---

## 🐛 常见问题

### Q1: 提示"ModuleNotFoundError"
```bash
# 确保激活了正确的环境
conda activate hanzigen
```

### Q2: 提示"CUDA out of memory"
```bash
# 降低批次大小（编辑配置文件）
# configs/vqvae_config.py: batch_size = 2
# configs/ldm_config.py: batch_size = 8
```

### Q3: 训练速度很慢
```bash
# 检查GPU状态
nvidia-smi

# 确保没有其他程序占用GPU
```

### Q4: 想重新开始训练
```bash
# 清理Z1的训练数据
rm -rf checkpoints/vqgan_Z1.pth checkpoints/ldm_Z1.pth
rm -rf samples_Z1/ runs/VQGAN/* runs/LDM/*
rm -rf data/glyph_images/Z1/

# 然后重新开始训练
bash scripts/full_training_pipeline.sh
```

---

## 🎉 训练完成后

### 查看生成效果
```bash
# 查看训练样本
ls samples_Z1/train/
ls samples_Z1/val/

# 查看评估样本
ls samples_Z1/eval_outputs/
```

### 查看评估指标
训练结束时会自动显示最终指标，或手动运行：
```bash
bash scripts/compute_metrics.sh
```

### 生成新字符
```bash
# TODO: 添加推理脚本
python inference.py --font Z1 --char "你好世界"
```

---

## 📞 需要帮助？

如果遇到问题，可以：
1. 查看训练日志：`runs/VQGAN/` 或 `runs/LDM/`
2. 检查错误信息
3. 告诉我具体的错误信息，我会帮你解决

---

**祝训练顺利！🚀**

