# 快速优化实施指南
## 立即提升训练效率和显存使用

---

## 🚀 快速开始

本指南提供**立即可实施**的优化方案，预计1-2小时内完成，显存降低50%，速度提升2倍。

---

## 优化1: 启用混合精度训练 (AMP)

### 实施步骤

#### 1. 修改 `models/vqvae/vqvae.py`

在 `_train_one_epoch` 方法中添加AMP支持：

```python
from torch.cuda.amp import autocast, GradScaler

class VQVAE(nn.Module):
    def __init__(self, ...):
        # ... 现有代码 ...
        self.scaler = GradScaler()  # 添加这行
    
    def _train_one_epoch(self, ...):
        # ... 现有代码 ...
        
        for batch_idx, batch in enumerate(train_loader):
            # 将现有的训练循环包装在autocast中
            with autocast():
                batch_losses = self._process_batch(
                    batch, is_training=True, optimizer=optimizer
                )
                total_loss = sum(batch_losses.values())
            
            # 使用scaler进行反向传播
            self.scaler.scale(total_loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
            optimizer.zero_grad()
            
            # ... 其余代码保持不变 ...
```

#### 2. 修改 `models/ldm/ldm.py`

同样添加AMP支持：

```python
from torch.cuda.amp import autocast, GradScaler

class LDM(nn.Module):
    def __init__(self, ...):
        # ... 现有代码 ...
        self.scaler = GradScaler()  # 添加这行
    
    def _train_one_epoch(self, ...):
        # ... 现有代码 ...
        
        for batch_idx, batch in enumerate(train_loader):
            with autocast():
                batch_loss = self._process_batch(
                    batch, is_training=True, optimizer=optimizer
                )
            
            self.scaler.scale(batch_loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
            optimizer.zero_grad()
            
            # ... 其余代码保持不变 ...
```

### 预期效果
- ✅ 显存降低: 30-50%
- ✅ 训练速度: 提升1.5-2倍
- ✅ 可以增加batch_size: 4 → 6-8

---

## 优化2: 启用梯度检查点

### 实施步骤

#### 1. 修改 `models/unet/unet.py`

在UNet的forward方法中使用checkpoint：

```python
from torch.utils.checkpoint import checkpoint

class UNet(nn.Module):
    def forward(self, x, t_emb, ref_emb):
        # ... 编码器部分保持不变 ...
        
        # 在解码器部分使用checkpoint
        for i, (up_block, skip_feat) in enumerate(zip(self.up_blocks, skip_features)):
            if skip_feat is not None:
                x = torch.cat([x, skip_feat], dim=1)
            
            # 对深层使用checkpoint
            if i >= len(self.up_blocks) - 2:  # 最后两层使用checkpoint
                x = checkpoint(up_block, x, t_emb)
            else:
                x = up_block(x, t_emb)
        
        # ... 其余代码保持不变 ...
```

### 预期效果
- ✅ 显存降低: 30-50%
- ⚠️ 训练速度: 可能降低10-20%（但显存节省更重要）

---

## 优化3: 优化DataLoader配置

### 实施步骤

#### 1. 修改 `datasets/loader.py`

```python
# 在Loader.from_dataset_config中
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=dataset_config.batch_size,
    shuffle=True,
    num_workers=min(4, dataset_config.num_workers),  # 增加到4（如果CPU允许）
    pin_memory=True if device.type == "cuda" else False,
    prefetch_factor=2 if num_workers > 0 else None,  # 增加到2
    persistent_workers=True if num_workers > 0 else False,  # 启用持久化
)
```

### 预期效果
- ✅ 数据加载速度: 提升20-30%
- ✅ GPU利用率: 提升10-20%

---

## 优化4: 减少IO操作频率

### 实施步骤

#### 1. 修改 `models/vqvae/vqvae.py`

```python
def _train_one_epoch(self, ...):
    # ... 现有代码 ...
    
    # 减少TensorBoard写入频率
    if epoch % 5 == 0:  # 每5个epoch写入一次（原来是每个epoch）
        for key, value in epoch_metrics.items():
            self.writer.add_scalar(f"train/{key}", value, epoch)
    
    # 减少验证频率
    if epoch % 5 == 0:  # 每5个epoch验证一次
        val_metrics = self._validate(val_loader, epoch)
```

#### 2. 修改 `models/ldm/ldm.py`

```python
def _train_one_epoch(self, ...):
    # ... 现有代码 ...
    
    # 减少样本保存频率
    if epoch % 10 == 0:  # 每10个epoch保存一次（原来是每5个epoch）
        self._save_samples(...)
    
    # 减少LPIPS评估频率
    if epoch % 20 == 0:  # 每20个epoch评估一次（原来是每10个epoch）
        self._evaluate_lpips(...)
```

### 预期效果
- ✅ IO阻塞时间: 减少50-70%
- ✅ 训练速度: 提升5-10%

---

## 优化5: 智能显存管理

### 实施步骤

#### 1. 修改 `models/vqvae/vqvae.py`

```python
def _train_one_epoch(self, ...):
    # ... 现有代码 ...
    
    for batch_idx, batch in enumerate(train_loader):
        # ... 训练代码 ...
        
        # 每100个batch清理一次显存
        if batch_idx % 100 == 0:
            torch.cuda.empty_cache()
            import gc
            gc.collect()
        
        # ... 其余代码 ...
```

#### 2. 修改 `models/ldm/ldm.py`

同样添加显存清理：

```python
def _train_one_epoch(self, ...):
    # ... 现有代码 ...
    
    for batch_idx, batch in enumerate(train_loader):
        # ... 训练代码 ...
        
        # 每100个batch清理一次显存
        if batch_idx % 100 == 0:
            torch.cuda.empty_cache()
            import gc
            gc.collect()
```

### 预期效果
- ✅ 显存碎片: 减少
- ✅ OOM风险: 降低

---

## 优化6: 增加Batch Size（在启用AMP后）

### 实施步骤

#### 1. 修改 `configs/vqvae_config.py`

```python
# 在启用AMP后，可以安全地增加batch_size
batch_size: int = 6  # 从4增加到6（如果显存允许，可以尝试8）
```

#### 2. 修改 `configs/ldm_config.py`

```python
batch_size: int = 6  # 从4增加到6（如果显存允许，可以尝试8）
```

### 预期效果
- ✅ 训练速度: 进一步提升（因为batch更大）
- ✅ 梯度估计: 更稳定

---

## 📊 优化效果总结

### 实施前
- 显存使用: ~2-3GB
- 训练速度: VQ-GAN ~3-5秒/epoch, LDM ~5-8秒/epoch
- Batch Size: 4

### 实施后（预期）
- 显存使用: ~1-1.5GB (降低50%)
- 训练速度: VQ-GAN ~1.5-2.5秒/epoch, LDM ~2.5-4秒/epoch (提升2倍)
- Batch Size: 6-8 (增加50-100%)

---

## ⚠️ 注意事项

1. **逐步实施**: 不要一次性应用所有优化，先实施AMP，验证效果后再继续
2. **保留备份**: 修改前备份原始文件
3. **监控显存**: 使用 `nvidia-smi` 监控显存使用情况
4. **测试验证**: 每个优化后都要运行小规模测试

---

## 🔧 故障排除

### 问题1: AMP导致数值不稳定
**解决方案**: 
- 检查是否有NaN值
- 可能需要调整GradScaler的初始scale
- 某些操作可能需要保持FP32

### 问题2: 梯度检查点导致速度过慢
**解决方案**:
- 只对深层使用checkpoint
- 或者完全禁用，只使用AMP

### 问题3: DataLoader导致CPU过载
**解决方案**:
- 减少num_workers到2
- 或者保持当前配置

---

## 📝 实施检查清单

- [ ] 备份原始文件
- [ ] 实施AMP优化
- [ ] 测试VQ-GAN训练（小规模）
- [ ] 测试LDM训练（小规模）
- [ ] 监控显存使用
- [ ] 验证训练速度提升
- [ ] 实施梯度检查点（可选）
- [ ] 优化DataLoader配置
- [ ] 减少IO操作频率
- [ ] 增加batch_size（如果显存允许）
- [ ] 更新文档

---

## 🎯 下一步

完成这些快速优化后，可以继续实施：
1. 架构优化（增加Codebook容量等）
2. 采样调度器改进
3. 损失函数权重调优

详见 `PROJECT_REVIEW_2025.md` 中的详细建议。

---

**文档版本**: v1.0
**创建日期**: 2025年1月
**最后更新**: 2025年1月
