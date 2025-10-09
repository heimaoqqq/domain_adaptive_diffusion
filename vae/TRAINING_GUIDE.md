# KL-VAE 训练指南

## 功能改进

### 1. 增强的可视化
- **三行显示**：原图、重建图、差异图（5倍放大）
- **损失曲线**：自动保存训练进度图
- **损失信息**：图片标题显示当前损失值

### 2. 最佳模型管理
- **自动删除旧模型**：只保留最新的最佳模型
- **最佳模型可视化**：保存最佳效果的重建图像
- **模型命名**：`kl_vae_best.pt`

### 3. 早停机制
- **默认patience**: 20个epoch
- **防止过拟合**：验证损失不再改善时自动停止

## 快速开始

```bash
cd domain_adaptive_diffusion/vae
python train_kl_vae.py \
    --data_dir ../../dataset/organized_gait_dataset/Normal_line \
    --epochs 100 \
    --visualize_every 5 \
    --batch_size 16 \
    --lr 4.5e-6
```

## 参数说明

- `--visualize_every`: 可视化频率（默认5个epoch）
- `--patience`: 早停耐心值（默认20）
- `--batch_size`: 批次大小（建议16-32）
- `--kl_weight`: KL损失权重（默认1e-6）

## 输出文件

```
vae_checkpoints/
├── kl_vae_best.pt          # 最佳模型（自动更新）
├── best_samples.png        # 最佳模型重建效果
├── best_samples_loss.png   # 损失曲线
├── samples_epoch_5.png     # 定期可视化
└── samples_epoch_5_loss.png
```

## 判断训练效果

微多普勒数据的合理指标：
- 重建损失 < 0.01：优秀
- 重建损失 < 0.05：良好
- KL损失 ≈ 1e-4：正常

## 注意事项

1. **不要使用数据增强**（已禁用）
2. **观察频谱主体**是否清晰
3. **检查时间连续性**是否保持
4. **早停触发**说明模型已收敛
