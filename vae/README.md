# VAE 训练说明

## 快速开始

```bash
cd domain_adaptive_diffusion/vae
python train_kl_vae.py \
    --data_dir ../../dataset/organized_gait_dataset/Normal_line \
    --epochs 100 \
    --batch_size 16
```

## 可视化特点

- **每个epoch自动生成对比图**
- **只显示原图 vs 重建图**（无损失曲线，无差异图）
- **文件命名**：`samples_epoch_N.png`
- **最佳模型**：`kl_vae_best.pt`（自动删除旧版本）

## 输出示例

每个epoch会生成类似这样的对比图：
```
第1行: 原始微多普勒图像（8个样本）
第2行: VAE重建结果
标题: Epoch N - Rec Loss: 0.XXXX
```

## 判断标准

- 重建损失 < 0.01：优秀
- 重建损失 < 0.05：良好
- 视觉检查：频谱主体清晰，时间连续性保持
