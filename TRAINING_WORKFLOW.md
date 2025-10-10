# 完整训练流程

## 步骤顺序

### 1. 训练VAE（需要原始图像）

**选项A：在所有步态类型上训练（推荐）**

单GPU训练：
```bash
cd domain_adaptive_diffusion/vae
python train_kl_vae.py \
    --data_dir ../../dataset/organized_gait_dataset \
    --epochs 100 \
    --val_split 0.1 \
    --batch_size 4
```

双GPU训练（Kaggle DDP）：
```bash
cd domain_adaptive_diffusion/vae
python train_kl_vae_ddp.py \
    --data_dir /kaggle/input/organized-gait-dataset \
    --epochs 100 \
    --val_split 0.1 \
    --batch_size 9 \  # 每个GPU的batch_size
    --world_size 2    # 使用2个GPU
```

**选项B：只在单个步态类型上训练**
```bash
cd domain_adaptive_diffusion/vae
python train_kl_vae.py \
    --data_dir ../../dataset/organized_gait_dataset/Normal_line \
    --epochs 100 \
    --val_split 0.1
```

**选项C：使用渐进式感知损失（推荐尝试）**
```bash
cd domain_adaptive_diffusion/vae
python train_kl_vae.py \
    --data_dir ../../dataset/organized_gait_dataset/Normal_line \
    --epochs 100 \
    --val_split 0.1 \
    --use_perceptual \
    --perceptual_weight 0.05 \
    --perceptual_start_epoch 10
```

注意：
- 推荐使用选项A，在所有数据上训练VAE
- VAE不区分源域/目标域，目标是学习通用的图像编码
- 自动划分90%训练集，10%验证集
- 使用固定随机种子确保可重复性
- 选项C的渐进式感知损失：前10个epoch只用MSE，之后加入感知损失细化细节

### 2. 准备DDPM数据（使用训练好的VAE）

```bash
cd ../scripts
python prepare_microdoppler_data.py \
    --dataset_root ../../dataset/organized_gait_dataset \
    --source_domain Normal_line \
    --target_domain Normal_free \
    --vae_checkpoint ../vae/checkpoints/kl_vae_best.pt
```
这一步会：
- 划分源域/目标域
- 划分训练/验证集（8:2）
- 选择目标域few-shot样本（5张/用户）
- 使用VAE编码成latent
- 保存到 ../data/processed/

### 3. 训练域自适应DDPM（使用latent）
```bash
python train.py \
    --config ../configs/small_dataset.yaml \
    --data_dir ../data/processed
```

## 数据流向

```
原始图像（所有步态类型）
    │
    v
VAE训练（90/10划分）
    │
    v
VAE模型（kl_vae_best.pt）
    │
    v
prepare_microdoppler_data.py
    │
    ├─> 源域数据编码（80/20划分）
    └─> 目标域few-shot数据编码（5张/用户）
             │
             v
    Latent数据(.pt文件)
             │
             v
        DDPM训练
```

## 注意事项

1. VAE应该在**所有数据**上训练（不区分域）
2. prepare_microdoppler_data.py需要**已训练好的VAE**
3. DDPM使用的是**编码后的latent**，不是原始图像

## Kaggle环境路径参考

如果在Kaggle中运行，需要调整路径：
- 数据集路径: `/kaggle/input/organized-gait-dataset`
- 分类器路径: `/kaggle/input/improved-classifier/best_improved_classifier.pth`
- 输出路径: `/kaggle/working/`
- VAE checkpoint: `/kaggle/working/kl_vae_best.pt`

