# 域自适应扩散模型 (Domain Adaptive Diffusion)

针对小数据集（31用户，每用户150张图像）的微多普勒生成任务优化的扩散模型。

## 核心架构选择

### 推荐方案：KL-VAE + 连续扩散
- **VAE**: KL-VAE (基于Stable Diffusion架构)
- **原因**: 简单、稳定、适合小数据集
- **特点**: 4x下采样，连续latent空间

## 项目结构

```
domain_adaptive_diffusion/
├── models/                      # 核心模型
│   ├── domain_diffusion.py     # 域自适应扩散模型
│   ├── conditional_unet.py     # 条件U-Net
│   └── losses.py              # MMD等损失函数
├── scripts/                    # 执行脚本  
│   ├── prepare_microdoppler_data.py  # 数据准备(先运行)
│   ├── train.py               # 训练模型
│   ├── generate.py            # 生成样本
│   └── evaluate.py            # 评估结果
├── configs/                    # 配置文件
│   ├── default.yaml           # 默认配置
│   └── domain_ada.yaml        # 域适应配置
├── utils/                      # 工具函数
└── vae/                       # VAE实现
    ├── kl_vae.py             # KL-VAE模型
    └── train_kl_vae.py       # VAE训练脚本
```

## 快速开始

### 1. 准备数据
```bash
cd scripts
python prepare_microdoppler_data.py \
    --source_domain Normal_line \
    --target_domain Normal_free \
    --train_ratio 0.8
```

### 2. 训练VAE（可选，如果没有预训练的）
```bash
cd ../vae
python train_kl_vae.py --data_path ../data/microdoppler_processed
```

### 3. 训练扩散模型
```bash
cd ../scripts
python train.py \
    --config ../configs/domain_ada.yaml \
    --source_domain Normal_line \
    --target_domain Normal_free
```

### 4. 生成样本
```bash
python generate.py \
    --checkpoint ../checkpoints/best_model.pt \
    --num_samples 100
```

## 关键参数

### 小数据集优化
- **批次大小**: 8-16 (防止过拟合)
- **学习率**: 1e-4 (稳定训练)
- **MMD权重**: 0.1-0.5 (域对齐强度)
- **训练轮数**: 100-200 epochs

### Few-shot设置
- **源域**: 全部数据
- **目标域**: 每用户5张 (diversity策略选择)

## 注意事项

1. **VAE选择**: 使用KL-VAE而非VQ-VAE，因为：
   - 训练更简单
   - 适合小数据集
   - 不需要复杂的离散建模

2. **内存优化**: 
   - 48GB GPU可以使用batch_size=16
   - 如果OOM，减小到8

3. **Kaggle环境**: 
   - 自动检测并适配路径
   - VAE会自动从输入数据集加载
