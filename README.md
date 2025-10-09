# Domain-Adaptive Diffusion for Few-Shot Micro-Doppler Generation

基于DDPM的域自适应扩散模型，用于微多普勒时频图的少样本域适应生成。

## 特点

- **Few-Shot域适应**：仅需目标域每用户3-5张图像
- **三阶段训练策略**：预训练→MMD对齐→微调
- **基于diversity的样本选择**：使用源域分类器智能选择目标域样本
- **UNet-based DDPM**：~30-50M参数的轻量级扩散模型

## 快速开始（云服务器）

```bash
# 1. 克隆项目
git clone https://github.com/heimaoqqq/domain_adaptive_diffusion.git
cd domain_adaptive_diffusion

# 2. 安装依赖
pip install -r requirements.txt

# 3. 准备数据和VAE
# 上传您的VAE checkpoint和数据集到服务器
# 修改scripts/prepare_microdoppler_data.py中的路径

# 4. 数据准备
cd scripts
python prepare_microdoppler_data.py

# 5. 训练模型
python train.py --config ../configs/domain_ada.yaml
```

## 环境要求

```bash
# Python 3.8+
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install denoising-diffusion-pytorch
pip install pillow numpy tqdm matplotlib pyyaml
pip install scikit-learn  # For K-means clustering
pip install safetensors  # For data storage
```

## 数据准备

### 1. 准备VAE模型
需要预训练的VA-VAE模型：
- Checkpoint路径：`path/to/vavae-stage3-epoch26-val_rec_loss0.0000.ckpt`
- 模型脚本：`path/to/simplified_vavae.py`

### 2. 准备数据集
```
dataset/
├── source_domain/  # 源域数据 (例如: Normal_line)
│   ├── ID_1/      # 用户1的图像
│   ├── ID_2/
│   └── ...
└── target_domain/  # 目标域数据 (例如: Normal_free)
    ├── ID_1/
    ├── ID_2/
    └── ...
```

### 3. 运行数据准备脚本

```bash
cd scripts

# 修改路径配置
python prepare_microdoppler_data.py \
    --dataset_root /path/to/dataset \
    --source_gait source_domain \
    --target_gait target_domain \
    --vae_checkpoint /path/to/vae.ckpt \
    --vae_script /path/to/simplified_vavae.py
```

这会：
- 源域：使用所有数据（~150张/用户）
- 目标域：diversity选择5张/用户
- 编码为latent并保存

## 训练

### 配置文件
修改 `configs/domain_ada.yaml`：

```yaml
training:
  pretrain:
    epochs: 50        # 源域预训练
    batch_size: 64
  align:
    epochs: 60        # MMD域对齐
    batch_size_source: 48
    batch_size_target: 16
  finetune:
    epochs: 30        # 目标域微调
    batch_size: 32

domain_adaptation:
  mmd_weight_schedule: cosine
  mmd_start_weight: 0.01
  mmd_end_weight: 0.15
  mmd_kernel_bandwidth: [0.25, 0.5, 1.0, 2.0, 4.0]
```

### 开始训练

```bash
python scripts/train.py --config configs/domain_ada.yaml
```

训练特性：
- **自动保存**：每5个epoch保存checkpoint
- **可视化**：每2个epoch生成样本
- **自动清理**：删除旧checkpoint节省空间
- **最佳模型**：自动保存验证loss最低的模型

## 生成样本

```bash
python scripts/generate.py \
    --checkpoint checkpoints/best_model.pt \
    --num_samples 3100 \
    --batch_size 31 \
    --save_dir outputs/generated
```

## 评估

```bash
python scripts/evaluate.py \
    --generated_dir outputs/generated \
    --real_dir data/target_domain \
    --classifier_path /path/to/classifier.pth
```

## 项目结构

```
domain_adaptive_diffusion/
├── configs/               # 配置文件
│   ├── default.yaml      # 默认配置
│   └── domain_ada.yaml    # 域适应配置
├── models/               # 模型定义
│   ├── conditional_unet.py  # 条件UNet
│   ├── domain_diffusion.py  # 域感知扩散
│   └── losses.py            # 损失函数（含MMD）
├── scripts/              # 主要脚本
│   ├── prepare_microdoppler_data.py  # 数据准备
│   ├── diversity_selector.py         # Few-shot选择
│   ├── train.py                      # 训练脚本
│   ├── generate.py                   # 生成脚本
│   └── evaluate.py                   # 评估脚本
└── utils/                # 工具函数
    ├── data_loader.py    # 数据加载
    ├── metrics.py        # 评估指标
    └── visualization.py  # 可视化工具
```

## 关键参数说明

### Scale Factor
VAE的scale factor影响latent分布：
- **scale_factor = 1.0**：保持原始分布（std ≈ 1.55）
- **scale_factor = 0.64**：归一化到单位方差

根据实际训练效果调整。

### MMD损失
用于域对齐的Maximum Mean Discrepancy损失：
- 多尺度RBF核：[0.25, 0.5, 1.0, 2.0, 4.0]
- 余弦退火权重：0.01 → 0.15

### GPU内存
- 48GB GPU推荐batch size：
  - 预训练：64
  - MMD对齐：源48 + 目标16
  - 微调：32

## 注意事项

1. **数据格式**：输入图像应为256x256 RGB格式
2. **VAE兼容**：确保VAE的latent维度为32，下采样16倍
3. **Few-Shot**：目标域样本质量很重要，diversity选择确保覆盖分布
4. **训练监控**：观察loss曲线判断是否需要调整学习率或MMD权重

## 引用

基于以下项目：
- [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch)
- VA-VAE架构

## License

MIT