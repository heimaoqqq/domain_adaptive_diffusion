# Domain-Adaptive Diffusion for Few-Shot Micro-Doppler Generation

基于DDPM的域自适应扩散模型，用于微多普勒时频图的少样本域适应生成。

## 🚀 Kaggle快速开始

### 方法1：使用kaggle_notebook.py（推荐）
```python
# 上传simplified_vavae.py到/kaggle/working/
# 然后运行：
!wget https://raw.githubusercontent.com/heimaoqqq/domain_adaptive_diffusion/main/kaggle_notebook.py
!python kaggle_notebook.py
```

### 方法2：手动步骤
```python
# 1. 克隆项目
!git clone https://github.com/heimaoqqq/domain_adaptive_diffusion.git
%cd domain_adaptive_diffusion

# 2. 上传simplified_vavae.py到/kaggle/working/

# 3. 安装依赖
!pip install -q denoising-diffusion-pytorch safetensors

# 4. 准备数据（自动检测Kaggle路径）
!python scripts/prepare_microdoppler_data.py

# 5. 开始训练
!python scripts/train.py --config configs/domain_ada.yaml
```

### Kaggle数据集要求

请在Kaggle Notebook中添加以下数据集：
- `organized-gait-dataset`: 包含Normal_line和Normal_free文件夹
- `stage3`: 包含VAE checkpoint (vavae-stage3-epoch26-val_rec_loss0.0000.ckpt)
- `best-improved-classifier-pth`: 包含分类器 (best_improved_classifier.pth)

上传`simplified_vavae.py`到`/kaggle/working/`目录。

## 特点

- **Few-Shot域适应**：仅需目标域每用户3-5张图像
- **三阶段训练策略**：预训练→MMD对齐→微调
- **智能路径检测**：自动检测Kaggle/本地环境
- **基于diversity的样本选择**：使用源域分类器智能选择目标域样本
- **UNet-based DDPM**：~30-50M参数的轻量级扩散模型

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
```

## 训练配置

修改 `configs/domain_ada.yaml` 调整训练参数：

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
```

## License

MIT