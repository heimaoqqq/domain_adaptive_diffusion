# Domain Adaptive Diffusion

VAE-integrated Guided Diffusion for micro-Doppler gait dataset.

## Features

- **VAE Integration**: Train in latent space (4x compression) for faster training
- **256x256 Resolution**: High-quality micro-Doppler spectrograms
- **31 User Classes**: Multi-user gait recognition
- **Based on OpenAI's Guided Diffusion**: Stable and proven architecture
- **Zero Initialization**: Proper output layer initialization for stable training

## Quick Start

### Installation
```bash
pip install torch torchvision numpy pillow
```

### Training
```bash
python scripts/train_vae.py \
    --data_dir ../dataset/organized_gait_dataset \
    --image_size 256 \
    --batch_size 8 \
    --class_cond True \
    --random_flip False
```

### Sampling
```bash
python scripts/sample_vae.py \
    --model_path logs/model.pt \
    --image_size 256 \
    --num_samples 31 \
    --use_ddim True \
    --ddim_steps 50
```

## Key Improvements

1. **Zero Module Initialization**: Following official Guided Diffusion
2. **Proper Scaling**: Latents normalized to std≈1.0 during training
3. **DDIM Sampling**: Fast 50-step generation by default
4. **No Data Augmentation**: Preserving micro-Doppler patterns

## Architecture

- Base channels: 128
- Channel multipliers: [1, 1, 2, 2, 4, 4] (for 256x256)
- Attention resolutions: [32, 16, 8]
- VAE latent: 64x64 (4x downsampling)

## Dataset Structure

```
dataset/organized_gait_dataset/
├── user_0/
├── user_1/
...
└── user_30/
```

## Citation

Based on "Diffusion Models Beat GANs on Image Synthesis" by OpenAI.
