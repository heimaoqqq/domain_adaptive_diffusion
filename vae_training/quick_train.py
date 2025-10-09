"""
Quick VAE training script for Kaggle
快速训练VAE - 适用于Kaggle环境
"""

import os
import sys

# Auto-detect environment
is_kaggle = os.path.exists('/kaggle/working')

if is_kaggle:
    # Kaggle paths
    data_dir = '/kaggle/input/organized-gait-dataset/Normal_line'
    checkpoint_dir = '/kaggle/working/vae_checkpoints'
else:
    # Local paths
    data_dir = 'D:\\Ysj\\新建文件夹\\VA-VAE\\dataset\\organized_gait_dataset\\Normal_line'
    checkpoint_dir = 'vae_checkpoints'

# Quick training with reduced epochs for testing
cmd = f"""
python train_vae.py \\
    --data_dir "{data_dir}" \\
    --checkpoint_dir "{checkpoint_dir}" \\
    --epochs 10 \\
    --batch_size 32 \\
    --save_every 2 \\
    --latent_channels 32 \\
    --scale_factor 0.18215
"""

print("=" * 60)
print("Quick VAE Training")
print("=" * 60)
print(f"Data directory: {data_dir}")
print(f"Checkpoint directory: {checkpoint_dir}")
print("=" * 60)

# First, analyze latent statistics with untrained model
print("\n1. Analyzing initial latent statistics...")
analyze_cmd = cmd.replace("--epochs 10", "--epochs 0 --analyze_only")
os.system(analyze_cmd)

# Ask user if they want to continue with training
if is_kaggle:
    # In Kaggle, always continue
    print("\n2. Starting training...")
    os.system(cmd)
else:
    # In local environment, ask user
    response = input("\nContinue with training? (y/n): ")
    if response.lower() == 'y':
        print("\n2. Starting training...")
        os.system(cmd)
    else:
        print("Training cancelled.")

print("\n✅ Done!")
print(f"Checkpoints saved in: {checkpoint_dir}")
print("\nTo use the trained VAE with DDPM:")
print(f"python scripts/prepare_microdoppler_data.py --vae_checkpoint {checkpoint_dir}/best_vae.pt")
