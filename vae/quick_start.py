"""
KL-VAE Quick Start
快速开始训练基于Stable Diffusion架构的VAE
"""

import os
import sys

# 检测环境
is_kaggle = os.path.exists('/kaggle/working')

print("=" * 60)
print("KL-VAE Quick Start")
print("基于Stable Diffusion架构的VAE")
print("=" * 60)

if is_kaggle:
    # Kaggle环境
    data_dir = '/kaggle/input/organized-gait-dataset/Normal_line'
    checkpoint_dir = '/kaggle/working/kl_vae_checkpoints'
    batch_size = 4  # Kaggle GPU内存较小
    epochs = 20     # 快速测试
else:
    # 本地环境
    data_dir = input("请输入数据目录路径 [默认: Normal_line]: ").strip()
    if not data_dir:
        # 尝试找到数据目录
        possible_paths = [
            r'D:\Ysj\新建文件夹\VA-VAE\dataset\organized_gait_dataset\Normal_line',
            '../../../dataset/organized_gait_dataset/Normal_line',
            'dataset/organized_gait_dataset/Normal_line'
        ]
        for path in possible_paths:
            if os.path.exists(path):
                data_dir = path
                break
        else:
            print("❌ 找不到数据目录，请手动指定")
            sys.exit(1)
    checkpoint_dir = 'kl_vae_checkpoints'
    batch_size = 8
    epochs = 50

print(f"\n配置:")
print(f"  数据目录: {data_dir}")
print(f"  保存目录: {checkpoint_dir}")
print(f"  批次大小: {batch_size}")
print(f"  训练轮数: {epochs}")

# 训练命令
train_cmd = f"""python train_kl_vae.py \\
    --data_dir "{data_dir}" \\
    --checkpoint_dir "{checkpoint_dir}" \\
    --batch_size {batch_size} \\
    --epochs {epochs} \\
    --save_every 5 \\
    --lr 4.5e-6 \\
    --kl_weight 1e-6 \\
    --warmup_epochs 5"""

print("\n" + "=" * 60)
print("步骤 1: 分析数据集统计")
print("=" * 60)

# 先分析一下
analyze_cmd = train_cmd.replace(f"--epochs {epochs}", "--epochs 0 --analyze_only")
os.system(analyze_cmd)

if not is_kaggle:
    response = input("\n继续训练? (y/n): ")
    if response.lower() != 'y':
        print("训练取消")
        sys.exit(0)

print("\n" + "=" * 60)
print("步骤 2: 开始训练")
print("=" * 60)

os.system(train_cmd)

print("\n" + "=" * 60)
print("✅ 训练完成!")
print("=" * 60)

print(f"\n下一步:")
print(f"1. 使用训练好的VAE准备数据:")
print(f"   python ../scripts/prepare_microdoppler_data.py --vae_checkpoint {checkpoint_dir}/best_kl_vae.pt")
print(f"\n2. 训练DDPM:")
print(f"   python ../scripts/train.py --config ../configs/domain_ada.yaml")

print("\n💡 提示:")
print("- KL-VAE使用4x下采样 (256x256 → 64x64)")
print("- 比原来的16x下采样保留更多细节")
print("- 标准scale_factor = 0.18215")
print("- 如果重建效果不好，可以调整kl_weight (默认1e-6)")
