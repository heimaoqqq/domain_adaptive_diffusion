#!/usr/bin/env python3
"""
Kaggle环境初始化脚本
在Kaggle Notebook中运行此脚本以设置环境
"""

import os
import shutil
import sys
from pathlib import Path

def setup_kaggle_environment():
    """设置Kaggle环境"""
    
    print("=" * 60)
    print("Kaggle环境初始化")
    print("=" * 60)
    
    # 检查是否在Kaggle环境
    if not os.path.exists('/kaggle/working'):
        print("❌ 不在Kaggle环境中，此脚本仅用于Kaggle")
        return False
    
    print("✅ 检测到Kaggle环境")
    
    # 检查必需的输入数据集
    required_inputs = {
        'organized-gait-dataset': '/kaggle/input/organized-gait-dataset',
        'stage3 (VAE)': '/kaggle/input/stage3/vavae-stage3-epoch26-val_rec_loss0.0000.ckpt',
        'best-improved-classifier': '/kaggle/input/best-improved-classifier-pth/best_improved_classifier.pth',
    }
    
    print("\n检查输入数据集:")
    all_present = True
    for name, path in required_inputs.items():
        if os.path.exists(path):
            print(f"  ✅ {name}: {path}")
        else:
            print(f"  ❌ {name}: 未找到 {path}")
            all_present = False
    
    if not all_present:
        print("\n❌ 缺少必需的输入数据集")
        print("请在Kaggle Notebook中添加以下数据集:")
        print("  1. organized-gait-dataset")
        print("  2. stage3 (VAE checkpoint)")
        print("  3. best-improved-classifier-pth")
        return False
    
    # 复制simplified_vavae.py到working目录（如果存在）
    if os.path.exists('/kaggle/input/simplified-vavae/simplified_vavae.py'):
        shutil.copy('/kaggle/input/simplified-vavae/simplified_vavae.py', 
                   '/kaggle/working/simplified_vavae.py')
        print("\n✅ 复制simplified_vavae.py到working目录")
    else:
        print("\n⚠️ 未找到simplified_vavae.py，需要手动上传到/kaggle/working/")
    
    # 检查domain_adaptive_diffusion项目
    if not os.path.exists('/kaggle/working/domain_adaptive_diffusion'):
        print("\n📦 克隆domain_adaptive_diffusion项目...")
        os.system('git clone https://github.com/heimaoqqq/domain_adaptive_diffusion.git /kaggle/working/domain_adaptive_diffusion')
    else:
        print("\n✅ domain_adaptive_diffusion项目已存在")
    
    # 安装依赖
    print("\n📦 安装Python依赖...")
    os.system('pip install -q denoising-diffusion-pytorch safetensors')
    
    # 创建输出目录
    output_dirs = [
        '/kaggle/working/data',
        '/kaggle/working/checkpoints',
        '/kaggle/working/outputs',
    ]
    
    for dir_path in output_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("\n✅ 创建输出目录:")
    for dir_path in output_dirs:
        print(f"  - {dir_path}")
    
    print("\n" + "=" * 60)
    print("✅ Kaggle环境初始化完成!")
    print("=" * 60)
    
    print("\n下一步:")
    print("1. 上传simplified_vavae.py到/kaggle/working/ (如果还没有)")
    print("\n2. 切换到项目目录:")
    print("   cd /kaggle/working/domain_adaptive_diffusion")
    print("\n3. 准备数据:")
    print("   python scripts/prepare_microdoppler_data.py")
    print("\n4. 开始训练:")
    print("   python scripts/train.py --config configs/domain_ada.yaml")
    
    return True

if __name__ == "__main__":
    success = setup_kaggle_environment()
    sys.exit(0 if success else 1)
