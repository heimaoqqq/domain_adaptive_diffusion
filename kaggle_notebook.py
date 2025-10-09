"""
Kaggle Notebook运行脚本
直接在Kaggle Notebook中运行此文件
"""

import os
import shutil
import subprocess
import sys

def setup_and_run():
    """在Kaggle中设置环境并运行数据准备"""
    
    print("=" * 60)
    print("Kaggle域自适应扩散模型 - 环境设置")
    print("=" * 60)
    
    # 1. 检查必需的数据集
    required = {
        'organized-gait-dataset': '/kaggle/input/organized-gait-dataset',
        'VAE checkpoint': '/kaggle/input/stage3/vavae-stage3-epoch26-val_rec_loss0.0000.ckpt',
        'Classifier': '/kaggle/input/best-improved-classifier-pth/best_improved_classifier.pth',
    }
    
    print("\n✅ 检查输入数据:")
    for name, path in required.items():
        if os.path.exists(path):
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name} - 缺失!")
            
    # 2. 复制simplified_vavae.py
    if os.path.exists('/kaggle/input/simplified-vavae/simplified_vavae.py'):
        shutil.copy('/kaggle/input/simplified-vavae/simplified_vavae.py', 
                   '/kaggle/working/simplified_vavae.py')
        print("\n✅ 已复制simplified_vavae.py")
    else:
        print("\n⚠️ 请上传simplified_vavae.py到/kaggle/working/")
        print("   或创建名为'simplified-vavae'的数据集")
        return
    
    # 3. 克隆项目
    if not os.path.exists('/kaggle/working/domain_adaptive_diffusion'):
        print("\n📦 克隆项目...")
        subprocess.run(['git', 'clone', 
                       'https://github.com/heimaoqqq/domain_adaptive_diffusion.git',
                       '/kaggle/working/domain_adaptive_diffusion'])
    
    # 4. 安装依赖
    print("\n📦 安装依赖...")
    subprocess.run(['pip', 'install', '-q', 'denoising-diffusion-pytorch', 'safetensors'])
    
    # 5. 切换到项目目录
    os.chdir('/kaggle/working/domain_adaptive_diffusion')
    
    # 6. 运行数据准备
    print("\n" + "=" * 60)
    print("🚀 开始准备数据...")
    print("=" * 60)
    
    subprocess.run([sys.executable, 'scripts/prepare_microdoppler_data.py'])
    
    print("\n✅ 数据准备完成!")
    print("\n下一步: 运行训练")
    print("!python scripts/train.py --config configs/domain_ada.yaml")

if __name__ == "__main__":
    setup_and_run()
