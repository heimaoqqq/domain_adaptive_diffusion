"""
快速验证latent scale问题
"""
import torch
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from domain_adaptive_diffusion.vae.kl_vae import KL_VAE
from domain_adaptive_diffusion.utils.data_loader import create_dataloaders
from torchvision.utils import save_image
import matplotlib.pyplot as plt


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. 加载VAE
    print("加载VAE...")
    vae = KL_VAE(ddconfig=None, embed_dim=4, scale_factor=0.18215)
    checkpoint = torch.load("/kaggle/input/kl-vae-best-pt/kl_vae_best.pt", map_location='cpu')
    if 'model_state_dict' in checkpoint:
        vae.load_state_dict(checkpoint['model_state_dict'])
    else:
        vae.load_state_dict(checkpoint)
    vae.eval().to(device)
    
    # 2. 加载训练数据
    print("\n加载训练数据...")
    train_loader, _ = create_dataloaders(
        data_path="/kaggle/input/data-latent",
        phase='pretrain',
        batch_size=8,
        num_workers=0,
        augmentation=False,
        device=device
    )
    
    batch = next(iter(train_loader))
    dataset_latents = batch['latent'].to(device)
    
    print(f"数据集中的latents统计:")
    print(f"  Mean: {dataset_latents.mean():.4f}")
    print(f"  Std: {dataset_latents.std():.4f}")
    print(f"  Min: {dataset_latents.min():.4f}")
    print(f"  Max: {dataset_latents.max():.4f}")
    
    # 3. 测试不同的scale解码
    print("\n测试不同scale的解码效果:")
    
    scales_to_test = [1.0, 1/0.18215, 0.18215, 4.0, 8.0]
    results = []
    
    for i, scale in enumerate(scales_to_test):
        with torch.no_grad():
            # 应用scale
            scaled_latents = dataset_latents * scale
            
            # 直接通过decoder解码（绕过decode_latents方法）
            decoded = vae.decode(scaled_latents)
            
            print(f"\nScale={scale:.4f}:")
            print(f"  Scaled latent range: [{scaled_latents.min():.4f}, {scaled_latents.max():.4f}]")
            print(f"  Decoded range: [{decoded.min():.4f}, {decoded.max():.4f}]")
            print(f"  Decoded mean: {decoded.mean():.4f}")
            print(f"  Decoded std: {decoded.std():.4f}")
            
            # 计算图像的边缘强度
            gray = decoded.mean(dim=1, keepdim=True)
            dx = gray[:, :, :, 1:] - gray[:, :, :, :-1]
            dy = gray[:, :, 1:, :] - gray[:, :, :-1, :]
            edge_strength = (dx.abs().mean() + dy.abs().mean()) / 2
            print(f"  边缘强度: {edge_strength:.4f}")
            
            results.append({
                'scale': scale,
                'decoded': decoded,
                'edge_strength': edge_strength
            })
    
    # 4. 找出最佳scale
    best_result = max(results, key=lambda x: x['edge_strength'])
    print(f"\n最佳scale: {best_result['scale']:.4f} (边缘强度: {best_result['edge_strength']:.4f})")
    
    # 5. 保存对比图
    print("\n保存对比图...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, result in enumerate(results[:6]):
        decoded = result['decoded']
        
        # 归一化到[0,1]
        if decoded.min() >= -1.1 and decoded.max() <= 1.1:
            decoded_norm = (decoded + 1) / 2
        else:
            decoded_norm = (decoded - decoded.min()) / (decoded.max() - decoded.min() + 1e-8)
        
        # 显示第一张图像
        img = decoded_norm[0].cpu().permute(1, 2, 0).numpy()
        axes[i].imshow(img)
        axes[i].set_title(f"Scale={result['scale']:.4f}\nEdge={result['edge_strength']:.4f}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('latent_scale_comparison.png')
    print("✅ 对比图已保存到: latent_scale_comparison.png")
    
    # 6. 建议
    print("\n" + "="*50)
    print("诊断建议:")
    print("="*50)
    
    if best_result['scale'] != 1.0:
        print(f"⚠️ 数据集中的latents可能需要乘以 {best_result['scale']:.4f} 才能正确解码！")
        print(f"   这表明数据准备时可能没有正确应用VAE的scale_factor")
        print(f"\n建议修复方案:")
        print(f"1. 重新准备数据集，确保保存latents时没有应用scale_factor")
        print(f"2. 或者在训练时调整数据加载，将latents除以 {best_result['scale']:.4f}")
    else:
        print("✅ Latent scale看起来正确")
    
    # 检查原始数据范围
    if dataset_latents.std() < 0.5:
        print(f"\n⚠️ 数据集latents的标准差较小 ({dataset_latents.std():.4f})，可能缺乏多样性")


if __name__ == "__main__":
    main()
