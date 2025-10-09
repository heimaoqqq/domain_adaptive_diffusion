"""
快速测试VAE功能
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# 添加模块路径
sys.path.append(str(Path(__file__).parent))
from kl_vae import KL_VAE


def create_test_image():
    """创建一个测试的微多普勒风格图像"""
    # 创建时频图风格的测试图像
    t = np.linspace(0, 1, 256)
    f = np.linspace(0, 1, 256)
    T, F = np.meshgrid(t, f)
    
    # 模拟微多普勒信号
    signal = np.sin(2 * np.pi * 5 * T) * np.exp(-F * 2)
    signal += 0.5 * np.sin(2 * np.pi * 10 * T + F) * np.exp(-F * 3)
    
    # 添加噪声
    signal += 0.1 * np.random.randn(256, 256)
    
    # 归一化到[0, 1]
    signal = (signal - signal.min()) / (signal.max() - signal.min())
    
    # 转换为RGB
    img = np.stack([signal] * 3, axis=-1)
    
    return torch.from_numpy(img).float().permute(2, 0, 1)


def test_vae():
    """测试VAE基本功能"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 创建模型
    print("\n1. 创建KL-VAE模型...")
    vae = KL_VAE(
        embed_dim=4,
        ch_mult=(1, 2, 2, 4)
    ).to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in vae.parameters())
    print(f"   总参数量: {total_params:,}")
    print(f"   下采样: {2**len(vae.encoder.ch_mult)}x")
    print(f"   Latent通道: {vae.embed_dim}")
    
    # 创建测试图像
    print("\n2. 创建测试图像...")
    test_img = create_test_image().unsqueeze(0).to(device)
    print(f"   输入形状: {test_img.shape}")
    
    # 测试编码
    print("\n3. 测试编码...")
    with torch.no_grad():
        latent = vae.encode(test_img)
    print(f"   Latent形状: {latent.shape}")
    print(f"   Latent统计: mean={latent.mean():.3f}, std={latent.std():.3f}")
    
    # 测试解码
    print("\n4. 测试解码...")
    with torch.no_grad():
        recon = vae.decode(latent)
    print(f"   重建形状: {recon.shape}")
    
    # 测试完整前向传播
    print("\n5. 测试完整前向传播...")
    with torch.no_grad():
        recon, posterior = vae(test_img)
        kl_loss = posterior.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
    
    rec_loss = torch.nn.functional.mse_loss(test_img, recon)
    print(f"   重建损失: {rec_loss:.4f}")
    print(f"   KL损失: {kl_loss:.4f}")
    
    # 可视化
    print("\n6. 保存可视化...")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # 原图
    axes[0].imshow(test_img[0].cpu().permute(1, 2, 0))
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # 重建
    axes[1].imshow(recon[0].cpu().permute(1, 2, 0).clamp(0, 1))
    axes[1].set_title(f'Reconstruction (MSE: {rec_loss:.4f})')
    axes[1].axis('off')
    
    # 差异
    diff = torch.abs(test_img[0] - recon[0]).cpu()
    axes[2].imshow(diff.permute(1, 2, 0) * 5, cmap='hot')
    axes[2].set_title('Difference (5x)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('test_vae_result.png', dpi=150, bbox_inches='tight')
    print("   保存到: test_vae_result.png")
    
    print("\n✅ VAE测试完成！所有功能正常。")
    

if __name__ == '__main__':
    test_vae()
