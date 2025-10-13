"""
诊断VAE重建误差问题 - 专注于归一化和缩放
"""

import torch
import sys
import os
from pathlib import Path
import numpy as np

# 添加项目路径
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from vae_wrapper import VAEInterface

def diagnose_vae():
    """诊断VAE重建问题"""
    print("=" * 60)
    print("VAE归一化诊断")
    print("=" * 60)
    
    # 创建VAE接口
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vae_path = "/kaggle/input/kl-vae-best-pt/kl_vae_best.pt"
    if not os.path.exists(vae_path):
        vae_path = "domain_adaptive_diffusion/vae/kl_vae_best.pt"
    
    vae = VAEInterface(vae_path=vae_path, device=device)
    
    print(f"\nVAE配置:")
    print(f"  scale_factor: {vae.scale_factor}")
    print(f"  设备: {device}")
    
    # 测试不同的归一化方式
    print("\n1. 测试不同的输入归一化")
    
    # 创建测试图像 (模拟真实的微多普勒图像)
    test_img_raw = torch.rand(1, 3, 256, 256).to(device)
    
    # 测试[0, 1]范围
    print("\n- 输入范围 [0, 1]:")
    test_reconstruction(vae, test_img_raw, "[0, 1]范围")
    
    # 测试[-1, 1]范围
    print("\n- 输入范围 [-1, 1]:")
    test_img_centered = test_img_raw * 2 - 1
    test_reconstruction(vae, test_img_centered, "[-1, 1]范围")
    
    # 测试ImageNet标准化
    print("\n- ImageNet标准化:")
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    test_img_imagenet = (test_img_raw - mean) / std
    test_reconstruction(vae, test_img_imagenet, "ImageNet标准化")
    
    # 分析VAE内部行为
    print("\n2. 分析VAE编码/解码细节")
    analyze_vae_internals(vae, test_img_raw)
    
    # 测试真实数据范围
    print("\n3. 测试真实数据范围模拟")
    # 模拟微多普勒数据（通常有特定的值范围）
    # 假设原始数据是uint8 [0, 255]，然后归一化到[0, 1]
    test_real_data(vae)
        
def test_reconstruction(vae, image, name):
    """测试单个图像的重建"""
    print(f"  输入范围: [{image.min():.3f}, {image.max():.3f}]")
    
    # 编码
    with torch.no_grad():
        latent = vae.encode_batch(image)
    print(f"  编码latent: shape={latent.shape}, mean={latent.mean():.4f}, std={latent.std():.4f}")
    print(f"  编码latent范围: [{latent.min():.3f}, {latent.max():.3f}]")
    
    # 解码
    with torch.no_grad():
        recon = vae.decode_batch(latent)
    print(f"  重建范围: [{recon.min():.3f}, {recon.max():.3f}]")
    
    # 将两者都clamp到[0,1]来计算误差
    image_clamped = torch.clamp(image, 0, 1)
    recon_clamped = torch.clamp(recon, 0, 1)
    
    # 计算误差
    mse = ((image_clamped - recon_clamped) ** 2).mean().item()
    mae = (image_clamped - recon_clamped).abs().mean().item()
    
    print(f"  重建误差: MSE={mse:.4f}, MAE={mae:.4f}")
    
    # 分析误差分布
    error = (image_clamped - recon_clamped).abs()
    print(f"  误差分布: mean={error.mean():.4f}, max={error.max():.4f}")
    
def analyze_vae_internals(vae, test_img):
    """分析VAE内部处理细节"""
    print("\n分析VAE内部处理:")
    
    # 直接调用VAE的encode方法查看中间结果
    with torch.no_grad():
        # 编码
        posterior = vae.vae.encode(test_img)
        z = posterior.sample()
        
        print(f"  原始latent (未缩放): mean={z.mean():.4f}, std={z.std():.4f}")
        print(f"  原始latent范围: [{z.min():.3f}, {z.max():.3f}]")
        
        # 应用scale factor
        z_scaled = z * vae.scale_factor
        print(f"  缩放后latent: mean={z_scaled.mean():.4f}, std={z_scaled.std():.4f}")
        print(f"  缩放后范围: [{z_scaled.min():.3f}, {z_scaled.max():.3f}]")
        
        # 解码
        z_unscaled = z_scaled / vae.scale_factor
        decoded = vae.vae.decode(z_unscaled)
        
        print(f"  解码输出范围: [{decoded.min():.3f}, {decoded.max():.3f}]")
        
def test_real_data(vae):
    """测试真实数据的典型范围"""
    print("\n测试真实数据范围:")
    
    # 模拟从PIL Image加载的数据
    # uint8 [0, 255] -> float32 [0, 1]
    fake_uint8 = torch.randint(0, 256, (1, 3, 256, 256), dtype=torch.uint8)
    fake_normalized = fake_uint8.float() / 255.0
    fake_normalized = fake_normalized.to(vae.device)
    
    print("\n- 模拟真实图像 (uint8归一化):")
    test_reconstruction(vae, fake_normalized, "真实数据模拟")
    
    # 测试不同的预处理
    print("\n建议:")
    print("  1. 确保输入图像在[0, 1]范围")
    print("  2. 使用 image = image.float() / 255.0 进行归一化")
    print("  3. 不要使用ImageNet标准化或[-1, 1]范围")

if __name__ == "__main__":
    diagnose_vae()
