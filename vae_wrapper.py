"""
VAE包装器 - 提供统一的VAE接口用于Guided-Diffusion集成
"""

import torch
import torch.nn as nn
import sys
import os
from pathlib import Path

# 添加可能的路径到sys.path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent

# 尝试添加多个可能的路径
for path in [
    parent_dir,  # guided_diffusion_vae的父目录
    parent_dir.parent,  # 再上一级
    Path('/kaggle/working'),  # Kaggle工作目录
    Path('/kaggle/working/domain_adaptive_diffusion'),  # Kaggle特定路径
]:
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

class VAEInterface:
    """VAE接口类，处理编码和解码"""
    
    def __init__(self, vae_path="domain_adaptive_diffusion/vae/vae_model.pt", device='cuda'):
        """
        初始化VAE接口
        
        Args:
            vae_path: VAE模型路径
            device: 运行设备
        """
        self.device = device
        self.scale_factor = 0.18215  # 标准LDM缩放因子
        
        # 加载VAE
        self._load_vae(vae_path)
        
    def _load_vae(self, vae_path):
        """加载VAE模型"""
        try:
            # 尝试导入VAE模块
            from domain_adaptive_diffusion.vae.kl_vae import KL_VAE
            
            # 创建KL_VAE实例 - 使用默认配置
            self.vae = KL_VAE(
                ddconfig=None,  # 使用默认配置 
                embed_dim=4,
                scale_factor=self.scale_factor
            )
            
            # 加载权重
            if os.path.exists(vae_path):
                print(f"加载VAE权重: {vae_path}")
                checkpoint = torch.load(vae_path, map_location=self.device)
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        self.vae.load_state_dict(checkpoint['model_state_dict'])
                    elif 'state_dict' in checkpoint:
                        self.vae.load_state_dict(checkpoint['state_dict'])
                    else:
                        # checkpoint本身就是state_dict
                        self.vae.load_state_dict(checkpoint)
                    
                    # 更新scale_factor如果checkpoint中有
                    if 'scale_factor' in checkpoint:
                        self.scale_factor = float(checkpoint['scale_factor'])
                        self.vae.scale_factor = self.scale_factor
                        print(f"✓ 使用checkpoint中的scale_factor: {self.scale_factor}")
                else:
                    self.vae.load_state_dict(checkpoint)
                print("✓ VAE权重加载成功")
            else:
                print(f"⚠️ 警告: VAE权重文件不存在: {vae_path}")
                print("   使用随机初始化的VAE（仅用于测试）")
            
            self.vae = self.vae.to(self.device)
            self.vae.eval()
            
        except ImportError:
            print("⚠️ 无法导入VAE模块，创建模拟VAE用于测试")
            self._create_dummy_vae()
    
    def _create_dummy_vae(self):
        """创建模拟VAE用于测试"""
        class DummyVAE(nn.Module):
            def __init__(self, scale_factor=0.18215):
                super().__init__()
                self.encoder = nn.Conv2d(3, 4, 1)  # 简单1x1卷积
                self.decoder = nn.Conv2d(4, 3, 1)
                self.scale_factor = scale_factor
                
            def encode(self, x):
                """模拟编码"""
                # 简单下采样4x
                B, C, H, W = x.shape
                x = nn.functional.interpolate(x, size=(H//4, W//4), mode='bilinear')
                return self.encoder(x)
            
            def decode(self, z):
                """模拟解码"""
                # 简单上采样4x
                x = self.decoder(z)
                H, W = x.shape[2] * 4, x.shape[3] * 4
                x = nn.functional.interpolate(x, size=(H, W), mode='bilinear')
                return torch.sigmoid(x)
                
            def encode_images(self, images):
                """兼容KL_VAE的encode_images接口"""
                latents = self.encode(images)
                return latents * self.scale_factor
                
            def decode_latents(self, latents):
                """兼容KL_VAE的decode_latents接口"""
                latents = latents / self.scale_factor
                return self.decode(latents)
        
        self.vae = DummyVAE(self.scale_factor).to(self.device)
        self.vae.eval()
        print("✓ 创建模拟VAE（仅用于测试）")
    
    def encode_batch(self, images):
        """
        编码图像批次到latent空间
        
        Args:
            images: [B, C, H, W] tensor in [0, 1] range
            
        Returns:
            latents: [B, 4, H/4, W/4] tensor, 已缩放
        """
        if not isinstance(images, torch.Tensor):
            images = torch.tensor(images)
        
        images = images.to(self.device)
        
        # 确保在[0, 1]范围
        if images.min() < -0.1 or images.max() > 1.1:
            print(f"⚠️ 输入图像范围异常: [{images.min():.2f}, {images.max():.2f}]")
            images = torch.clamp(images, 0, 1)
        
        with torch.no_grad():
            # 使用VAE的encode_images方法
            latents = self.vae.encode_images(images)
                
        return latents
    
    def decode_batch(self, latents):
        """
        解码latent批次到图像
        
        Args:
            latents: [B, 4, H, W] tensor, 已缩放的latent
            
        Returns:
            images: [B, 3, H*4, W*4] tensor in [0, 1] range
        """
        if not isinstance(latents, torch.Tensor):
            latents = torch.tensor(latents)
            
        latents = latents.to(self.device)
        
        with torch.no_grad():
            # 使用VAE的decode_latents方法
            images = self.vae.decode_latents(latents)
                
        return images
    
    def get_latent_size(self, image_size):
        """
        计算latent尺寸
        
        Args:
            image_size: 输入图像尺寸
            
        Returns:
            latent_size: latent空间尺寸
        """
        # VAE通常下采样4x
        return image_size // 4
    
    def test_roundtrip(self, image_size=64):
        """
        测试编码-解码往返
        
        Args:
            image_size: 测试图像尺寸
        """
        print(f"\n测试VAE往返 (size={image_size})...")
        
        # 创建测试图像
        test_image = torch.rand(1, 3, image_size, image_size).to(self.device)
        print(f"输入图像: {test_image.shape}, 范围: [{test_image.min():.2f}, {test_image.max():.2f}]")
        
        # 编码
        latent = self.encode_batch(test_image)
        print(f"编码后: {latent.shape}, 范围: [{latent.min():.2f}, {latent.max():.2f}]")
        print(f"        std: {latent.std():.4f}")
        
        # 解码
        reconstructed = self.decode_batch(latent)
        print(f"解码后: {reconstructed.shape}, 范围: [{reconstructed.min():.2f}, {reconstructed.max():.2f}]")
        
        # 计算误差
        mse = ((test_image - reconstructed) ** 2).mean()
        print(f"重建误差 (MSE): {mse:.4f}")
        
        return mse < 0.1  # 合理的重建误差阈值


def test_vae_interface():
    """测试VAE接口"""
    print("=" * 60)
    print("测试VAE接口")
    print("=" * 60)
    
    # 创建接口
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vae = VAEInterface(device=device)
    
    # 测试编码
    print("\n1. 测试批量编码")
    images = torch.rand(4, 3, 64, 64).to(device)
    latents = vae.encode_batch(images)
    print(f"✓ 编码成功: {images.shape} -> {latents.shape}")
    
    # 测试解码
    print("\n2. 测试批量解码")
    reconstructed = vae.decode_batch(latents)
    print(f"✓ 解码成功: {latents.shape} -> {reconstructed.shape}")
    
    # 测试往返
    print("\n3. 测试往返重建")
    success = vae.test_roundtrip(64)
    if success:
        print("✅ 往返测试通过")
    else:
        print("⚠️ 往返测试有较大误差（可能是模拟VAE）")
    
    # 测试缩放
    print("\n4. 测试标准化")
    latents_normalized = latents / vae.scale_factor
    print(f"标准化后std: {latents_normalized.std():.4f} (目标≈1.0)")
    
    print("\n" + "=" * 60)
    print("VAE接口测试完成")
    print("=" * 60)


if __name__ == "__main__":
    test_vae_interface()

