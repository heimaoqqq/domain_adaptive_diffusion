"""
VAE包装器 - 提供统一的VAE接口用于Guided-Diffusion集成
"""

import torch
import torch.nn as nn
import sys
import os
from pathlib import Path

# 路径将在_load_vae方法中处理

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
        # 添加项目根目录到sys.path（参考train_adm_official.py的做法）
        import sys
        from pathlib import Path
        
        # 从当前文件位置向上找到项目根目录
        current_file = Path(__file__).resolve()
        
        # 检查是否在Kaggle环境
        if '/kaggle/working' in str(current_file):
            # 在Kaggle中，domain_adaptive_diffusion直接在/kaggle/working下
            project_root = Path('/kaggle/working')
            print(f"检测到Kaggle环境，设置项目根目录为: {project_root}")
        else:
            # 本地环境: guided_diffusion_vae/vae_wrapper.py -> 向上两级到达项目根目录
            project_root = current_file.parent.parent
            print(f"本地环境，项目根目录为: {project_root}")
        
        # 添加到sys.path
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
            print(f"添加到sys.path: {project_root}")
        
        try:
            # 尝试从多个位置导入
            try:
                # 如果kl_vae.py在guided_diffusion_vae/vae/目录下
                from vae.kl_vae import KL_VAE
                print("✓ 成功导入KL_VAE (from vae.kl_vae)")
            except ImportError:
                # 如果在原始位置
                from domain_adaptive_diffusion.vae.kl_vae import KL_VAE
                print("✓ 成功导入KL_VAE (from domain_adaptive_diffusion.vae.kl_vae)")
            
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
                
                # 冻结VAE参数
                for param in self.vae.parameters():
                    param.requires_grad = False
            else:
                print(f"⚠️ 警告: VAE权重文件不存在: {vae_path}")
                raise FileNotFoundError(f"必须提供预训练的VAE权重: {vae_path}")
            
            self.vae = self.vae.to(self.device)
            self.vae.eval()
            
        except ImportError as e:
            print(f"\n❌ 无法导入KL_VAE模块: {e}")
            print("\n可能的解决方案：")
            print("1. 确保domain_adaptive_diffusion目录在正确的位置")
            print("2. 在Kaggle中，确保文件结构如下：")
            print("   /kaggle/working/domain_adaptive_diffusion/")
            print("   ├── vae/")
            print("   │   └── kl_vae.py")
            print("   └── guided_diffusion_vae/")
            print("       └── vae_wrapper.py (当前文件)")
            
            # 尝试打印一些调试信息
            print(f"\n当前工作目录: {os.getcwd()}")
            print(f"当前文件路径: {Path(__file__).resolve()}")
            print(f"项目根目录: {project_root.resolve()}")
            print(f"sys.path包含: {sys.path[:5]}...")  # 只显示前5个
            
            # 检查文件是否存在
            kl_vae_file = project_root / 'domain_adaptive_diffusion' / 'vae' / 'kl_vae.py'
            if kl_vae_file.exists():
                print(f"\n✓ 找到kl_vae.py文件: {kl_vae_file}")
                print("但是导入失败，可能是依赖问题")
            else:
                print(f"\n❌ 找不到kl_vae.py文件，预期位置: {kl_vae_file}")
                
            raise ImportError(f"无法导入KL_VAE，请检查文件结构和依赖: {e}")
    
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

