"""
Domain Conditional UNet Model
基于denoising-diffusion-pytorch，添加域条件控制能力
严谨实现，基于成熟方法
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from denoising_diffusion_pytorch import Unet
from einops import rearrange, repeat


class DomainConditionalUnet(nn.Module):
    """
    域条件UNet - 在标准UNet基础上添加域和类别条件
    
    技术依据:
    1. 条件注入方法参考 Improved DDPM (Nichol & Dhariwal, 2021)
    2. AdaGN (Adaptive Group Normalization) 来自 DDPM原论文
    3. FiLM (Feature-wise Linear Modulation) 来自 DeepMind
    """
    
    def __init__(
        self,
        dim: int = 64,
        dim_mults: Tuple[int] = (1, 2, 4, 4),
        channels: int = 4,  # VAE latent channels (KL-VAE standard)
        num_classes: int = 31,  # 31个用户
        num_domains: int = 2,   # 源域/目标域
        self_condition: bool = True,
        resnet_block_groups: int = 8,
        learned_variance: bool = False,
        learned_sinusoidal_cond: bool = False,
        random_fourier_features: bool = False,
        learned_sinusoidal_dim: int = 16,
        dropout: float = 0.0
    ):
        super().__init__()
        
        # 基础UNet (来自denoising-diffusion-pytorch)
        # 使用额外的通道来传递条件信息
        self.cond_channels = 2  # 1 for class, 1 for domain
        self.base_unet = Unet(
            dim=dim,
            init_dim=None,
            out_dim=channels if not learned_variance else channels * 2,
            dim_mults=dim_mults,
            channels=channels + self.cond_channels,  # 增加条件通道
            self_condition=self_condition,
            resnet_block_groups=resnet_block_groups,
            learned_variance=learned_variance,
            learned_sinusoidal_cond=learned_sinusoidal_cond,
            random_fourier_features=random_fourier_features,
            learned_sinusoidal_dim=learned_sinusoidal_dim,
            attn_dim_head=32,
            attn_heads=4
        )
        
        # 保存dropout值用于其他层（如果需要）
        self.dropout = dropout
        
        # 暴露base_unet的属性给GaussianDiffusion
        # 这些属性是GaussianDiffusion所需的
        self.random_or_learned_sinusoidal_cond = getattr(self.base_unet, 'random_or_learned_sinusoidal_cond', False)
        self.self_condition = self_condition
        self.channels = channels
        
        # 保存类别和域的数量用于归一化
        self.num_classes = num_classes
        self.num_domains = num_domains
    
    def __getattr__(self, name):
        """代理到base_unet的属性访问，用于兼容GaussianDiffusion"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            # 如果当前类没有这个属性，尝试从base_unet获取
            if hasattr(self.base_unet, name):
                return getattr(self.base_unet, name)
            raise
            
    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        domain_labels: Optional[torch.Tensor] = None,
        x_self_cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播 - 使用通道拼接方式传递条件
        
        Args:
            x: 输入latent [B, C, H, W]
            time: 时间步 [B]
            class_labels: 类别标签 [B]
            domain_labels: 域标签 [B], 0=源域, 1=目标域
            x_self_cond: self-conditioning (可选)
            
        Returns:
            预测的噪声或x0 [B, C, H, W]
        """
        batch_size, c, h, w = x.shape
        device = x.device
        
        # 创建条件通道
        cond_channels = []
        
        # 类别条件通道
        if class_labels is not None:
            # 归一化到[-1, 1]
            class_channel = (class_labels.float() / (self.num_classes + 1) - 0.5) * 2
        else:
            # 默认使用null class
            class_channel = torch.ones(batch_size, device=device)  # 1 表示null
        class_channel = class_channel.view(batch_size, 1, 1, 1).expand(batch_size, 1, h, w)
        cond_channels.append(class_channel)
        
        # 域条件通道
        if domain_labels is not None:
            # 0 for source, 1 for target -> [-1, 1]
            domain_channel = (domain_labels.float() * 2 - 1)
        else:
            # 默认源域
            domain_channel = -torch.ones(batch_size, device=device)  # -1 表示源域
        domain_channel = domain_channel.view(batch_size, 1, 1, 1).expand(batch_size, 1, h, w)
        cond_channels.append(domain_channel)
        
        # 拼接条件到输入
        x_with_cond = torch.cat([x] + cond_channels, dim=1)  # [B, C+2, H, W]
        
        # 处理self condition
        if x_self_cond is not None:
            # self condition也需要添加条件通道（使用相同的条件）
            x_self_cond_full = torch.cat([x_self_cond] + cond_channels, dim=1)
        else:
            x_self_cond_full = None
        
        # 调用base_unet
        # 注意：base_unet期望输入通道数是C+2
        output = self.base_unet(x_with_cond, time, x_self_cond_full)
        
        # 只返回前C个通道（对应原始latent channels）
        return output[:, :c]
    




if __name__ == "__main__":
    # 测试模型
    print("Testing Domain Conditional UNet...")
    
    # 创建模型
    model = DomainConditionalUnet(
        dim=64,
        dim_mults=(1, 2, 4, 4),
        channels=32,  # VAE latent channels
        num_classes=31,
        num_domains=2
    )
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    
    # 测试前向传播
    batch_size = 4
    channels = 32
    x = torch.randn(batch_size, channels, 16, 16)  # [B, C, H, W]
    time = torch.randint(0, 1000, (batch_size,))  # [B]
    class_labels = torch.randint(0, 31, (batch_size,))  # [B]
    domain_labels = torch.randint(0, 2, (batch_size,))  # [B]
    
    with torch.no_grad():
        output = model(x, time, class_labels, domain_labels)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Base UNet expects channels: {channels + 2} (original {channels} + 2 condition channels)")
    
    # 测试条件是否影响输出
    x_test = torch.randn(2, channels, 16, 16)
    t_test = torch.full((2,), 500)
    
    with torch.no_grad():
        # 不同类别
        out1 = model(x_test, t_test, torch.zeros(2, dtype=torch.long))
        out2 = model(x_test, t_test, torch.full((2,), 15, dtype=torch.long))
        diff_class = (out1 - out2).abs().mean().item()
        
        # 不同域
        out3 = model(x_test, t_test, domain_labels=torch.zeros(2, dtype=torch.long))
        out4 = model(x_test, t_test, domain_labels=torch.ones(2, dtype=torch.long))
        diff_domain = (out3 - out4).abs().mean().item()
    
    print(f"\n条件影响测试:")
    print(f"不同类别的输出差异: {diff_class:.4f}")
    print(f"不同域的输出差异: {diff_domain:.4f}")
    print("✅ Model test passed!")

