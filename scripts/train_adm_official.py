"""
基于OpenAI官方guided-diffusion的ADM训练脚本
适配VAE latent space训练
完全遵循官方架构，无任何hack
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import numpy as np
from tqdm import tqdm
import time
from datetime import datetime
import math
from typing import Dict, Optional, Tuple, List

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils import (
    set_seed,
    count_parameters,
    AverageMeter,
    get_device,
    load_config,
    save_config,
)


# ================== 第一部分：基础组件（来自guided-diffusion） ==================

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    创建正弦时间步嵌入
    来自：guided_diffusion/nn.py L103-121
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class GroupNorm32(nn.GroupNorm):
    """
    32位浮点GroupNorm
    来自：guided_diffusion/nn.py L17-19
    """
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def normalization(channels):
    """
    创建GroupNorm层
    来自：guided_diffusion/nn.py L93-100
    """
    return GroupNorm32(32, channels)


def zero_module(module):
    """
    将模块参数初始化为零
    来自：guided_diffusion/nn.py L25-31
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class TimestepBlock(nn.Module):
    """
    任何接受时间步嵌入作为第二个参数的模块
    来自：guided_diffusion/unet.py L54-64
    """
    def forward(self, x, emb):
        raise NotImplementedError


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    顺序模块，将时间步嵌入传递给子模块
    来自：guided_diffusion/unet.py L66-78
    """
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


# ================== 第二部分：Attention Block ==================

class AttentionBlock(nn.Module):
    """
    注意力块，允许空间位置相互关注
    基于guided_diffusion/unet.py L259-305
    """
    def __init__(self, channels, num_heads=1, num_head_channels=-1):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0
            self.num_heads = channels // num_head_channels
        
        self.norm = normalization(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)
        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))
    
    def forward(self, x):
        b, c, *spatial = x.shape
        x_orig = x
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x_orig.reshape(b, c, -1) + h).reshape(b, c, *spatial)


class QKVAttention(nn.Module):
    """
    QKV注意力模块
    基于guided_diffusion/unet.py L328-358
    """
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads
    
    def forward(self, qkv):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)


# ================== 第三部分：ResBlock with Scale-Shift Norm ==================

class ResBlock(TimestepBlock):
    """
    残差块，支持scale-shift norm和上下采样
    基于guided_diffusion/unet.py L154-256
    """
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_scale_shift_norm=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_scale_shift_norm = use_scale_shift_norm
        self.up = up
        self.down = down
        
        # 处理上下采样
        if up:
            self.h_upd = nn.Upsample(scale_factor=2, mode='nearest')
            self.x_upd = nn.Upsample(scale_factor=2, mode='nearest')
        elif down:
            self.h_upd = nn.AvgPool2d(kernel_size=2, stride=2)
            self.x_upd = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            self.h_upd = self.x_upd = nn.Identity()
        
        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )
        
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
            ),
        )
        
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)
    
    def forward(self, x, emb):
        # 应用上下采样到输入
        if self.up or self.down:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        
        if self.use_scale_shift_norm:
            # Scale-shift norm: ADM的核心创新
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        
        return self.skip_connection(x) + h


# ================== 第三部分：简化的ADM UNet ==================

class SimplifiedADMUNet(nn.Module):
    """
    ADM UNet，适配VAE latent space
    基于guided_diffusion/unet.py的UNetModel
    支持FP16和ResBlock采样
    """
    def __init__(
        self,
        in_channels=4,  # VAE latent channels
        model_channels=128,
        out_channels=4,
        num_res_blocks=2,
        channel_mult=(1, 2, 3, 3),
        dropout=0.1,
        num_classes=32,  # 31 users + 1 null
        use_scale_shift_norm=True,
        use_fp16=False,
        resblock_updown=True,  # 使用ResBlock进行上下采样
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.channel_mult = channel_mult
        self.num_classes = num_classes
        self.use_scale_shift_norm = use_scale_shift_norm
        self.use_fp16 = use_fp16
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.resblock_updown = resblock_updown
        
        # 时间嵌入
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # 类别嵌入（官方方法）
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)
        
        # 输入卷积
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                nn.Conv2d(in_channels, model_channels, 3, padding=1)
            )
        ])
        
        # 下采样块
        ch = model_channels
        input_block_chans = [ch]
        ds = 1
        
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                
                # 在特定分辨率添加注意力块（模仿官方配置）
                if level >= len(channel_mult) - 2:  # 最后两层添加注意力
                    layers.append(AttentionBlock(ch, num_heads=8))
                
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            
            if level != len(channel_mult) - 1:
                # 下采样
                if self.resblock_updown:
                    # 使用ResBlock进行下采样
                    self.input_blocks.append(
                        TimestepEmbedSequential(
                            ResBlock(
                                ch,
                                time_embed_dim,
                                dropout,
                                out_channels=ch,
                                use_scale_shift_norm=use_scale_shift_norm,
                                down=True,
                            )
                        )
                    )
                else:
                    # 简单卷积下采样
                    self.input_blocks.append(
                        TimestepEmbedSequential(
                            nn.Conv2d(ch, ch, 3, stride=2, padding=1)
                        )
                    )
                input_block_chans.append(ch)
                ds *= 2
        
        # 中间块（包含注意力）
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(ch, num_heads=8),  # 中间块总是有注意力
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        
        # 上采样块
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                
                # 在对应位置添加注意力块
                if level >= len(channel_mult) - 2 and i < num_res_blocks:
                    layers.append(AttentionBlock(ch, num_heads=8))
                
                if level and i == num_res_blocks:
                    # 上采样
                    if self.resblock_updown:
                        # 使用ResBlock进行上采样
                        layers.insert(
                            0,
                            ResBlock(
                                ch,
                                time_embed_dim,
                                dropout,
                                out_channels=ch,
                                use_scale_shift_norm=use_scale_shift_norm,
                                up=True,
                            )
                        )
                    else:
                        # 简单转置卷积上采样
                        layers.append(
                            nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)
                        )
                    ds //= 2
                
                self.output_blocks.append(TimestepEmbedSequential(*layers))
        
        # 输出卷积
        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(ch, out_channels, 3, padding=1)),
        )
    
    def forward(self, x, timesteps, y=None):
        """
        前向传播（支持FP16）
        x: [B, C, H, W] VAE latents
        timesteps: [B] 时间步
        y: [B] 类别标签（可选）
        """
        assert (y is not None) == (self.num_classes is not None)
        
        # 时间嵌入
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        
        # 类别嵌入（官方方法：直接相加）
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        
        # UNet前向传播
        hs = []
        h = x.type(self.dtype)  # 转换到目标精度
        
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        
        h = self.middle_block(h, emb)
        
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        
        h = h.type(x.dtype)
        return self.out(h)


# ================== 第四部分：VAE集成 ==================

class VAEWrapper:
    """
    VAE包装器，处理编码和解码
    """
    def __init__(self, vae_checkpoint_path, device='cuda'):
        # 导入已训练的KL-VAE
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent.parent))
        from domain_adaptive_diffusion.vae.kl_vae import KL_VAE
        
        self.device = device
        
        # 加载KL-VAE（使用默认配置）
        self.vae = KL_VAE(
            ddconfig=None,  # 使用默认配置
            embed_dim=4,    # 4通道latent
            scale_factor=0.18215  # 标准scale factor
        )
        
        # 加载权重
        checkpoint = torch.load(vae_checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            self.vae.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.vae.load_state_dict(checkpoint)
        
        self.vae.eval()
        
        # 移动到正确的设备
        self.vae = self.vae.to(device)
        
        # 冻结VAE
        for param in self.vae.parameters():
            param.requires_grad = False
        
        # 不需要手动管理scale_factor，KL_VAE内置了
    
    def encode(self, images):
        """编码图像到latent space"""
        with torch.no_grad():
            # 使用KL_VAE内置的encode_images方法（自动处理scale factor）
            latents = self.vae.encode_images(images)
        return latents
    
    def decode(self, latents):
        """解码latents到图像"""
        with torch.no_grad():
            # 使用KL_VAE内置的decode_latents方法（自动处理scale factor）
            images = self.vae.decode_latents(latents)
        return images


# ================== 第五部分：扩散过程 ==================

class GaussianDiffusion:
    """
    高斯扩散过程
    基于guided_diffusion/gaussian_diffusion.py
    """
    def __init__(
        self,
        timesteps=1000,
        beta_schedule="linear",
        beta_start=0.00085,
        beta_end=0.012,
    ):
        self.timesteps = timesteps
        
        # 创建beta schedule
        if beta_schedule == "linear":
            betas = np.linspace(beta_start, beta_end, timesteps)
        elif beta_schedule == "scaled_linear":
            betas = np.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # 预计算扩散参数
        self.betas = torch.tensor(betas, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0]),
            self.alphas_cumprod[:-1]
        ])
        
        # 计算扩散过程的参数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # 后验分布参数
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
    
    def q_sample(self, x_start, t, noise=None):
        """
        前向扩散过程：给定x_0和t，采样x_t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].to(x_start.device)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].to(x_start.device)
        
        # 扩展维度以匹配图像
        while len(sqrt_alphas_cumprod_t.shape) < len(x_start.shape):
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t[..., None]
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t[..., None]
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        计算训练损失
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # 前向扩散
        x_t = self.q_sample(x_start, t, noise=noise)
        
        # 模型预测
        model_output = model(x_t, t, **model_kwargs)
        
        # MSE损失（预测噪声）
        loss = F.mse_loss(model_output, noise, reduction='none')
        loss = loss.mean(dim=list(range(1, len(loss.shape))))
        
        return {"loss": loss}
    
    @torch.no_grad()
    def p_sample(self, model, x, t, model_kwargs=None):
        """
        逆向采样一步
        """
        if model_kwargs is None:
            model_kwargs = {}
        
        # 模型预测
        model_output = model(x, t, **model_kwargs)
        
        # 计算均值
        pred_xstart = (
            self.sqrt_recip_alphas_cumprod[t].to(x.device) * x
            - self.sqrt_recipm1_alphas_cumprod[t].to(x.device) * model_output
        )
        pred_xstart = torch.clamp(pred_xstart, -1, 1)
        
        # 计算后验均值
        posterior_mean = (
            self.posterior_mean_coef1[t].to(x.device) * pred_xstart
            + self.posterior_mean_coef2[t].to(x.device) * x
        )
        
        # 采样
        noise = torch.randn_like(x) if t[0] > 0 else 0
        posterior_variance = self.posterior_variance[t].to(x.device)
        while len(posterior_variance.shape) < len(x.shape):
            posterior_variance = posterior_variance[..., None]
        
        return posterior_mean + torch.sqrt(posterior_variance) * noise
    
    @torch.no_grad()
    def sample(self, model, shape, model_kwargs=None):
        """
        完整的采样过程
        """
        if model_kwargs is None:
            model_kwargs = {}
        
        device = next(model.parameters()).device
        
        # 从噪声开始
        x = torch.randn(shape, device=device)
        
        # 逐步去噪
        for t in tqdm(reversed(range(self.timesteps)), desc="Sampling"):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t_batch, model_kwargs)
        
        return x


# ================== 第六部分：训练器 ==================

class OfficialADMTrainer:
    """
    官方ADM训练器（支持FP16混合精度训练）
    """
    def __init__(self, config: Dict, device: str = 'cuda'):
        self.config = config
        self.device = device
        self.use_fp16 = config.get('use_fp16', True)
        
        # 设置实验目录
        self.exp_dir = Path(config['output_dir'])
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        save_config(config, self.exp_dir / 'config.yaml')
        
        # 初始化VAE
        print("初始化VAE...")
        self.vae = VAEWrapper(config['vae_checkpoint'], device=device)
        
        # 初始化模型（确保所有数值参数类型正确）
        print("初始化ADM模型...")
        self.model = SimplifiedADMUNet(
            in_channels=4,
            model_channels=int(config['model'].get('model_channels', 128)),
            out_channels=4,
            num_res_blocks=int(config['model'].get('num_res_blocks', 2)),
            channel_mult=tuple(config['model'].get('channel_mult', [1, 2, 3, 3])),
            dropout=float(config['model'].get('dropout', 0.1)),
            num_classes=int(config['model'].get('num_classes', 32)),
            use_scale_shift_norm=True,  # ADM核心
            use_fp16=self.use_fp16,  # 启用FP16
            resblock_updown=True,  # 使用ResBlock采样
        ).to(device)
        
        print(f"模型参数量: {count_parameters(self.model):,}")
        
        # 初始化扩散过程（确保数值类型正确）
        self.diffusion = GaussianDiffusion(
            timesteps=int(config['diffusion']['timesteps']),
            beta_schedule=config['diffusion']['beta_schedule'],
            beta_start=float(config['diffusion']['beta_start']),
            beta_end=float(config['diffusion']['beta_end']),
        )
        
        # 初始化优化器（确保类型正确）
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(config['training']['learning_rate']),
            weight_decay=float(config['training']['weight_decay']),
            betas=(0.9, 0.999),
        )
        
        # EMA
        self.ema_model = None
        if config.get('use_ema', True):
            self.ema_model = self._create_ema_model()
        
        # FP16 Scaler
        self.scaler = GradScaler() if self.use_fp16 else None
        
        # 设置数据加载器（使用现有的data_loader.py）
        from utils import create_dataloaders
        self.train_loader, self.val_loader = create_dataloaders(
            data_path=config['data']['latent_path'],
            phase='pretrain',  # 使用pretrain模式（不需要domain balance）
            batch_size=int(config['training']['batch_size']),
            num_workers=4,
            augmentation=False,  # 不使用数据增强
            device=device
        )
        
        self.global_step = 0
        
        print(f"✅ FP16混合精度: {'启用' if self.use_fp16 else '禁用'}")
        print(f"✅ ResBlock上下采样: 启用")
        print(f"✅ 数据集大小: 训练={len(self.train_loader.dataset)}, 验证={len(self.val_loader.dataset) if self.val_loader else 0}")
    
    def _create_ema_model(self):
        """创建EMA模型"""
        import copy
        ema_model = copy.deepcopy(self.model)
        ema_model.eval()
        for param in ema_model.parameters():
            param.requires_grad = False
        return ema_model
    
    def _update_ema(self, decay=0.9999):
        """更新EMA模型"""
        if self.ema_model is None:
            return
        
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)
    
    def train_step(self, batch):
        """训练一步（支持FP16）"""
        self.model.train()
        
        # 获取数据
        latents = batch['latent'].to(self.device)
        labels = batch['class_label'].to(self.device)
        batch_size = latents.shape[0]
        
        # 随机时间步
        t = torch.randint(0, self.diffusion.timesteps, (batch_size,), device=self.device)
        
        # 混合精度训练
        if self.use_fp16:
            with autocast():
                # 计算损失（自动转换到FP16）
                losses = self.diffusion.training_losses(
                    self.model,
                    latents,
                    t,
                    model_kwargs={"y": labels}
                )
                loss = losses["loss"].mean()
            
            # 反向传播（使用scaler）
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            # 梯度裁剪（需要先unscale）
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # 优化器步进
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # 标准训练（FP32）
            losses = self.diffusion.training_losses(
                self.model,
                latents,
                t,
                model_kwargs={"y": labels}
            )
            loss = losses["loss"].mean()
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        
        # 更新EMA
        self._update_ema()
        
        self.global_step += 1
        
        return loss.item()
    
    def train(self):
        """训练循环"""
        num_epochs = int(self.config['training'].get('num_epochs', self.config['training'].get('epochs', 100)))
        
        for epoch in range(num_epochs):
            # 训练
            train_loss = AverageMeter()
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch in pbar:
                loss = self.train_step(batch)
                train_loss.update(loss)
                pbar.set_postfix({'loss': f'{train_loss.avg:.4f}'})
            
            # 每个epoch生成条件图像
            print(f"\n📸 生成条件扩散图像...")
            self.generate_and_visualize(epoch)
            
            # 验证和保存
            if (epoch + 1) % 5 == 0:
                self.validate()
                self.save_checkpoint(epoch)
    
    @torch.no_grad()
    def validate(self):
        """验证生成质量"""
        self.model.eval()
        
        # 生成样本
        num_samples = 8
        shape = (num_samples, 4, 32, 32)  # VAE latent shape
        
        # 随机类别
        labels = torch.randint(0, 31, (num_samples,), device=self.device)
        
        # FP16推理
        with autocast(enabled=self.use_fp16):
            # 采样
            samples = self.diffusion.sample(
                self.model,
                shape,
                model_kwargs={"y": labels}
            )
        
        # 解码（VAE在FP32运行）
        images = self.vae.decode(samples.float())
        
        # 保存图像（这里省略可视化代码）
        print(f"生成样本统计: mean={images.mean():.3f}, std={images.std():.3f}")
    
    @torch.no_grad()
    def generate_and_visualize(self, epoch, num_samples=16):
        """
        生成并可视化条件扩散图像
        每个epoch结束时调用
        """
        self.model.eval()
        
        # 使用EMA模型（如果有）
        model = self.ema_model if self.ema_model is not None else self.model
        
        # 生成每个用户的样本
        samples_per_user = max(1, num_samples // 31)
        all_samples = []
        all_labels = []
        
        # 为前8个用户生成样本（可视化限制）
        num_users_to_show = min(8, 31)
        
        # FP16推理
        with autocast(enabled=self.use_fp16):
            for user_id in range(num_users_to_show):
                # 创建批次标签
                labels = torch.full((samples_per_user,), user_id, device=self.device)
                shape = (samples_per_user, 4, 32, 32)
                
                # 采样
                print(f"  生成用户{user_id}的样本...")
                samples = self.diffusion.sample(
                    model,
                    shape,
                    model_kwargs={"y": labels}
                )
                
                all_samples.append(samples)
                all_labels.append(labels)
        
        # 合并所有样本
        all_samples = torch.cat(all_samples, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # 解码到图像空间（VAE在FP32运行）
        print("  解码latents到图像...")
        images = self.vae.decode(all_samples.float())  # [N, 3, 256, 256]
        
        # 创建网格可视化
        self._save_image_grid(images, all_labels, epoch)
    
    def _save_image_grid(self, images, labels, epoch):
        """
        保存图像网格
        """
        import torchvision
        import matplotlib.pyplot as plt
        
        # 确保图像在[0, 1]范围
        images = torch.clamp((images + 1) / 2, 0, 1)
        
        # 创建图像网格
        n_images = min(16, images.shape[0])  # 最多显示16张
        grid = torchvision.utils.make_grid(
            images[:n_images], 
            nrow=4, 
            padding=2, 
            normalize=False
        )
        
        # 转换为numpy
        grid_np = grid.cpu().numpy().transpose(1, 2, 0)
        
        # 创建图像
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(grid_np)
        ax.axis('off')
        
        # 添加标题
        user_ids = labels[:n_images].cpu().numpy()
        title = f"Epoch {epoch} - Conditional Generation\n"
        title += f"Users: {user_ids[:4].tolist()} (row 1)\n"
        if n_images > 4:
            title += f"Users: {user_ids[4:8].tolist()} (row 2)\n"
        if n_images > 8:
            title += f"Users: {user_ids[8:12].tolist()} (row 3)\n"
        if n_images > 12:
            title += f"Users: {user_ids[12:16].tolist()} (row 4)"
        
        ax.set_title(title, fontsize=12)
        
        # 保存图像
        save_path = self.exp_dir / f'generation_epoch_{epoch:04d}.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ 条件生成图像已保存到: {save_path}")
        
        # 打印统计信息
        print(f"  图像统计: mean={images.mean():.3f}, std={images.std():.3f}")
        print(f"  范围: [{images.min():.3f}, {images.max():.3f}]")
    
    def save_checkpoint(self, epoch):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
        }
        if self.ema_model is not None:
            checkpoint['ema_state_dict'] = self.ema_model.state_dict()
        
        path = self.exp_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")


def main():
    parser = argparse.ArgumentParser(description='训练ADM条件扩散模型')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    
    # 可选参数，用于覆盖配置文件中的路径
    parser.add_argument('--data_dir', type=str, default=None, 
                       help='数据目录，覆盖配置文件中的latent_path')
    parser.add_argument('--vae_checkpoint', type=str, default=None,
                       help='VAE检查点路径，覆盖配置文件')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录，覆盖配置文件')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='批次大小，覆盖配置文件')
    parser.add_argument('--num_epochs', type=int, default=None,
                       help='训练轮数，覆盖配置文件')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 覆盖配置文件中的参数
    if args.data_dir:
        config['data']['latent_path'] = args.data_dir
        print(f"覆盖数据路径: {args.data_dir}")
    
    if args.vae_checkpoint:
        config['vae_checkpoint'] = args.vae_checkpoint
        print(f"覆盖VAE路径: {args.vae_checkpoint}")
    
    if args.output_dir:
        config['output_dir'] = args.output_dir
        print(f"覆盖输出目录: {args.output_dir}")
    
    if args.batch_size:
        config['training']['batch_size'] = int(args.batch_size)
        print(f"覆盖批次大小: {args.batch_size}")
    
    if args.num_epochs:
        config['training']['num_epochs'] = int(args.num_epochs)
        print(f"覆盖训练轮数: {args.num_epochs}")
    
    # 设置随机种子
    set_seed(int(config.get('seed', 42)))
    
    # 打印最终配置
    print("\n" + "="*60)
    print("最终训练配置:")
    print(f"  数据路径: {config['data']['latent_path']}")
    print(f"  VAE模型: {config['vae_checkpoint']}")
    print(f"  输出目录: {config['output_dir']}")
    print(f"  批次大小: {int(config['training']['batch_size'])}")
    print(f"  训练轮数: {int(config['training'].get('num_epochs', config['training'].get('epochs', 100)))}")
    print(f"  FP16: {'启用' if config.get('use_fp16', False) else '禁用'}")
    print("="*60 + "\n")
    
    # 创建训练器
    trainer = OfficialADMTrainer(config, args.device)
    
    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()
