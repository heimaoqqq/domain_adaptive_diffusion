"""
ADM风格的条件扩散模型训练
基于OpenAI guided-diffusion的scale-shift norm机制
为Universal Guidance优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import argparse
import os
import sys
from pathlib import Path
from tqdm import tqdm
import time
from datetime import datetime
from typing import Dict, Optional, Tuple, List
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

# HuggingFace Diffusers
from diffusers import UNet2DConditionModel, DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from transformers import CLIPTextModel, CLIPTokenizer

# 复用现有的工具函数
from utils import (
    create_dataloaders,
    set_seed,
    count_parameters,
    EMA,
    AverageMeter,
    get_device,
    load_config,
    save_config,
    create_exp_dir
)

# VAE将在_init_vae中动态导入


# ADM风格的条件注入模块
class ClassEmbedder(nn.Module):
    """
    类别嵌入器 - 基于OpenAI ADM的官方实现
    简洁高效，适合小数据集
    """
    def __init__(self, num_classes, embed_dim=512, use_cfg=False):
        super().__init__()
        self.num_classes = num_classes
        self.use_cfg = use_cfg
        
        # 类别嵌入表
        self.embedding = nn.Embedding(
            num_classes + (1 if use_cfg else 0),  # +1 for null class if using CFG
            embed_dim
        )
        
        # 简单的投影层（ADM官方设计）
        self.linear = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # 初始化
        nn.init.normal_(self.embedding.weight, std=0.02)
        if use_cfg:
            # null class初始化为0
            self.embedding.weight.data[-1] = 0
    
    def forward(self, class_labels, drop_prob=0.0):
        """
        前向传播
        Args:
            class_labels: [B] 类别标签
            drop_prob: CFG的dropout概率（训练时使用）
        """
        # CFG: 训练时随机替换为null class
        if self.use_cfg and self.training and drop_prob > 0:
            batch_size = class_labels.shape[0]
            drop_mask = torch.rand(batch_size, device=class_labels.device) < drop_prob
            class_labels = torch.where(
                drop_mask,
                torch.full_like(class_labels, self.num_classes),  # null class index
                class_labels
            )
        
        # 获取嵌入
        x = self.embedding(class_labels)
        
        # 通过简单投影
        x = self.linear(x)
        
        return x


class ADMDiffusionTrainer:
    """
    ADM风格的条件扩散模型训练器
    使用scale-shift norm进行条件注入，为UG优化
    """
    def __init__(self, config: Dict, device: str = 'cuda'):
        self.config = config
        self.device = device
        
        # 创建实验目录（不使用时间戳）
        output_dir = config.get('output_dir', './experiments/diffusers_baseline')
        self.exp_dir = Path(output_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存配置
        save_config(config, self.exp_dir / 'config.yaml')
        print(f"✅ 配置已保存到: {self.exp_dir / 'config.yaml'}")
        
        # 初始化调试记录器
        self.debug_metrics = {
            'train_losses': [],      # 每步的损失
            'val_losses': [],        # 验证损失
            'grad_norms': [],        # 梯度范数
            'noise_pred_stats': [],  # 噪声预测统计
            'condition_response': [], # 条件响应强度
            'class_diversity': [],   # 类别多样性
            'timestep_losses': {},   # 按时间步的损失
        }
        
        # 创建日志文件
        self.log_file = self.exp_dir / 'debug_log.txt'
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"Training Debug Log - {datetime.now()}\n")
            f.write("="*60 + "\n")
        
        # 初始化组件
        self._init_vae()
        self._init_model()
        self._init_optimizer()
        self._init_ema()
        
        print(f"✅ 初始化完成，实验目录: {self.exp_dir}")
    
    def _log_debug(self, message):
        """写入调试日志"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + "\n")
    
    def _init_vae(self):
        """加载预训练的VAE（用于可视化）"""
        vae_checkpoint = self.config.get('vae_checkpoint', None)
        if vae_checkpoint and Path(vae_checkpoint).exists():
            # 导入正确的KL_VAE类
            from domain_adaptive_diffusion.vae.kl_vae import KL_VAE as DDPM_KL_VAE
            
            # 创建VAE实例 - KL_VAE使用默认配置即可
            self.vae = DDPM_KL_VAE(
                embed_dim=4,  # latent通道数
                scale_factor=0.18215  # 标准scale factor
            )
            
            # 加载权重
            checkpoint = torch.load(vae_checkpoint, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                # 从完整的checkpoint中提取模型权重
                self.vae.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                self.vae.load_state_dict(checkpoint['state_dict'])
            else:
                self.vae.load_state_dict(checkpoint)
            
            self.vae.to(self.device)
            self.vae.eval()
            print(f"✅ KL-VAE 加载成功: {vae_checkpoint}")
        else:
            self.vae = None
            print("⚠️ 未找到VAE，将无法可视化生成结果")
    
    @torch.no_grad()
    def _check_vae_ranges(self):
        """检测VAE的输入输出范围，确保归一化正确"""
        if self.vae is None:
            return
            
        print("\n🔍 检测VAE输入输出范围...")
        
        # 从训练数据中采样一些真实的latents
        sample_batch = next(iter(self.train_loader))
        real_latents = sample_batch['latent'][:4].to(self.device)
        
        print(f"训练数据中的latent范围:")
        print(f"  - Mean: {real_latents.mean().item():.4f}")
        print(f"  - Std: {real_latents.std().item():.4f}")
        print(f"  - Min: {real_latents.min().item():.4f}")
        print(f"  - Max: {real_latents.max().item():.4f}")
        
        # 测试解码
        decoded_images = self.vae.decode_latents(real_latents)
        print(f"\n解码后的图像范围:")
        print(f"  - Min: {decoded_images.min().item():.4f}")
        print(f"  - Max: {decoded_images.max().item():.4f}")
        
        # 测试采样的latents范围（应该与真实latents相似）
        noise = torch.randn(4, 4, self.config['data']['latent_size'], 
                          self.config['data']['latent_size'], device=self.device)
        print(f"\n随机噪声范围:")
        print(f"  - Mean: {noise.mean().item():.4f}")
        print(f"  - Std: {noise.std().item():.4f}")
        
        # 计算真实的scale factor
        print(f"\nVAE scale_factor: {self.vae.scale_factor}")
        
        # 验证完整的编码-解码流程
        print("\n🔄 测试完整的编码-解码流程...")
        # 创建一个假的图像（在[0,1]范围）
        fake_image = torch.rand(1, 3, 256, 256, device=self.device)
        print(f"输入图像范围: [{fake_image.min():.4f}, {fake_image.max():.4f}]")
        
        # 编码
        encoded = self.vae.encode_images(fake_image)
        print(f"编码后的latent范围（包含scale_factor）: [{encoded.min():.4f}, {encoded.max():.4f}]")
        
        # 解码
        decoded = self.vae.decode_latents(encoded)
        print(f"解码后的图像范围: [{decoded.min():.4f}, {decoded.max():.4f}]")
        
        print("=" * 50)
    
    def _init_model(self):
        """初始化ADM风格的UNet和调度器"""
        model_config = self.config.get('model', {})
        
        # 使用UNet2DModel而非UNet2DConditionModel，因为我们要自己处理条件
        from diffusers import UNet2DModel
        
        # 创建ADM-Small UNet - 适合小数据集
        # 基于OpenAI guided-diffusion的小模型配置
        self.unet = UNet2DModel(
            sample_size=self.config['data']['latent_size'],
            in_channels=model_config.get('in_channels', 4),
            out_channels=model_config.get('out_channels', 4),
            layers_per_block=model_config.get('layers_per_block', 2),  # 减少层数
            block_out_channels=tuple(model_config.get('block_out_channels', 
                                    [128, 256, 384, 384])),  # ADM-Small通道配置
            down_block_types=[
                "DownBlock2D",        # 第1层：纯卷积
                "DownBlock2D",        # 第2层：纯卷积  
                "AttnDownBlock2D",    # 第3层：带注意力
                "DownBlock2D"         # 第4层：纯卷积
            ],
            up_block_types=[
                "UpBlock2D",          # 第4层：纯卷积
                "AttnUpBlock2D",      # 第3层：带注意力
                "UpBlock2D",          # 第2层：纯卷积
                "UpBlock2D"           # 第1层：纯卷积
            ],
            attention_head_dim=model_config.get('attention_head_dim', 8),  # 小注意力头
            dropout=model_config.get('dropout', 0.1),
            norm_num_groups=model_config.get('norm_num_groups', 32),
            norm_eps=float(model_config.get('norm_eps', 1e-6)),
            act_fn=model_config.get('act_fn', 'silu'),
            resnet_time_scale_shift="scale_shift",  # ADM核心：scale-shift norm
        ).to(self.device)
        
        # 获取时间嵌入维度
        time_embed_dim = self.unet.time_embedding.linear_1.out_features
        
        # 创建类别嵌入器（官方简洁版）
        self.class_embedder = ClassEmbedder(
            num_classes=model_config.get('num_class_embeds', 32) - 1,  # 31个用户
            embed_dim=model_config.get('class_embed_dim', 512),  # 标准嵌入维度
            use_cfg=False  # 小数据集暂不用CFG
        ).to(self.device)
        
        # 简化CFG配置（小数据集不需要）
        self.cfg_scale = 1.0  # 不使用CFG
        self.cfg_dropout = 0.0
        self.use_cfg = False
        
        # 条件组合器 - ADM官方设计（简单高效）
        # 参考：OpenAI guided-diffusion
        self.cond_combiner = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim + 512, time_embed_dim)
        ).to(self.device)
        
        # 打印模型信息
        unet_params = count_parameters(self.unet)
        class_params = count_parameters(self.class_embedder)
        cond_params = count_parameters(self.cond_combiner)
        total_params = unet_params + class_params + cond_params
        
        print(f"✅ ADM模型初始化完成:")
        print(f"   - UNet参数量: {unet_params:,}")
        print(f"   - 类别嵌入器参数量: {class_params:,}")
        print(f"   - 条件组合器参数量: {cond_params:,}")
        print(f"   - 总参数量: {total_params:,}")
        print(f"   - 使用scale-shift norm: True")
        
        # 测试前向传播
        with torch.no_grad():
            test_input = torch.randn(1, 4, 32, 32).to(self.device)
            test_timestep = torch.tensor([500]).to(self.device)
            test_label = torch.tensor([0]).to(self.device)
            
            # 获取时间嵌入
            t_emb = self.unet.time_embedding(test_timestep)
            # 获取类别嵌入
            c_emb = self.class_embedder(test_label)
            # 组合条件
            combined_emb = self.cond_combiner(torch.cat([t_emb, c_emb], dim=1))
            
            # 注意：我们需要修改UNet的前向传播来使用组合的嵌入
            # 暂时使用标准前向传播测试
            test_output = self.unet(test_input, test_timestep, return_dict=False)[0]
            print(f"   - 初始化测试: 输出std={test_output.std().item():.4f}")
        
        
        # 创建噪声调度器
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            variance_type="fixed_small",
            clip_sample=True,  # 启用clip以稳定训练
            prediction_type="epsilon",  # 预测噪声
        )
        
        # 用于推理的调度器（可选DDIM加速）
        self.inference_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            clip_sample=True,  # 启用clip以防止数值爆炸
            set_alpha_to_one=False,
            steps_offset=1,
            prediction_type="epsilon",
        )
        
    def _init_optimizer(self):
        """初始化优化器和学习率调度器"""
        train_config = self.config.get('training', {}).get('pretrain', {})
        
        # 合并所有参数 - 包括类别嵌入器和条件组合器
        all_params = list(self.unet.parameters()) + \
                    list(self.class_embedder.parameters()) + \
                    list(self.cond_combiner.parameters())
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            all_params,
            lr=float(train_config.get('learning_rate', 1e-4)),
            betas=(0.9, 0.999),
            weight_decay=float(train_config.get('weight_decay', 0.01)),
            eps=1e-8
        )
        
        # 学习率调度器（使用cosine）
        num_epochs = train_config.get('epochs', 50)
        batch_size = train_config.get('batch_size', 8)
        # 假设每个epoch大约有4000个样本
        num_training_steps = num_epochs * (4000 // batch_size)
        num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup
        
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
    def _init_ema(self):
        """初始化EMA"""
        ema_config = self.config.get('ema', {})
        if ema_config.get('use_ema', True):
            self.ema = EMA(
                self.unet,
                decay=ema_config.get('beta', 0.9999)  # EMA使用decay参数，不是beta
            )
            print("✅ EMA初始化完成")
        else:
            self.ema = None
    
    def _forward_with_adm_condition(self, x, timesteps, class_labels):
        """
        ADM风格的条件前向传播
        简洁高效，基于官方实现
        """
        # 获取时间嵌入
        t_emb = self.unet.time_embedding(timesteps)
        
        # 获取类别嵌入（不使用CFG dropout）
        c_emb = self.class_embedder(class_labels, drop_prob=0.0)
        
        # 组合条件嵌入 - 使用增强的组合器
        combined_emb = self.cond_combiner(torch.cat([t_emb, c_emb], dim=1))
        
        # 临时替换time_embedding以返回组合的嵌入
        original_time_embedding = self.unet.time_embedding
        
        class CombinedEmbedding(nn.Module):
            def __init__(self, combined_emb):
                super().__init__()
                self.combined_emb = combined_emb
            
            def forward(self, t):
                return self.combined_emb
        
        self.unet.time_embedding = CombinedEmbedding(combined_emb)
        
        # 正常的UNet前向传播
        output = self.unet(x, timesteps, return_dict=False)[0]
        
        # 恢复原始的time_embedding
        self.unet.time_embedding = original_time_embedding
        
        return output
            
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """单步训练"""
        self.unet.train()
        
        # 获取数据
        latents = batch['latent'].to(self.device)  # [B, 4, H, W]
        labels = batch['class_label'].to(self.device)    # [B]
        batch_size = latents.shape[0]
        
        # 重要：数据预处理
        # prepare_microdoppler_data.py已经将latents乘以了scale_factor (0.18215)
        # 但是Diffusers期望数据在标准范围内（std约1）
        # 解决方案：在训练时先除以scale_factor，让数据回到标准范围
        
        # 首次训练时打印调试信息
        if not hasattr(self, '_debug_printed'):
            print(f"\n📊 训练数据调试信息:")
            print(f"  - 原始Latent shape: {latents.shape}")
            print(f"  - 原始Latent mean: {latents.mean().item():.4f}")
            print(f"  - 原始Latent std: {latents.std().item():.4f}")
            
            # 除以scale_factor让数据回到标准范围
            if hasattr(self.vae, 'scale_factor'):
                latents_normalized = latents / self.vae.scale_factor
                print(f"  - 归一化后mean: {latents_normalized.mean().item():.4f}")
                print(f"  - 归一化后std: {latents_normalized.std().item():.4f}")
            
            self._debug_printed = True
        
        # 重要：除以scale_factor让数据回到标准范围
        if hasattr(self.vae, 'scale_factor'):
            latents = latents / self.vae.scale_factor
        
        # 采样随机时间步
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=self.device
        ).long()
        
        # 添加噪声
        noise = torch.randn_like(latents)
        
        # 调试：检查噪声是否正确缩放
        if not hasattr(self, '_noise_debug_printed'):
            print(f"\n🔊 噪声调试信息:")
            print(f"  - 原始latents std: {latents.std().item():.4f}")
            print(f"  - 噪声std: {noise.std().item():.4f}")
            self._noise_debug_printed = True
        
        noisy_latents = self.noise_scheduler.add_noise(
            latents, noise, timesteps
        )
        
        # 再次检查加噪后的范围
        if not hasattr(self, '_noisy_debug_printed'):
            print(f"  - 加噪后std: {noisy_latents.std().item():.4f}")
            print(f"  - 加噪后range: [{noisy_latents.min().item():.4f}, {noisy_latents.max().item():.4f}]")
            self._noisy_debug_printed = True
        
        # 前向传播 - ADM风格的条件注入
        with autocast(enabled=self.config.get('use_amp', True)):
            model_pred = self._forward_with_adm_condition(noisy_latents, timesteps, labels)
            
            # 计算损失（预测噪声）
            loss = F.mse_loss(model_pred, noise)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 计算梯度范数（调试用）
        grad_norm = 0.0
        for model in [self.unet, self.class_embedder, self.cond_combiner]:
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.norm().item() ** 2
        grad_norm = grad_norm ** 0.5
        
        # 梯度裁剪
        if self.config.get('gradient_clip', 1.0) > 0:
            torch.nn.utils.clip_grad_norm_(
                self.unet.parameters(), 
                self.config.get('gradient_clip', 1.0)
            )
        
        self.optimizer.step()
        self.lr_scheduler.step()
        
        # 记录调试信息
        self.debug_metrics['train_losses'].append(loss.item())
        self.debug_metrics['grad_norms'].append(grad_norm)
        
        # 记录噪声预测统计
        with torch.no_grad():
            noise_pred_mean = model_pred.mean().item()
            noise_pred_std = model_pred.std().item()
            self.debug_metrics['noise_pred_stats'].append({
                'mean': noise_pred_mean,
                'std': noise_pred_std,
                'min': model_pred.min().item(),
                'max': model_pred.max().item()
            })
            
            # 按时间步记录损失
            for t in timesteps.cpu().numpy():
                if t not in self.debug_metrics['timestep_losses']:
                    self.debug_metrics['timestep_losses'][t] = []
                self.debug_metrics['timestep_losses'][t].append(loss.item())
        
        # 调试：偶尔检查条件是否影响输出
        if not hasattr(self, '_train_condition_checked'):
            self._train_condition_checked = 0
        self._train_condition_checked += 1
        
        if self._train_condition_checked % 500 == 0:  # 每500步检查一次
            cond_response = self._test_condition_response()
            self.debug_metrics['condition_response'].append(cond_response)
            self._log_debug(f"Step {self._train_condition_checked}: 条件响应={cond_response:.4f}")
        
        # 更新EMA
        if self.ema is not None:
            self.ema.update()
        
        return {
            'loss': loss.item(),
            'lr': self.optimizer.param_groups[0]['lr']
        }
    
    @torch.no_grad()
    def _test_condition_response(self):
        """测试条件机制的响应强度"""
        self.unet.eval()
        
        # 创建测试输入
        test_batch_size = 8
        test_latents = torch.randn(test_batch_size, 4, 32, 32, device=self.device)
        test_timesteps = torch.full((test_batch_size,), 500, device=self.device)
        
        # 测试不同类别
        different_labels = torch.arange(test_batch_size, device=self.device) % 31
        out_different = self._forward_with_adm_condition(
            test_latents,
            test_timesteps,
            different_labels
        )
        
        # 测试相同类别
        same_labels = torch.full((test_batch_size,), 5, device=self.device)
        out_same = self._forward_with_adm_condition(
            test_latents,
            test_timesteps,
            same_labels
        )
        
        # 计算差异
        diff_between_classes = 0
        for i in range(test_batch_size - 1):
            diff_between_classes += (out_different[i] - out_different[i+1]).abs().mean().item()
        diff_between_classes /= (test_batch_size - 1)
        
        diff_same_class = 0
        for i in range(test_batch_size - 1):
            diff_same_class += (out_same[i] - out_same[i+1]).abs().mean().item()
        diff_same_class /= (test_batch_size - 1)
        
        # 条件响应强度 = 不同类别差异 / 相同类别差异
        condition_response = diff_between_classes / (diff_same_class + 1e-6)
        
        self.unet.train()
        return condition_response
    
    @torch.no_grad()
    def sample(self, num_samples: int = 8, labels: Optional[torch.Tensor] = None, use_ddpm: bool = None) -> torch.Tensor:
        """生成样本"""
        self.unet.eval()
        
        # 如果没有指定标签，随机采样
        if labels is None:
            labels = torch.randint(0, 31, (num_samples,), device=self.device)
            
        # 自动决定使用哪种采样器
        if use_ddpm is None:
            # 如果模型训练步数少于10000步，使用DDPM以获得更稳定的结果
            use_ddpm = not hasattr(self, 'global_step') or self.global_step < 10000
            if not hasattr(self, '_sampling_method_printed'):
                print(f"  自动选择采样方法: {'DDPM（稳定）' if use_ddpm else 'DDIM（快速）'}")
                self._sampling_method_printed = True
        
        # 初始化随机噪声 - 确保尺寸与训练数据匹配
        latent_size = self.config['data']['latent_size']
        latents = torch.randn(
            num_samples, 4, 
            latent_size,
            latent_size,
            device=self.device
        )
        
        # 调试：验证生成的初始噪声范围
        if not hasattr(self, '_sample_debug_printed'):
            print(f"\n🎲 采样调试信息:")
            print(f"  - 初始噪声shape: {latents.shape}")
            print(f"  - 初始噪声mean: {latents.mean().item():.4f}")
            print(f"  - 初始噪声std: {latents.std().item():.4f}")
            self._sample_debug_printed = True
        
        # 选择并设置采样器
        if use_ddpm:
            # 使用DDPM采样器
            scheduler = self.noise_scheduler
            scheduler.set_timesteps(1000)
            # 只使用最后50步以加速（从纯噪声开始的最后50步）
            timesteps = scheduler.timesteps[-50:]
            desc = "Sampling (DDPM)"
        else:
            # 使用DDIM采样器
            scheduler = self.inference_scheduler
            scheduler.set_timesteps(50)
            timesteps = scheduler.timesteps
            desc = "Sampling (DDIM)"
        
        # 应用EMA权重
        if self.ema is not None:
            self.ema.apply_shadow()
        
        # 去噪过程
        for t in tqdm(timesteps, desc=desc):
            # 扩展t到batch维度
            timestep = t.expand(num_samples).to(self.device)
            
            # 预测噪声 - 使用ADM风格的条件注入
            noise_pred = self._forward_with_adm_condition(latents, timestep, labels)
            
            # 去噪一步
            latents = scheduler.step(
                noise_pred, t, latents, return_dict=False
            )[0]
            
            # 调试：检查中间步骤的范围（仅在使用DDIM时）
            if not use_ddpm and (t == timesteps[0] or t == timesteps[-1]):
                print(f"  Step {t}: latent range [{latents.min():.2f}, {latents.max():.2f}], std={latents.std():.2f}")
                # 同时检查预测的噪声
                print(f"    Noise pred: range [{noise_pred.min():.2f}, {noise_pred.max():.2f}], std={noise_pred.std():.2f}")
                
                # 额外调试：检查不同条件是否产生不同输出
                if t == timesteps[0] and not hasattr(self, '_condition_test_done'):
                    self._condition_test_done = True
                    with torch.no_grad():
                        # 测试相同输入，不同标签
                        test_labels = torch.tensor([0, 15], device=self.device)
                        test_noise1 = self._forward_with_adm_condition(
                            latents[:1].repeat(2, 1, 1, 1), 
                            timestep[:1].repeat(2),
                            test_labels
                        )
                        diff = (test_noise1[0] - test_noise1[1]).abs().mean()
                        print(f"    条件测试: 不同类别输出差异 = {diff:.6f}")
        
        # 恢复原始权重
        if self.ema is not None:
            self.ema.restore()
        
        # 重要：如果训练时除以了scale_factor，生成后需要乘回来
        if hasattr(self.vae, 'scale_factor'):
            latents = latents * self.vae.scale_factor
            print(f"  ✅ 乘以scale_factor后: std={latents.std().item():.4f}, range=[{latents.min():.4f}, {latents.max():.4f}]")
        
        return latents
    
    @torch.no_grad()
    def visualize_samples(self, epoch: int, num_samples: int = 8):
        """可视化生成的样本 - 生成少量代表性样本"""
        if self.vae is None:
            print("⚠️ 无VAE，跳过可视化")
            return
            
        # 选择几个代表性的类别进行生成
        selected_classes = [0, 5, 10, 15, 20, 25, 30][:num_samples]  # 均匀选择类别
        all_latents = []
        all_labels = []
        
        print(f"生成{len(selected_classes)}个类别的样本...")
        for class_id in selected_classes:
            labels = torch.full((1,), class_id, device=self.device)
            latents = self.sample(1, labels)
            all_latents.append(latents)
            all_labels.append(class_id)
        
        # 合并所有样本
        all_latents = torch.cat(all_latents, dim=0)[:num_samples]
        
        # 调试：检查生成的latents范围
        print(f"\n生成的latents统计（去噪后）:")
        print(f"  - Shape: {all_latents.shape}")
        print(f"  - Mean: {all_latents.mean().item():.4f}")
        print(f"  - Std: {all_latents.std().item():.4f}")
        print(f"  - Min: {all_latents.min().item():.4f}")
        print(f"  - Max: {all_latents.max().item():.4f}")
        
        # 与训练数据对比
        if hasattr(self, 'train_loader'):
            sample_batch = next(iter(self.train_loader))
            real_latents = sample_batch['latent'][:7].to(self.device)
            print(f"\n训练数据latents统计（对比）:")
            print(f"  - Mean: {real_latents.mean().item():.4f}")
            print(f"  - Std: {real_latents.std().item():.4f}")
            print(f"  - Min: {real_latents.min().item():.4f}")
            print(f"  - Max: {real_latents.max().item():.4f}")
        
        # 解码为图像
        images = self.vae.decode_latents(all_latents)
        
        # 调试：检查解码后的图像范围
        print(f"\n解码后的图像统计（clamp前）:")
        print(f"  - Min: {images.min().item():.4f}")
        print(f"  - Max: {images.max().item():.4f}")
        
        images = torch.clamp(images, 0.0, 1.0)
        
        # 转换为numpy
        images_np = images.cpu().permute(0, 2, 3, 1).numpy()
        
        # 创建网格图
        n_samples = len(all_labels)
        n_cols = min(4, n_samples)
        n_rows = (n_samples + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*3))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        elif n_cols == 1:
            axes = [[ax] for ax in axes]
        else:
            axes = axes.flatten()
        
        for i, (img, label) in enumerate(zip(images_np, all_labels)):
            if n_rows == 1 and n_cols == 1:
                ax = axes[0]
            else:
                ax = axes[i] if isinstance(axes, list) else axes.flatten()[i]
            ax.imshow(img)
            ax.set_title(f'User {label+1}', fontsize=12, fontweight='bold')
            ax.axis('off')
        
        # 隐藏多余的子图
        total_subplots = n_rows * n_cols
        for i in range(n_samples, total_subplots):
            if isinstance(axes, list):
                axes[i].axis('off')
            else:
                axes.flatten()[i].axis('off')
        
        plt.suptitle(f'Epoch {epoch} - Conditional Generated Samples', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.exp_dir / f'samples_epoch_{epoch}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 条件扩散样本已保存到: {save_path}")
    
    def train(self):
        """主训练循环"""
        train_config = self.config.get('training', {}).get('pretrain', {})
        num_epochs = train_config.get('epochs', 50)
        batch_size = train_config.get('batch_size', 8)
        
        # 创建数据加载器（只使用源域数据）
        data_path = Path(self.config['data']['latent_path'])
        train_loader, val_loader = create_dataloaders(
            data_path=data_path,
            batch_size=batch_size,
            phase='pretrain',
            num_workers=4
        )
        
        # 保存到实例变量，供其他方法使用
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        print(f"\n🚀 开始训练:")
        print(f"   - Epochs: {num_epochs}")
        print(f"   - Batch size: {batch_size}")
        print(f"   - 训练样本数: {len(train_loader.dataset)}")
        print(f"   - 验证样本数: {len(val_loader.dataset) if val_loader else 0}")
        
        # 在数据加载器创建后检测VAE范围
        if self.vae is not None:
            self._check_vae_ranges()
        
        # 在训练开始前生成一次样本作为基准
        print("\n📸 生成初始样本作为基准...")
        self.visualize_samples(0)
        
        # 调试：测试DDPM vs DDIM的差异
        print("\n🔍 测试DDPM vs DDIM采样差异...")
        self._test_sampling_difference()
        
        # 训练循环
        for epoch in range(num_epochs):
            # 训练阶段
            train_losses = []
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
            
            for batch in pbar:
                metrics = self.train_step(batch)
                train_losses.append(metrics['loss'])
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'lr': f"{metrics['lr']:.2e}"
                })
            
            # 计算平均损失
            avg_train_loss = np.mean(train_losses)
            
            # 记录epoch级别的调试信息
            self._log_debug(f"\n{'='*60}")
            self._log_debug(f"Epoch {epoch+1}/{num_epochs} 完成")
            self._log_debug(f"  训练损失: {avg_train_loss:.6f}")
            
            # 计算并记录梯度范数统计
            if self.debug_metrics['grad_norms']:
                recent_grad_norms = self.debug_metrics['grad_norms'][-len(train_loader):]
                grad_norm_mean = np.mean(recent_grad_norms)
                grad_norm_std = np.std(recent_grad_norms)
                self._log_debug(f"  梯度范数: {grad_norm_mean:.4f} ± {grad_norm_std:.4f}")
            
            # 记录噪声预测统计
            if self.debug_metrics['noise_pred_stats']:
                recent_stats = self.debug_metrics['noise_pred_stats'][-100:]
                pred_mean = np.mean([s['mean'] for s in recent_stats])
                pred_std = np.mean([s['std'] for s in recent_stats])
                self._log_debug(f"  噪声预测: mean={pred_mean:.4f}, std={pred_std:.4f}")
            
            # 测试条件响应
            condition_response = self._test_condition_response()
            self.debug_metrics['condition_response'].append(condition_response)
            self._log_debug(f"  条件响应强度: {condition_response:.4f}")
            
            if condition_response < 1.2:
                self._log_debug("    ⚠️ 条件响应较弱，可能还在学习中")
            elif condition_response < 2.0:
                self._log_debug("    🟡 条件响应中等，开始生效")
            else:
                self._log_debug("    ✅ 条件响应强，条件机制工作良好")
            
            # 验证阶段
            if val_loader is not None:
                val_losses = []
                self.unet.eval()
                
                with torch.no_grad():
                    for batch in val_loader:
                        latents = batch['latent'].to(self.device)
                        labels = batch['class_label'].to(self.device)
                        batch_size = latents.shape[0]
                        
                        # 随机时间步
                        timesteps = torch.randint(
                            0, self.noise_scheduler.config.num_train_timesteps,
                            (batch_size,), device=self.device
                        ).long()
                        
                        # 添加噪声
                        noise = torch.randn_like(latents)
                        noisy_latents = self.noise_scheduler.add_noise(
                            latents, noise, timesteps
                        )
                        
                        # 预测 - 只使用class_labels
                        dummy_encoder_states = torch.zeros(batch_size, 1, 1280, device=self.device)
                        model_pred = self.unet(
                            noisy_latents, timesteps, 
                            class_labels=labels,
                            encoder_hidden_states=dummy_encoder_states,
                            return_dict=False
                        )[0]
                        
                        loss = F.mse_loss(model_pred, noise)
                        val_losses.append(loss.item())
                
                avg_val_loss = np.mean(val_losses)
            else:
                avg_val_loss = 0.0
            
            print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss: {avg_val_loss:.4f}")
            
            # 定期保存模型和生成样本
            if (epoch + 1) % 5 == 0:
                # 保存检查点（包含调试信息）
                checkpoint = {
                    'epoch': epoch + 1,
                    'unet_state_dict': self.unet.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.lr_scheduler.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'config': self.config,
                    'debug_metrics': self.debug_metrics,  # 保存调试信息
                    'condition_response': condition_response,  # 保存条件响应
                }
                
                if self.ema is not None:
                    checkpoint['ema_state_dict'] = self.ema.state_dict()
                
                save_path = self.exp_dir / f'checkpoint_epoch_{epoch+1}.pt'
                torch.save(checkpoint, save_path)
                print(f"✅ 检查点已保存: {save_path}")
                
                # 保存调试信息到单独的文件
                debug_save_path = self.exp_dir / f'debug_metrics_epoch_{epoch+1}.pkl'
                import pickle
                with open(debug_save_path, 'wb') as f:
                    pickle.dump(self.debug_metrics, f)
                self._log_debug(f"调试信息已保存: {debug_save_path}")
            
            # 每个epoch结束时生成样本
            print(f"\n🎨 生成第{epoch+1}个epoch的条件扩散图像...")
            self.visualize_samples(epoch + 1)
        
        print("\n🎉 训练完成!")
    
    @torch.no_grad()
    def _test_sampling_difference(self):
        """测试DDPM和DDIM采样的差异"""
        # 创建DDPM调度器用于对比
        ddpm_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            variance_type="fixed_small",
            clip_sample=False,
            prediction_type="epsilon",
        )
        
        # 测试纯噪声去噪
        print("\n测试1: 纯噪声去噪（无条件）")
        noise = torch.randn(1, 4, 32, 32, device=self.device)
        
        # DDPM采样（使用更少步数以加快速度）
        ddpm_scheduler.set_timesteps(50)
        latents_ddpm = noise.clone()
        
        for i, t in enumerate(ddpm_scheduler.timesteps[:10]):  # 只测试前10步
            with torch.no_grad():
                # 简单的噪声预测（应该预测0）
                # 使用null class
                null_class = torch.tensor([31], device=self.device)  # null class
                noise_pred = self._forward_with_adm_condition(
                    latents_ddpm,
                    t.unsqueeze(0).to(self.device),
                    null_class
                )
                latents_ddpm = ddpm_scheduler.step(noise_pred, t, latents_ddpm, return_dict=False)[0]
                
            if i < 3:  # 打印前3步
                print(f"  DDPM Step {i}, t={t}: std={latents_ddpm.std():.3f}")
                
        # DDIM采样
        self.inference_scheduler.set_timesteps(50)
        latents_ddim = noise.clone()
        
        for i, t in enumerate(self.inference_scheduler.timesteps[:10]):  # 只测试前10步
            with torch.no_grad():
                noise_pred = self._forward_with_adm_condition(
                    latents_ddim,
                    t.unsqueeze(0).to(self.device),
                    null_class
                )
                latents_ddim = self.inference_scheduler.step(noise_pred, t, latents_ddim, return_dict=False)[0]
                
            if i < 3:  # 打印前3步
                print(f"  DDIM Step {i}, t={t}: std={latents_ddim.std():.3f}")


def main():
    parser = argparse.ArgumentParser(description='Train Diffusion Model with HuggingFace Diffusers')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    # 简化的命令行参数（覆盖配置文件）
    parser.add_argument('--data_dir', type=str, help='Path to latent data directory')
    parser.add_argument('--vae_checkpoint', type=str, help='Path to VAE checkpoint')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs')
    parser.add_argument('--lr', type=float, help='Learning rate')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 应用命令行覆盖
    if args.data_dir:
        config['data']['latent_path'] = args.data_dir
    if args.vae_checkpoint:
        config['vae_checkpoint'] = args.vae_checkpoint
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.batch_size:
        config['training']['pretrain']['batch_size'] = args.batch_size
    if args.num_epochs:
        config['training']['pretrain']['epochs'] = args.num_epochs
    if args.lr:
        config['training']['pretrain']['learning_rate'] = args.lr
    
    # 设置随机种子
    set_seed(config.get('seed', 42))
    
    # 创建ADM风格的训练器并开始训练
    trainer = ADMDiffusionTrainer(config, device=args.device)
    trainer.train()


if __name__ == '__main__':
    main()
