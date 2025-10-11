"""
基于HuggingFace Diffusers的域适应扩散模型训练脚本
第一阶段：实现基础的条件DDPM（31个用户类别）
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


class SimpleDiffusionTrainer:
    """
    基于Diffusers的简化版训练器
    第一阶段：只实现源域的条件生成
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
        
        # 初始化组件
        self._init_vae()
        self._init_model()
        self._init_optimizer()
        self._init_ema()
        
        print(f"✅ 初始化完成，实验目录: {self.exp_dir}")
        
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
    
    def _init_model(self):
        """初始化UNet和调度器"""
        model_config = self.config.get('model', {})
        
        # 创建条件编码器（用于增强条件能力）
        self.condition_encoder = nn.Sequential(
            nn.Embedding(31, 128),  # 31个用户
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, model_config.get('cross_attention_dim', 128))
        ).to(self.device)
        
        # 创建UNet2DConditionModel
        self.unet = UNet2DConditionModel(
            # 基础配置
            in_channels=model_config.get('in_channels', 4),
            out_channels=model_config.get('out_channels', 4),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D"
            ),
            up_block_types=(
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D", 
                "UpBlock2D",
                "UpBlock2D"
            ),
            # 模型大小配置
            block_out_channels=tuple(model_config.get('block_out_channels', [128, 256, 512, 512])),
            layers_per_block=model_config.get('layers_per_block', 3),
            attention_head_dim=model_config.get('attention_head_dim', 16),
            # 条件配置
            num_class_embeds=model_config.get('num_class_embeds', 31),
            class_embed_type=model_config.get('class_embed_type', 'timestep'),
            class_embeddings_concat=model_config.get('class_embeddings_concat', True),
            # Cross attention配置
            cross_attention_dim=model_config.get('cross_attention_dim', 128),
            # 其他配置
            norm_num_groups=model_config.get('norm_num_groups', 32),
            norm_eps=float(model_config.get('norm_eps', 1e-6)),
            resnet_time_scale_shift="default",
            act_fn=model_config.get('act_fn', 'silu'),
        )
        
        self.unet.to(self.device)
        
        # 打印模型信息
        unet_params = count_parameters(self.unet)
        encoder_params = count_parameters(self.condition_encoder)
        total_params = unet_params + encoder_params
        print(f"✅ 模型初始化完成:")
        print(f"   - UNet参数量: {unet_params:,}")
        print(f"   - 条件编码器参数量: {encoder_params:,}")
        print(f"   - 总参数量: {total_params:,}")
        print(f"   - 输入通道: 4 (VAE latent)")
        print(f"   - 类别数量: 31 (用户)")
        
        # 创建噪声调度器
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            variance_type="fixed_small",
            clip_sample=False,
        )
        
        # 用于推理的调度器（可选DDIM加速）
        self.inference_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )
        
    def _init_optimizer(self):
        """初始化优化器和学习率调度器"""
        train_config = self.config.get('training', {}).get('pretrain', {})
        
        # 合并所有参数
        all_params = list(self.unet.parameters()) + list(self.condition_encoder.parameters())
        
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
            
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """单步训练"""
        self.unet.train()
        self.condition_encoder.train()
        
        # 获取数据
        latents = batch['latent'].to(self.device)  # [B, 4, H, W]
        labels = batch['class_label'].to(self.device)    # [B]
        batch_size = latents.shape[0]
        
        # 重要：HuggingFace Diffusers期望latents已经被scale_factor缩放
        # 我们的数据加载器应该已经加载了缩放后的latents
        # 如果没有，需要在这里缩放：
        # latents = latents * self.vae.scale_factor
        
        # 采样随机时间步
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=self.device
        ).long()
        
        # 添加噪声
        noise = torch.randn_like(latents)
        noisy_latents = self.noise_scheduler.add_noise(
            latents, noise, timesteps
        )
        
        # 生成条件嵌入
        condition_embeds = self.condition_encoder(labels)  # [B, cross_attention_dim]
        condition_embeds = condition_embeds.unsqueeze(1)  # [B, 1, cross_attention_dim]
        
        # 前向传播 - 使用增强的条件
        with autocast(enabled=self.config.get('use_amp', True)):
            model_pred = self.unet(
                noisy_latents,
                timesteps,
                class_labels=labels,  # 类别标签
                encoder_hidden_states=condition_embeds,  # 条件嵌入用于cross attention
                return_dict=False
            )[0]
            
            # 计算损失（预测噪声）
            loss = F.mse_loss(model_pred, noise)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        if self.config.get('gradient_clip', 1.0) > 0:
            torch.nn.utils.clip_grad_norm_(
                self.unet.parameters(), 
                self.config.get('gradient_clip', 1.0)
            )
        
        self.optimizer.step()
        self.lr_scheduler.step()
        
        # 更新EMA
        if self.ema is not None:
            self.ema.update()
        
        return {
            'loss': loss.item(),
            'lr': self.optimizer.param_groups[0]['lr']
        }
    
    @torch.no_grad()
    def sample(self, num_samples: int = 8, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """生成样本"""
        self.unet.eval()
        self.condition_encoder.eval()
        
        # 如果没有指定标签，随机采样
        if labels is None:
            labels = torch.randint(0, 31, (num_samples,), device=self.device)
        
        # 生成条件嵌入
        condition_embeds = self.condition_encoder(labels)
        condition_embeds = condition_embeds.unsqueeze(1)  # [B, 1, cross_attention_dim]
        
        # 初始化随机噪声
        latents = torch.randn(
            num_samples, 4, 
            self.config['data']['latent_size'],
            self.config['data']['latent_size'],
            device=self.device
        )
        
        # 设置推理调度器
        self.inference_scheduler.set_timesteps(50)  # 使用50步DDIM
        
        # 应用EMA权重
        if self.ema is not None:
            self.ema.apply_shadow()
        
        # 去噪过程
        for t in tqdm(self.inference_scheduler.timesteps, desc="Sampling"):
            # 扩展t到batch维度
            timestep = t.expand(num_samples).to(self.device)
            
            # 预测噪声
            noise_pred = self.unet(
                latents,
                timestep,
                class_labels=labels,
                encoder_hidden_states=condition_embeds,  # 使用条件嵌入
                return_dict=False
            )[0]
            
            # 去噪一步
            latents = self.inference_scheduler.step(
                noise_pred, t, latents, return_dict=False
            )[0]
        
        # 恢复原始权重
        if self.ema is not None:
            self.ema.restore()
        
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
        
        # 解码为图像
        images = self.vae.decode_latents(all_latents)
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
        
        print(f"\n🚀 开始训练:")
        print(f"   - Epochs: {num_epochs}")
        print(f"   - Batch size: {batch_size}")
        print(f"   - 训练样本数: {len(train_loader.dataset)}")
        print(f"   - 验证样本数: {len(val_loader.dataset) if val_loader else 0}")
        
        # 在训练开始前生成一次样本作为基准
        print("\n📸 生成初始样本作为基准...")
        self.visualize_samples(0)
        
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
                        
                        # 生成条件嵌入
                        condition_embeds = self.condition_encoder(labels)
                        condition_embeds = condition_embeds.unsqueeze(1)
                        
                        # 预测
                        model_pred = self.unet(
                            noisy_latents, timesteps, 
                            class_labels=labels,
                            encoder_hidden_states=condition_embeds,
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
                # 保存检查点
                checkpoint = {
                    'epoch': epoch + 1,
                    'unet_state_dict': self.unet.state_dict(),
                    'condition_encoder_state_dict': self.condition_encoder.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.lr_scheduler.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'config': self.config
                }
                
                if self.ema is not None:
                    checkpoint['ema_state_dict'] = self.ema.state_dict()
                
                save_path = self.exp_dir / f'checkpoint_epoch_{epoch+1}.pt'
                torch.save(checkpoint, save_path)
                print(f"✅ 检查点已保存: {save_path}")
            
            # 每个epoch结束时生成样本
            print(f"\n🎨 生成第{epoch+1}个epoch的条件扩散图像...")
            self.visualize_samples(epoch + 1)
        
        print("\n🎉 训练完成!")


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
    
    # 创建训练器并开始训练
    trainer = SimpleDiffusionTrainer(config, device=args.device)
    trainer.train()


if __name__ == '__main__':
    main()
