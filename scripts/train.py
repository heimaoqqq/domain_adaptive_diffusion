"""
域适应扩散模型训练脚本（基于Epoch版本）
实现三阶段训练策略：预训练→域对齐→目标域微调
包含训练过程可视化和严格的归一化处理
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
from typing import Dict, Optional, Tuple
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# wandb是可选的，默认不使用
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️ wandb not installed, logging will be local only")

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from models import DomainConditionalUnet, DomainAdaptiveDiffusion
from utils import (
    create_dataloaders,
    set_seed,
    count_parameters,
    EMA,
    AverageMeter,
    adjust_learning_rate,
    clip_grad_norm,
    get_device,
    move_to_device,
    load_config,
    save_config,
    create_exp_dir
)
from utils.visualization import plot_training_curves


class EpochBasedTrainer:
    """
    基于Epoch的域适应扩散模型训练器
    管理三阶段训练流程，支持可视化
    """
    
    def __init__(self, config: Dict, device: str = None):
        """
        Args:
            config: 配置字典
            device: 训练设备
        """
        self.config = config
        self.device = device or get_device()
        print(f"Using device: {self.device}")
        
        # 设置随机种子
        set_seed(config.get('seed', 42))
        
        # 创建实验目录
        # 使用output_dir（如果在命令行中指定）或默认值
        base_dir = config.get('experiment', {}).get('output_dir', './checkpoints')
        if 'paths' in config and 'checkpoint_dir' in config['paths']:
            base_dir = config['paths']['checkpoint_dir']
            
        self.exp_dir = create_exp_dir(
            base_dir,
            config.get('experiment', {}).get('name', 'domain_ada'),
            config
        )
        self.checkpoint_dir = Path(self.exp_dir) / 'checkpoints'
        self.sample_dir = Path(self.exp_dir) / 'samples'
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.sample_dir.mkdir(exist_ok=True, parents=True)
        
        # 初始化模型
        self._init_models()
        
        # 初始化优化器
        self._init_optimizer()
        
        # 加载VAE（用于可视化）
        self.vae = None
        if config['training'].get('save_visualization', True):
            self._init_vae()
        
        # 训练状态
        self.current_phase = 'pretrain'
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        self.best_model_path = None
        
        # 日志
        self.losses_history = {
            'train_loss': [],
            'val_loss': [],
            'mmd_loss': []
        }
        
        # 初始化wandb（可选，默认不使用）
        self.use_wandb = config.get('logging', {}).get('use_wandb', False) and WANDB_AVAILABLE
        if self.use_wandb:
            self._init_wandb()
        else:
            if config.get('logging', {}).get('use_wandb', False) and not WANDB_AVAILABLE:
                print("⚠️ wandb requested but not available, continuing without it")
    
    def _init_models(self):
        """初始化模型"""
        model_config = self.config['model']
        diffusion_config = self.config['diffusion']
        
        # 创建UNet
        self.unet = DomainConditionalUnet(
            dim=model_config['dim'],
            dim_mults=tuple(model_config['dim_mults']),  # 确保是tuple
            channels=model_config['channels'],
            num_classes=model_config['num_classes'],
            num_domains=model_config['num_domains'],
            self_condition=model_config.get('self_condition', True),
            resnet_block_groups=model_config.get('resnet_block_groups', 8),
            learned_variance=model_config.get('learned_variance', False),
            learned_sinusoidal_cond=model_config.get('learned_sinusoidal_cond', False),
            random_fourier_features=model_config.get('random_fourier_features', False),
            learned_sinusoidal_dim=model_config.get('learned_sinusoidal_dim', 16),
            dropout=model_config.get('dropout', 0.0)
        ).to(self.device)
        
        print(f"UNet parameters: {count_parameters(self.unet):,}")
        
        # 创建扩散模型
        self.diffusion = DomainAdaptiveDiffusion(
            model=self.unet,
            image_size=self.config['data']['image_size'],
            timesteps=diffusion_config.get('timesteps', 1000),
            sampling_timesteps=diffusion_config.get('sampling_timesteps', None),
            loss_type=diffusion_config.get('loss_type', 'l2'),
            objective=diffusion_config.get('objective', 'pred_noise'),
            beta_schedule=diffusion_config.get('beta_schedule', 'cosine'),
            ddim_sampling_eta=diffusion_config.get('ddim_sampling_eta', 0.0),
            auto_normalize=diffusion_config.get('auto_normalize', True),
            offset_noise_strength=diffusion_config.get('offset_noise_strength', 0.0),
            min_snr_loss_weight=diffusion_config.get('min_snr_loss_weight', False),
            min_snr_gamma=diffusion_config.get('min_snr_gamma', 5)
        ).to(self.device)
        
        # EMA
        if self.config['domain_adaptation'].get('use_ema', True):
            self.ema = EMA(
                self.unet,
                decay=self.config['domain_adaptation']['ema_decay']
            )
        else:
            self.ema = None
    
    def _init_vae(self):
        """初始化VAE用于可视化"""
        vae_checkpoint = self.config.get('vae', {}).get('checkpoint', '../vae/checkpoints/kl_vae_best.pt')
        
        try:
            import sys
            from pathlib import Path
            vae_module_path = Path(__file__).parent.parent / 'vae'
            sys.path.insert(0, str(vae_module_path))
            from kl_vae import KL_VAE
            
            if Path(vae_checkpoint).exists():
                checkpoint = torch.load(vae_checkpoint, map_location=self.device)
                self.vae = KL_VAE()
                self.vae.load_state_dict(checkpoint['model_state_dict'])
                self.vae = self.vae.to(self.device)
                self.vae.eval()
                print(f"KL-VAE loaded for visualization: {vae_checkpoint}")
            else:
                self.vae = None
                print("VAE checkpoint not found, skipping visualization")
        except Exception as e:
            print(f"Warning: Could not load VAE for visualization: {e}")
            self.vae = None
    
    def _init_optimizer(self):
        """初始化优化器"""
        # 使用预训练阶段的配置初始化
        pretrain_config = self.config['training']['pretrain']
        
        optimizer_type = self.config['training'].get('optimizer', 'adamw')
        if optimizer_type == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.unet.parameters(),
                lr=float(pretrain_config['learning_rate']),  # 确保转换为float
                betas=(
                    float(self.config['training'].get('beta1', 0.9)),
                    float(self.config['training'].get('beta2', 0.999))
                ),
                eps=float(self.config['training'].get('eps', 1e-8)),
                weight_decay=float(pretrain_config.get('weight_decay', 0.01))
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        # 混合精度训练
        self.use_amp = self.config['training'].get('amp', True) and self.device != 'cpu'
        if self.use_amp:
            self.scaler = GradScaler()
    
    def _init_wandb(self):
        """初始化wandb"""
        if WANDB_AVAILABLE:
            wandb.init(
                project=self.config['logging'].get('wandb_project', 'domain-adaptive-diffusion'),
                entity=self.config['logging'].get('wandb_entity'),
                name=self.config['experiment'].get('name', 'domain_ada'),
                config=self.config,
                tags=self.config['experiment'].get('tags', [])
            )
            print(f"✅ wandb initialized: {wandb.run.name}")
    
    def decode_and_visualize(
        self,
        latents: torch.Tensor,
        save_path: str,
        title: str = "Generated Samples",
        num_show: int = 16
    ):
        """
        解码latents并保存可视化
        
        正确处理归一化：
        1. VAE decode输出范围通常是[-1, 1]
        2. 需要转换到[0, 1]用于可视化
        """
        if self.vae is None:
            print("VAE not available for visualization")
            return
        
        with torch.no_grad():
            # 确保只可视化前num_show个样本
            latents = latents[:num_show]
            
            # VAE解码
            if hasattr(self.vae, 'decode'):
                images = self.vae.decode(latents)
            else:
                # 如果没有decode方法，尝试decoder
                images = self.vae.decoder(latents) if hasattr(self.vae, 'decoder') else latents
            
            # 归一化到[0, 1]
            # 假设VAE输出在[-1, 1]范围内（这是标准做法）
            images = (images + 1.0) / 2.0
            images = torch.clamp(images, 0.0, 1.0)
            
            # 转换为numpy
            images = images.cpu().numpy()
            images = (images * 255).astype(np.uint8)
            
            # 创建网格可视化
            grid_size = int(np.sqrt(num_show))
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
            axes = axes.flatten()
            
            for i in range(num_show):
                if i < len(images):
                    # 转换通道顺序：CHW -> HWC
                    img = np.transpose(images[i], (1, 2, 0))
                    axes[i].imshow(img)
                axes[i].axis('off')
            
            plt.suptitle(title, fontsize=16)
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Visualization saved to {save_path}")
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        total_epochs: int,
        phase: str
    ) -> Dict[str, float]:
        """
        训练一个epoch
        
        Returns:
            平均损失字典
        """
        self.unet.train()
        
        epoch_losses = {
            'total_loss': AverageMeter(),
            'base_loss': AverageMeter(),
            'mmd_loss': AverageMeter()
        }
        
        # 获取阶段配置
        phase_config = self.config['training'][phase]
        warmup_epochs = phase_config.get('warmup_epochs', 5)
        
        # 确保学习率是浮点数
        base_lr = float(phase_config['learning_rate'])
        lr_min = float(self.config['training'].get('lr_min', 1e-6))
        
        # 计算当前epoch的学习率
        if epoch < warmup_epochs:
            # Warmup阶段
            lr = base_lr * (epoch + 1) / warmup_epochs
        else:
            # Cosine退火
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            lr = lr_min + 0.5 * (base_lr - lr_min) * (1 + np.cos(np.pi * progress))
        
        # 设置学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = float(lr)
        
        # 训练循环
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            # 准备数据
            latents = batch['latent'].to(self.device)
            class_labels = batch['class_label'].to(self.device)
            domain_labels = batch['domain_label'].to(self.device)
            
            # 采样时间步
            batch_size = latents.shape[0]
            t = torch.randint(
                0, self.diffusion.timesteps, (batch_size,),
                device=self.device, dtype=torch.long
            )
            
            # 前向传播
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    loss_dict = self.diffusion.p_losses_with_details(
                        latents, t,
                        class_labels=class_labels,
                        domain_labels=domain_labels
                    )
                    loss = loss_dict['total_loss']
                
                # 反向传播
                self.scaler.scale(loss).backward()
                
                # 梯度裁剪
                if self.config['training'].get('gradient_clip', 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = clip_grad_norm(
                        self.unet,
                        self.config['training']['gradient_clip']
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss_dict = self.diffusion.p_losses_with_details(
                    latents, t,
                    class_labels=class_labels,
                    domain_labels=domain_labels
                )
                loss = loss_dict['total_loss']
                
                loss.backward()
                
                # 梯度裁剪
                if self.config['training'].get('gradient_clip', 0) > 0:
                    grad_norm = clip_grad_norm(
                        self.unet,
                        self.config['training']['gradient_clip']
                    )
                
                self.optimizer.step()
            
            # 更新EMA
            if self.ema is not None:
                self.ema.update()
            
            # 记录损失
            for key, value in loss_dict.items():
                if key in epoch_losses:
                    epoch_losses[key].update(value.item() if torch.is_tensor(value) else value, batch_size)
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{epoch_losses['total_loss'].avg:.4f}",
                'lr': f"{lr:.2e}"
            })
            
            self.global_step += 1
        
        # 返回平均损失
        avg_losses = {key: meter.avg for key, meter in epoch_losses.items()}
        
        return avg_losses
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """验证"""
        if dataloader is None:
            return {}
        
        self.unet.eval()
        val_losses = AverageMeter()
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                latents = batch['latent'].to(self.device)
                class_labels = batch['class_label'].to(self.device)
                domain_labels = batch['domain_label'].to(self.device)
                
                # 随机时间步
                batch_size = latents.shape[0]
                t = torch.randint(
                    0, self.diffusion.timesteps, (batch_size,),
                    device=self.device, dtype=torch.long
                )
                
                # 计算损失
                loss_dict = self.diffusion.p_losses_with_details(
                    latents, t,
                    class_labels=class_labels,
                    domain_labels=domain_labels
                )
                
                val_losses.update(loss_dict['total_loss'].item(), batch_size)
        
        self.unet.train()
        
        return {'val_loss': val_losses.avg}
    
    def sample_and_visualize(self, epoch: int, phase: str):
        """生成样本并可视化"""
        self.unet.eval()
        
        num_samples = self.config['training'].get('num_samples', 31)
        
        with torch.no_grad():
            # 使用EMA模型（如果有）
            if self.ema is not None:
                self.ema.store()
                self.ema.copy_to()
            
            # 生成每个用户的样本
            class_labels = torch.arange(
                min(num_samples, self.config['model']['num_classes'])
            ).to(self.device)
            
            # 生成目标域样本
            domain_labels = torch.ones_like(class_labels)  # 目标域=1
            
            # 采样
            samples = self.diffusion.sample(
                batch_size=len(class_labels),
                class_labels=class_labels,
                domain_labels=domain_labels,
                guidance_scale=self.config['generation']['guidance_scale']
            )
            
            # 恢复原始模型（如果使用EMA）
            if self.ema is not None:
                self.ema.restore()
            
            # 可视化
            save_path = self.sample_dir / f'{phase}_epoch_{epoch}.png'
            self.decode_and_visualize(
                samples,
                str(save_path),
                title=f'{phase.capitalize()} - Epoch {epoch}'
            )
        
        self.unet.train()
    
    def save_checkpoint(self, phase: str, epoch: int, is_best: bool = False):
        """
        保存checkpoint
        如果是最佳模型，删除之前的最佳模型
        """
        checkpoint = {
            'phase': phase,
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.unet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'losses_history': self.losses_history,
            'best_loss': self.best_loss
        }
        
        if self.ema is not None:
            checkpoint['ema_state_dict'] = self.ema.state_dict()
        
        if is_best:
            # 删除旧的最佳模型
            if self.best_model_path and self.best_model_path.exists():
                self.best_model_path.unlink()
                print(f"Deleted old best model: {self.best_model_path}")
            
            # 保存新的最佳模型
            filepath = self.checkpoint_dir / f'best_{phase}_epoch_{epoch}.pt'
            torch.save(checkpoint, filepath)
            self.best_model_path = filepath
            print(f"Saved best model: {filepath}")
        else:
            # 普通checkpoint
            filepath = self.checkpoint_dir / f'{phase}_epoch_{epoch}.pt'
            torch.save(checkpoint, filepath)
            
            # 删除旧的checkpoint（只保留最近3个）
            if self.config['training'].get('delete_old_checkpoints', True):
                pattern = f'{phase}_epoch_*.pt'
                checkpoints = sorted(self.checkpoint_dir.glob(pattern))
                if len(checkpoints) > 3:
                    for old_ckpt in checkpoints[:-3]:
                        if 'best' not in old_ckpt.name:
                            old_ckpt.unlink()
                            print(f"Deleted old checkpoint: {old_ckpt}")
    
    def train_phase(self, phase: str):
        """
        训练一个阶段（基于epoch）
        """
        print(f"\n{'='*60}")
        print(f"🎯 Starting {phase.upper()} phase")
        print(f"{'='*60}")
        
        # 获取阶段配置
        phase_config = self.config['training'][phase]
        num_epochs = phase_config.get('epochs', phase_config.get('num_epochs', 20))
        
        # 创建数据加载器
        batch_size = phase_config.get('batch_size', 32)
        if phase == 'align':
            # 对齐阶段的特殊处理
            batch_size_source = phase_config.get('batch_size_source', 48)
            batch_size_target = phase_config.get('batch_size_target', 16)
            batch_size = batch_size_source + batch_size_target
        
        train_loader, val_loader = create_dataloaders(
            data_path=self.config['data']['latent_path'],
            phase=phase,
            batch_size=batch_size,
            num_workers=self.config['training'].get('num_workers', 4),
            domain_balance_ratio=self.config['domain_adaptation']['domain_balance_ratio'],
            augmentation=phase_config.get('use_augmentation', False),
            augmentation_config=self.config.get('augmentation'),
            device=self.device
        )
        
        # 重置优化器学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = float(phase_config['learning_rate'])
        
        # 训练循环
        for epoch in range(num_epochs):
            # 训练一个epoch
            train_losses = self.train_epoch(
                train_loader,
                epoch,
                num_epochs,
                phase
            )
            
            # 验证
            val_losses = self.validate(val_loader)
            
            # 打印epoch总结
            print(f"\n{phase.upper()} Epoch {epoch+1}/{num_epochs} Summary:")
            print(f"  Train Loss: {train_losses['total_loss']:.4f}")
            if 'mmd_loss' in train_losses and train_losses['mmd_loss'] > 0:
                print(f"  MMD Loss: {train_losses['mmd_loss']:.4f}")
            if val_losses:
                print(f"  Val Loss: {val_losses['val_loss']:.4f}")
            
            # 记录历史
            self.losses_history['train_loss'].append(train_losses['total_loss'])
            if val_losses:
                self.losses_history['val_loss'].append(val_losses['val_loss'])
            
            # 生成可视化样本
            if (epoch + 1) % self.config['training'].get('sample_every_epochs', 2) == 0:
                self.sample_and_visualize(epoch + 1, phase)
            
            # 保存checkpoint
            if (epoch + 1) % self.config['training'].get('save_every_epochs', 5) == 0:
                self.save_checkpoint(phase, epoch + 1, is_best=False)
            
            # 保存最佳模型
            current_loss = val_losses.get('val_loss', train_losses['total_loss'])
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.save_checkpoint(phase, epoch + 1, is_best=True)
            
            # 早停检查
            if self.config['training'].get('early_stopping', True):
                # 这里可以添加早停逻辑
                pass
        
        # 阶段结束，保存最终checkpoint
        self.save_checkpoint(phase, num_epochs, is_best=False)
        
        # 绘制训练曲线
        self.plot_training_curves(phase)
    
    def plot_training_curves(self, phase: str):
        """绘制训练曲线"""
        plt.figure(figsize=(12, 4))
        
        # 训练损失
        plt.subplot(1, 2, 1)
        plt.plot(self.losses_history['train_loss'], label='Train Loss')
        if self.losses_history['val_loss']:
            plt.plot(self.losses_history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{phase.capitalize()} Training Loss')
        plt.legend()
        plt.grid(True)
        
        # MMD损失（如果有）
        if 'mmd_loss' in self.losses_history and any(self.losses_history['mmd_loss']):
            plt.subplot(1, 2, 2)
            plt.plot(self.losses_history['mmd_loss'], label='MMD Loss')
            plt.xlabel('Step')
            plt.ylabel('MMD Loss')
            plt.title('Domain Alignment Loss')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.exp_dir / f'{phase}_training_curves.png')
        plt.close()
    
    def train(self):
        """完整的三阶段训练流程"""
        print(f"Starting domain adaptive diffusion training")
        print(f"Experiment directory: {self.exp_dir}")
        
        # Phase 1: 预训练
        self.current_phase = 'pretrain'
        self.train_phase('pretrain')
        
        # Phase 2: 域对齐
        self.current_phase = 'align'
        self.train_phase('align')
        
        # Phase 3: 目标域微调
        self.current_phase = 'finetune'
        self.train_phase('finetune')
        
        print("\n" + "="*60)
        print("✅ Training completed!")
        print(f"Best model saved at: {self.best_model_path}")
        print(f"Final loss: {self.best_loss:.4f}")
        print("="*60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Train domain adaptive diffusion model (epoch-based)')
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/domain_ada.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (default: auto-detect)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    # 日志选项
    parser.add_argument('--no-wandb', action='store_true', 
                        help='Disable wandb even if configured (default: wandb is disabled)')
    parser.add_argument('--use-wandb', action='store_true',
                        help='Enable wandb logging (requires wandb package)')
    
    # 常用参数的简化版本
    parser.add_argument('--data_dir', type=str, 
                        help='Path to processed latent data (same as --data.latent_path)')
    parser.add_argument('--vae_checkpoint', type=str,
                        help='Path to VAE checkpoint (same as --vae.checkpoint)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for checkpoints')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size for all phases')
    parser.add_argument('--num_epochs', type=int, default=None,
                        help='Override total epochs (distributed across phases)')
    
    # 覆盖配置的参数（完整路径版本）
    parser.add_argument('--data.latent_path', type=str, help='Override data path')
    parser.add_argument('--vae.checkpoint', type=str, help='Override VAE checkpoint')
    parser.add_argument('--experiment.name', type=str, help='Override experiment name')
    parser.add_argument('--training.pretrain.epochs', type=int, help='Override pretrain epochs')
    parser.add_argument('--training.align.epochs', type=int, help='Override align epochs')
    parser.add_argument('--training.finetune.epochs', type=int, help='Override finetune epochs')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 处理简化参数映射
    if args.data_dir:
        if 'data' not in config:
            config['data'] = {}
        config['data']['latent_path'] = args.data_dir
        
    if args.vae_checkpoint:
        if 'vae' not in config:
            config['vae'] = {}
        config['vae']['checkpoint'] = args.vae_checkpoint
        
    if args.output_dir:
        if 'experiment' not in config:
            config['experiment'] = {}
        config['experiment']['output_dir'] = args.output_dir
        
    if args.batch_size is not None:
        # 应用到所有训练阶段
        for phase in ['pretrain', 'align', 'finetune']:
            if 'training' not in config:
                config['training'] = {}
            if phase not in config['training']:
                config['training'][phase] = {}
            if phase == 'align':
                # 对齐阶段特殊处理
                config['training'][phase]['batch_size_source'] = int(args.batch_size * 0.75)
                config['training'][phase]['batch_size_target'] = int(args.batch_size * 0.25)
            else:
                config['training'][phase]['batch_size'] = args.batch_size
                
    if args.num_epochs is not None:
        # 按比例分配到各阶段 (例如: 40% pretrain, 40% align, 20% finetune)
        pretrain_epochs = int(args.num_epochs * 0.4)
        align_epochs = int(args.num_epochs * 0.4)
        finetune_epochs = args.num_epochs - pretrain_epochs - align_epochs
        
        if 'training' not in config:
            config['training'] = {}
        for phase, epochs in [('pretrain', pretrain_epochs), 
                              ('align', align_epochs), 
                              ('finetune', finetune_epochs)]:
            if phase not in config['training']:
                config['training'][phase] = {}
            config['training'][phase]['epochs'] = epochs
    
    # 处理wandb设置
    if args.no_wandb:
        if 'logging' not in config:
            config['logging'] = {}
        config['logging']['use_wandb'] = False
    elif args.use_wandb:
        if 'logging' not in config:
            config['logging'] = {}
        config['logging']['use_wandb'] = True
        if not WANDB_AVAILABLE:
            print("❌ Error: wandb requested but not installed!")
            print("   Please install with: pip install wandb")
            import sys
            sys.exit(1)
    
    # 应用命令行覆盖（处理带点的参数）
    skip_keys = {'config', 'device', 'resume', 'no_wandb', 'use_wandb', 
                 'data_dir', 'vae_checkpoint', 'output_dir', 'batch_size', 'num_epochs'}
    for key, value in vars(args).items():
        if key in skip_keys:
            continue
        if value is not None and '.' in key:
            # 处理嵌套配置
            keys = key.split('.')
            current = config
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
    
    # 创建训练器
    trainer = EpochBasedTrainer(config, device=args.device)
    
    # 恢复训练（如果指定）
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        # TODO: 实现恢复逻辑
    
    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()
