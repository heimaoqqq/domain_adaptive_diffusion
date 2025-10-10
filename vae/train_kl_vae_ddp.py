"""
Train KL-VAE with DDP support for Kaggle dual GPU
基于Stable Diffusion架构的VAE训练 - DDP版本
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from PIL import Image
import argparse
import os
import time
from pathlib import Path
from tqdm import tqdm
import matplotlib
# 在Kaggle环境使用Agg后端避免GUI问题
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 混合精度训练
from torch.cuda.amp import autocast, GradScaler

# DDP imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

from kl_vae import KL_VAE

# Try to import perceptual loss (optional)
try:
    from perceptual_loss import CombinedPerceptualLoss
    PERCEPTUAL_AVAILABLE = True
except ImportError:
    print("Warning: perceptual_loss module not found. Perceptual loss will be disabled.")
    PERCEPTUAL_AVAILABLE = False


class MicroDopplerDataset(Dataset):
    """Dataset for micro-Doppler spectrograms"""
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Collect all images
        self.images = []
        
        # Only print on main process
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Searching for images in: {self.root_dir}")
            print(f"Directory exists: {self.root_dir.exists()}")
        
        # List all subdirectories for debugging
        if self.root_dir.exists():
            subdirs = list(self.root_dir.iterdir())
            
            # Check if this is a single gait type directory with ID_* folders
            id_dirs = [d for d in subdirs if d.is_dir() and d.name.startswith('ID_')]
            
            if id_dirs:  # This is a gait type directory
                for user_dir in sorted(id_dirs):
                    # Collect all jpg files (case-insensitive)
                    jpg_files = []
                    seen_paths = set()
                    
                    for pattern in ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG']:
                        for img_path in user_dir.glob(pattern):
                            # Normalize path to avoid duplicates on case-insensitive filesystems
                            normalized_path = str(img_path).lower()
                            if normalized_path not in seen_paths:
                                jpg_files.append(img_path)
                                seen_paths.add(normalized_path)
                    
                    self.images.extend(jpg_files)
            else:
                # Check if root contains gait type subdirectories
                gait_dirs = [d for d in subdirs 
                           if d.is_dir() and '_' in d.name and not d.name.startswith('.')]
                
                for gait_dir in sorted(gait_dirs):
                    for user_dir in sorted(gait_dir.iterdir()):
                        if user_dir.is_dir() and user_dir.name.startswith('ID_'):
                            # Collect all jpg files (case-insensitive)
                            seen_paths = set()
                            for pattern in ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG']:
                                for img_path in user_dir.glob(pattern):
                                    # Normalize path to avoid duplicates on case-insensitive filesystems
                                    normalized_path = str(img_path).lower()
                                    if normalized_path not in seen_paths:
                                        self.images.append(img_path)
                                        seen_paths.add(normalized_path)
                    
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Total images found: {len(self.images)}")
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image


def setup_ddp(rank, world_size):
    """Setup DDP environment"""
    # 如果已经由torchrun设置了环境变量，不要覆盖
    if 'MASTER_ADDR' not in os.environ:
        # 对Kaggle环境使用127.0.0.1而不是localhost
        os.environ['MASTER_ADDR'] = '127.0.0.1'
    if 'MASTER_PORT' not in os.environ:
        # 使用不同的端口避免冲突
        os.environ['MASTER_PORT'] = '29500'
    
    # Initialize process group
    try:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        # Set device
        torch.cuda.set_device(rank)
        
        # 验证设置
        if rank == 0:
            print(f"DDP initialized: Rank {rank}/{world_size}, Device: cuda:{rank}")
    except Exception as e:
        print(f"Error initializing DDP: {e}")
        # 尝试使用gloo作为后备
        print("Trying gloo backend...")
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)


def cleanup_ddp():
    """Cleanup DDP"""
    dist.destroy_process_group()


def all_reduce_mean(tensor):
    """Average tensor across all GPUs"""
    if not dist.is_initialized():
        return tensor
    
    world_size = dist.get_world_size()
    if world_size == 1:
        return tensor
    
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= world_size
    return tensor


def train_epoch(model: DDP, 
                dataloader: DataLoader, 
                optimizer: torch.optim.Optimizer,
                kl_weight: float,
                device: torch.device,
                rank: int,
                perceptual_loss_fn=None,
                gradient_accumulation_steps: int = 1,
                epoch: int = 0,
                use_fp16: bool = False,
                scaler=None) -> Dict[str, float]:
    """Train for one epoch with DDP"""
    model.train()
    
    total_loss = 0
    total_rec_loss = 0
    total_kl_loss = 0
    total_perceptual_loss = 0
    
    # 验证数据量
    n_batches_local = len(dataloader)
    if rank == 0 and epoch == 0:
        print(f"  GPU {rank}: Processing {n_batches_local} batches")
    
    # Only show progress bar on rank 0
    if rank == 0:
        progress_bar = tqdm(dataloader, desc='Training')
    else:
        progress_bar = dataloader
    
    optimizer.zero_grad()  # 移到循环外
        
    for i, batch in enumerate(progress_bar):
        batch = batch.to(device)
        
        # Compute loss with or without FP16
        if use_fp16 and scaler is not None:
            with autocast():
                losses = model.module.get_loss(batch, kl_weight=kl_weight, perceptual_loss_fn=perceptual_loss_fn)
                loss = losses['loss'] / gradient_accumulation_steps
            
            # Backward with scaled gradients
            scaler.scale(loss).backward()
        else:
            # Standard FP32 training
            losses = model.module.get_loss(batch, kl_weight=kl_weight, perceptual_loss_fn=perceptual_loss_fn)
            loss = losses['loss'] / gradient_accumulation_steps
            loss.backward()
        
        # Update weights every gradient_accumulation_steps
        if (i + 1) % gradient_accumulation_steps == 0:
            if use_fp16 and scaler is not None:
                # Unscale gradients and clip
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # Step with scaler
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard update
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            optimizer.zero_grad()
        
        # Record losses (记录原始loss，不是缩放后的)
        total_loss += losses['loss'].item()
        total_rec_loss += losses['rec_loss'].item()
        total_kl_loss += losses['kl_loss'].item()
        if 'perceptual_loss' in losses:
            total_perceptual_loss += losses['perceptual_loss'].item()
        
        # Update progress bar (only on rank 0)
        if rank == 0 and hasattr(progress_bar, 'set_postfix'):
            postfix_dict = {
                'loss': f"{losses['loss'].item():.4f}",  # 显示原始loss
                'rec': f"{losses['rec_loss'].item():.4f}",
                'kl': f"{losses['kl_loss'].item():.4f}"
            }
            if 'perceptual_loss' in losses and losses['perceptual_loss'].item() > 0:
                postfix_dict['perc'] = f"{losses['perceptual_loss'].item():.4f}"
            if gradient_accumulation_steps > 1:
                postfix_dict['acc'] = f"{(i + 1) % gradient_accumulation_steps}/{gradient_accumulation_steps}"
            progress_bar.set_postfix(postfix_dict)
        
    n_batches = len(dataloader)
    
    # Aggregate losses across GPUs
    avg_losses = {
        'loss': all_reduce_mean(torch.tensor(total_loss / n_batches, device=device)).item(),
        'rec_loss': all_reduce_mean(torch.tensor(total_rec_loss / n_batches, device=device)).item(),
        'kl_loss': all_reduce_mean(torch.tensor(total_kl_loss / n_batches, device=device)).item(),
        'perceptual_loss': all_reduce_mean(torch.tensor(total_perceptual_loss / n_batches, device=device)).item() if perceptual_loss_fn else 0.0
    }
    
    return avg_losses


@torch.no_grad()
def validate(model: DDP, 
             dataloader: DataLoader, 
             kl_weight: float,
             device: torch.device,
             rank: int,
             save_samples: bool = False,
             save_path: str = None,
             use_fp16: bool = False) -> Dict[str, float]:
    """Validate model with DDP"""
    model.eval()
    
    total_loss = 0
    total_rec_loss = 0
    total_kl_loss = 0
    
    # For visualization (only on rank 0)
    first_batch = None
    first_recon = None
    
    # Only show progress bar on rank 0
    if rank == 0:
        progress_bar = tqdm(dataloader, desc='Validation')
    else:
        progress_bar = dataloader
    
    for i, batch in enumerate(progress_bar):
        batch = batch.to(device)
        
        # Forward pass with or without FP16
        if use_fp16:
            with autocast():
                recon, posterior = model.module(batch)
                rec_loss = nn.functional.mse_loss(batch, recon)
                kl_loss = posterior.kl()
                kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
                loss = rec_loss + kl_weight * kl_loss
        else:
            recon, posterior = model.module(batch)
            rec_loss = nn.functional.mse_loss(batch, recon)
            kl_loss = posterior.kl()
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
            loss = rec_loss + kl_weight * kl_loss
        
        # Record losses
        total_loss += loss.item()
        total_rec_loss += rec_loss.item()
        total_kl_loss += kl_loss.item()
        
        # Save first batch for visualization (only on rank 0)
        if i == 0 and save_samples and rank == 0:
            # 转换为float32避免FP16的dtype问题
            first_batch = batch[:8].cpu().float()
            first_recon = recon[:8].cpu().float()
            
    n_batches = len(dataloader)
    
    # Aggregate losses across GPUs
    avg_losses = {
        'loss': all_reduce_mean(torch.tensor(total_loss / n_batches, device=device)).item(),
        'rec_loss': all_reduce_mean(torch.tensor(total_rec_loss / n_batches, device=device)).item(),
        'kl_loss': all_reduce_mean(torch.tensor(total_kl_loss / n_batches, device=device)).item()
    }
    
    # Save reconstructions - only on rank 0
    if save_samples and first_batch is not None and rank == 0:
        try:
            # 确保路径存在
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
                
            fig = plt.figure(figsize=(16, 6))
            
            # 只显示原图和重建图对比
            n_samples = min(8, first_batch.shape[0])
            
            for i in range(n_samples):
                # 原图
                ax1 = plt.subplot(2, n_samples, i + 1)
                # 确保转换为float32并移到CPU，避免dtype问题
                img_orig = first_batch[i].cpu().float().permute(1, 2, 0).clamp(0, 1).numpy()
                ax1.imshow(img_orig)
                ax1.axis('off')
                if i == 0:
                    ax1.set_title('Original', fontsize=14, fontweight='bold')
                    
                # 重建图
                ax2 = plt.subplot(2, n_samples, n_samples + i + 1)
                # 确保转换为float32并移到CPU，避免dtype问题
                img_recon = first_recon[i].cpu().float().permute(1, 2, 0).clamp(0, 1).numpy()
                ax2.imshow(img_recon)
                ax2.axis('off')
                if i == 0:
                    ax2.set_title('Reconstruction', fontsize=14, fontweight='bold')
            
            # 添加epoch和损失信息作为标题
            epoch_num = save_path.split('epoch_')[1].split('.')[0] if 'epoch_' in save_path else 'best'
            fig.suptitle(f'Epoch {epoch_num} - Rec Loss: {avg_losses["rec_loss"]:.4f}', 
                         fontsize=16, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close('all')  # 确保关闭所有图形
            print(f"  Saved visualization: {save_path}")
        except Exception as e:
            print(f"  Warning: Failed to save visualization: {e}")
            plt.close('all')  # 确保清理
    
    return avg_losses


def train_ddp(rank, world_size, args):
    """Main training function for each process"""
    # Setup DDP
    setup_ddp(rank, world_size)
    
    # Device for this process
    device = torch.device(f'cuda:{rank}')
    
    if rank == 0:
        print(f"Using DDP with {world_size} GPUs")
    
    # 检测Kaggle环境并设置正确路径
    if os.path.exists('/kaggle'):
        # 在Kaggle环境
        if not args.checkpoint_dir.startswith('/kaggle'):
            args.checkpoint_dir = f'/kaggle/working/{args.checkpoint_dir}'
            if rank == 0:
                print(f"Detected Kaggle environment, using: {args.checkpoint_dir}")
    
    # Create directories (only on rank 0)
    if rank == 0:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        print(f"Checkpoint directory: {args.checkpoint_dir}")
    
    # Data transforms - 可以考虑降低分辨率以节省显存
    img_size = args.img_size  # 默认256，可以改为128或192节省显存
    if rank == 0 and img_size != 256:
        print(f"Using reduced image size: {img_size}x{img_size}")
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])
    
    # Dataset
    full_dataset = MicroDopplerDataset(args.data_dir, transform=transform)
    
    # Split
    n_val = int(len(full_dataset) * args.val_split)
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    
    if rank == 0:
        print(f"\nDataset: {len(full_dataset)} images")
        print(f"Train: {n_train}, Val: {n_val}")
    
    # Distributed samplers - 重要：每个GPU只处理数据的一部分
    # DistributedSampler会自动将数据分成world_size份
    # 每个GPU处理 n_samples/world_size 个样本
    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=True,
        seed=42  # 固定种子保证一致性
    )
    val_sampler = DistributedSampler(
        val_dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=False
    )
    
    # 验证数据分配
    if rank == 0:
        print(f"Data distribution check:")
        print(f"  Total train samples: {len(train_dataset)}")
        print(f"  Samples per GPU: {len(train_dataset) // world_size}")
        print(f"  Train sampler length: {len(train_sampler)}")
    
    # Dataloaders with DistributedSampler
    # 注意：每个GPU的batch_size，总batch_size = batch_size * world_size
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,  # Important for DDP
        persistent_workers=True if args.num_workers > 0 else False  # 保持数据加载器活跃
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    # Model
    ddconfig = dict(
        double_z=True,
        z_channels=args.embed_dim,
        in_channels=3,
        out_ch=3,
        ch=args.ch,
        ch_mult=tuple(args.ch_mult),
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.0
    )
    
    model = KL_VAE(
        ddconfig=ddconfig,
        embed_dim=args.embed_dim,
        scale_factor=args.scale_factor
    ).to(device)
    
    # Wrap model in DDP
    model = DDP(model, device_ids=[rank])
    
    # 尝试使用torch.compile加速（PyTorch 2.0+）
    try:
        if hasattr(torch, 'compile'):
            model = torch.compile(model, mode='reduce-overhead')
            if rank == 0:
                print("  Using torch.compile optimization")
    except Exception:
        pass  # 如果不支持就跳过
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nModel: KL-VAE")
        print(f"  Parameters: {total_params:,}")
        print(f"  Latent: {args.embed_dim}ch, {2**len(args.ch_mult)}x downsample")
    
    # Optimizer (adjust learning rate for larger effective batch size)
    # 学习率缩放：根据有效batch size的比例
    # 基准：batch_size=4 (单GPU), lr=4.5e-6
    base_batch_size = 4  # 原始单GPU配置
    current_batch_size = args.batch_size * world_size * args.gradient_accumulation_steps
    lr_scale = np.sqrt(current_batch_size / base_batch_size)
    effective_lr = args.lr * lr_scale
    
    
    optimizer = optim.Adam(model.parameters(), lr=effective_lr, betas=(0.5, 0.9))
    
    # Resume (only load on rank 0 then broadcast)
    start_epoch = 0
    if args.resume and rank == 0:
        checkpoint = torch.load(args.resume, map_location=device)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resumed from epoch {start_epoch}")
    
    # Synchronize across GPUs
    dist.barrier()
    
    # Initialize perceptual loss if requested
    perceptual_loss_fn = None
    if args.use_perceptual:
        if not PERCEPTUAL_AVAILABLE:
            if rank == 0:
                print("\n警告: 感知损失模块不可用，将使用标准MSE损失")
            args.use_perceptual = False
        else:
            perceptual_loss_fn = CombinedPerceptualLoss(
                mse_weight=1.0,  # Keep full MSE
                perceptual_weight=args.perceptual_weight,
                style_weight=0.0,  # No style loss for spectrograms
                loss_type='vgg',
                device=device
            )
            if rank == 0:
                print(f"\n感知损失配置:")
                print(f"  - 将在第 {args.perceptual_start_epoch} epoch后启用")
                print(f"  - 感知损失权重: {args.perceptual_weight}")
    
    # Setup FP16 if requested
    scaler = None
    if args.use_fp16:
        scaler = GradScaler()
        if rank == 0:
            print("\n✅ Using FP16 mixed precision training")
            print("   Expected memory savings: ~40-50%")
            print("   Expected batch size increase: 1.5-2x")
    
    # Training
    best_val_loss = float('inf')
    patience_counter = 0
    
    if rank == 0:
        print(f"\n训练配置:")
        print(f"  Epochs: {args.epochs}")
        print(f"  Per GPU batch size: {args.batch_size}")
        print(f"  Total batch size: {args.batch_size * world_size}")
        print(f"  Learning rate: {effective_lr:.2e}")
        print(f"  GPUs: {world_size}")
        print(f"  Batches per GPU per epoch: {len(train_loader)}")
        if args.use_fp16:
            print(f"  Mixed Precision: FP16 enabled")
    
    for epoch in range(start_epoch, args.epochs):
        # Set epoch for distributed sampler (important!)
        train_sampler.set_epoch(epoch)
        
        epoch_start = time.time()
        
        if rank == 0:
            print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        
        # KL weight warmup
        if epoch < args.warmup_epochs:
            kl_weight = args.kl_weight * (epoch + 1) / args.warmup_epochs
        else:
            kl_weight = args.kl_weight
            
        if rank == 0:
            print(f"KL weight: {kl_weight:.2e}")
        
        # Decide whether to use perceptual loss
        current_perceptual_fn = None
        if args.use_perceptual and epoch >= args.perceptual_start_epoch:
            current_perceptual_fn = perceptual_loss_fn
            if rank == 0:
                print(f"感知损失: 已启用 (权重={args.perceptual_weight})")
        elif args.use_perceptual and rank == 0:
            epochs_until_perceptual = args.perceptual_start_epoch - epoch
            print(f"感知损失: {epochs_until_perceptual} epoch后启用")
        
        # Train
        train_losses = train_epoch(model, train_loader, optimizer, kl_weight, device, rank, 
                                  current_perceptual_fn, gradient_accumulation_steps=args.gradient_accumulation_steps,
                                  epoch=epoch, use_fp16=args.use_fp16, scaler=scaler)
        
        if rank == 0:
            print(f"Train - Loss: {train_losses['loss']:.4f}, "
                  f"Rec: {train_losses['rec_loss']:.4f}, "
                  f"KL: {train_losses['kl_loss']:.4f}", end='')
            if train_losses.get('perceptual_loss', 0) > 0:
                print(f", Perceptual: {train_losses['perceptual_loss']:.4f}")
            else:
                print()
        
        # Validate - 每个epoch都生成可视化 (only on rank 0)
        sample_path = None
        if rank == 0:
            sample_path = os.path.join(args.checkpoint_dir, f'samples_epoch_{epoch+1}.png')
        
        val_losses = validate(model, val_loader, kl_weight, device, rank,
                            save_samples=(rank == 0),  # Only save on rank 0
                            save_path=sample_path,
                            use_fp16=args.use_fp16)
        
        if rank == 0:
            print(f"Val - Loss: {val_losses['loss']:.4f}, "
                  f"Rec: {val_losses['rec_loss']:.4f}, "
                  f"KL: {val_losses['kl_loss']:.4f}")
            
            # 显示epoch时间
            epoch_time = time.time() - epoch_start
            print(f"Epoch time: {epoch_time:.1f}s ({epoch_time/60:.1f}min)")
        
        # Save checkpoints (only on rank 0)
        if rank == 0:
            # Create checkpoint data
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict(),  # Save unwrapped model
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_losses,
                'val_loss': val_losses,
                'scale_factor': model.module.scale_factor,
                'embed_dim': model.module.embed_dim,
                'model_type': 'kl_vae_ddpm'
            }
            
            # Save periodic checkpoint
            if (epoch + 1) % args.save_every == 0:
                checkpoint_path = os.path.join(
                    args.checkpoint_dir,
                    f'kl_vae_epoch_{epoch+1}.pt'
                )
                torch.save(checkpoint, checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")
                
            # Save best model (delete old best)
            if val_losses['loss'] < best_val_loss - args.min_delta:
                best_val_loss = val_losses['loss']
                
                # 删除旧的最佳模型
                old_best_path = os.path.join(args.checkpoint_dir, 'kl_vae_best.pt')
                if os.path.exists(old_best_path):
                    os.remove(old_best_path)
                    print(f"Removed old best model: {old_best_path}")
                
                # 保存新的最佳模型
                checkpoint['is_best'] = True
                checkpoint['best_val_loss'] = best_val_loss
                best_path = os.path.join(args.checkpoint_dir, 'kl_vae_best.pt')
                torch.save(checkpoint, best_path)
                print(f"✅ Saved new best model (val_loss: {best_val_loss:.4f}): {best_path}")
                
                # 复制当前epoch的可视化作为最佳模型的可视化
                current_sample_path = os.path.join(args.checkpoint_dir, f'samples_epoch_{epoch}.png')
                best_sample_path = os.path.join(args.checkpoint_dir, 'best_samples.png')
                if os.path.exists(current_sample_path):
                    import shutil
                    shutil.copy2(current_sample_path, best_sample_path)
                    print(f"✅ Saved best model visualization: {best_sample_path}")
                
                # 重置patience计数器
                patience_counter = 0
            else:
                # 没有改进，增加patience计数器
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"\n⚠️ 早停: {args.patience}个epoch没有改进")
                    print(f"最佳验证损失: {best_val_loss:.4f}")
                    break
        
        # 确保所有进程同步
        if dist.is_initialized():
            # 在barrier之前打印状态（仅rank 0）
            if rank == 0:
                print(f"  Waiting for all GPUs to complete epoch {epoch}...")
            dist.barrier()
            if rank == 0:
                print(f"  All GPUs synchronized, continuing to next epoch\n")
        
        # Check if we should stop (broadcast from rank 0)
        if dist.is_initialized():
            should_stop = torch.tensor([patience_counter >= args.patience], device=device)
            dist.broadcast(should_stop, src=0)
            if should_stop[0]:
                if rank == 0:
                    print(f"Early stopping triggered on all GPUs")
                break
    
    # 训练总结 (only on rank 0)
    if rank == 0:
        print("\n" + "="*60)
        print("训练总结")
        print("="*60)
        print(f"总训练轮数: {epoch + 1 - start_epoch}")
        print(f"最佳验证损失: {best_val_loss:.4f}")
        print(f"DDP训练：使用 {world_size} GPUs")
        print(f"\n最佳模型保存在: {args.checkpoint_dir}/kl_vae_best.pt")
        print(f"可视化对比图保存在: {args.checkpoint_dir}/")
        print("="*60)
        
        print("\n✅ DDP训练完成!")
    
    # Cleanup
    cleanup_ddp()


def main():
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=12)  # Per GPU batch size for T4
    parser.add_argument('--num_workers', type=int, default=4)  # 增加数据加载线程
    
    # 支持torchrun启动
    parser.add_argument('--local_rank', type=int, default=-1, 
                       help='Local rank for torchrun')
    
    # Model
    parser.add_argument('--embed_dim', type=int, default=4)  # SD standard
    parser.add_argument('--ch', type=int, default=128)
    parser.add_argument('--ch_mult', type=int, nargs='+', default=[1, 2, 2, 4])
    parser.add_argument('--scale_factor', type=float, default=0.18215)
    parser.add_argument('--img_size', type=int, default=256, 
                       help='Image size (reduce to save memory)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=4.5e-6)  # SD's learning rate
    parser.add_argument('--kl_weight', type=float, default=1e-6)  # Very small KL weight
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--use_fp16', action='store_true',
                       help='Use FP16 mixed precision training (saves ~50% memory)')
    
    # Perceptual loss settings
    parser.add_argument('--use_perceptual', action='store_true', 
                       help='Use perceptual loss after warmup')
    parser.add_argument('--perceptual_weight', type=float, default=0.01,
                       help='Weight for perceptual loss')
    parser.add_argument('--perceptual_start_epoch', type=int, default=10,
                       help='Epoch to start using perceptual loss')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--min_delta', type=float, default=1e-4, 
                       help='Minimum improvement for early stopping')
    
    # Checkpoints
    parser.add_argument('--checkpoint_dir', type=str, default='vae_checkpoints')
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    
    # DDP
    parser.add_argument('--world_size', type=int, default=-1,
                       help='Number of GPUs to use (-1 for all available)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='Number of gradient accumulation steps (simulate larger batch size)')
    
    args = parser.parse_args()
    
    # 检查是否由torchrun启动（通过环境变量检测）
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # torchrun启动模式
        print("Detected torchrun launch")
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        print(f"Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}")
        args.world_size = world_size
        
        # 直接调用train_ddp
        train_ddp(rank, world_size, args)
    elif args.local_rank != -1:
        # 旧版本torchrun兼容
        print("Detected legacy torchrun launch")
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        print(f"Rank: {rank}, World Size: {world_size}")
        args.world_size = world_size
        
        # 直接调用train_ddp
        train_ddp(rank, world_size, args)
    else:
        # 传统mp.spawn启动模式
        # Determine world size
        if args.world_size == -1:
            args.world_size = torch.cuda.device_count()
        
        if args.world_size > 1:
            print(f"Starting DDP training with {args.world_size} GPUs...")
            # Launch DDP training
            mp.spawn(train_ddp, args=(args.world_size, args), nprocs=args.world_size, join=True)
        else:
            print("Single GPU detected. Please use train_kl_vae.py for single GPU training.")
            print("Or set --world_size to the number of GPUs you want to use.")


if __name__ == '__main__':
    main()
