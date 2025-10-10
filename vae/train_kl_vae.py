"""
Train KL-VAE for DDPM
基于Stable Diffusion架构的VAE训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import argparse
import os
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict
import numpy as np

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
        
        print(f"Searching for images in: {self.root_dir}")
        print(f"Directory exists: {self.root_dir.exists()}")
        
        # List all subdirectories for debugging
        if self.root_dir.exists():
            subdirs = list(self.root_dir.iterdir())
            print(f"Found {len(subdirs)} items in root directory")
            
            # Check if this is a single gait type directory with ID_* folders
            id_dirs = [d for d in subdirs if d.is_dir() and d.name.startswith('ID_')]
            print(f"Found {len(id_dirs)} ID_* directories")
            
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
                    
                    print(f"  {user_dir.name}: {len(jpg_files)} images")
                    self.images.extend(jpg_files)
            else:
                # Check if root contains gait type subdirectories
                gait_dirs = [d for d in subdirs 
                           if d.is_dir() and '_' in d.name and not d.name.startswith('.')]
                print(f"Found {len(gait_dirs)} gait type directories")
                
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
                    
        print(f"Total images found: {len(self.images)}")
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image


def train_epoch(model: KL_VAE, 
                dataloader: DataLoader, 
                optimizer: torch.optim.Optimizer,
                kl_weight: float,
                device: str,
                perceptual_loss_fn=None) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    total_rec_loss = 0
    total_kl_loss = 0
    total_perceptual_loss = 0
    
    progress_bar = tqdm(dataloader, desc='Training')
    for batch in progress_bar:
        batch = batch.to(device)
        
        # Compute loss
        losses = model.get_loss(batch, kl_weight=kl_weight, perceptual_loss_fn=perceptual_loss_fn)
        loss = losses['loss']
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        # Record losses
        total_loss += loss.item()
        total_rec_loss += losses['rec_loss'].item()
        total_kl_loss += losses['kl_loss'].item()
        if 'perceptual_loss' in losses:
            total_perceptual_loss += losses['perceptual_loss'].item()
        
        # Update progress bar
        postfix_dict = {
            'loss': f"{loss.item():.4f}",
            'rec': f"{losses['rec_loss'].item():.4f}",
            'kl': f"{losses['kl_loss'].item():.4f}"
        }
        if 'perceptual_loss' in losses and losses['perceptual_loss'].item() > 0:
            postfix_dict['perc'] = f"{losses['perceptual_loss'].item():.4f}"
        progress_bar.set_postfix(postfix_dict)
        
    n_batches = len(dataloader)
    return {
        'loss': total_loss / n_batches,
        'rec_loss': total_rec_loss / n_batches,
        'kl_loss': total_kl_loss / n_batches,
        'perceptual_loss': total_perceptual_loss / n_batches if perceptual_loss_fn else 0.0
    }


@torch.no_grad()
def validate(model: KL_VAE, 
             dataloader: DataLoader, 
             kl_weight: float,
             device: str,
             save_samples: bool = False,
             save_path: str = None) -> Dict[str, float]:
    """Validate model"""
    model.eval()
    
    total_loss = 0
    total_rec_loss = 0
    total_kl_loss = 0
    
    # For visualization
    first_batch = None
    first_recon = None
    
    for i, batch in enumerate(tqdm(dataloader, desc='Validation')):
        batch = batch.to(device)
        
        # Forward pass
        recon, posterior = model(batch)
        
        # Compute losses
        rec_loss = nn.functional.mse_loss(batch, recon)
        kl_loss = posterior.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        loss = rec_loss + kl_weight * kl_loss
        
        # Record losses
        total_loss += loss.item()
        total_rec_loss += rec_loss.item()
        total_kl_loss += kl_loss.item()
        
        # Save first batch for visualization
        if i == 0 and save_samples:
            first_batch = batch[:8].cpu()
            first_recon = recon[:8].cpu()
            
    n_batches = len(dataloader)
    
    # Save reconstructions - simple comparison
    if save_samples and first_batch is not None:
        fig = plt.figure(figsize=(16, 6))
        
        # 只显示原图和重建图对比
        n_samples = min(8, first_batch.shape[0])
        
        for i in range(n_samples):
            # 原图
            ax1 = plt.subplot(2, n_samples, i + 1)
            img_orig = first_batch[i].permute(1, 2, 0).clamp(0, 1)
            ax1.imshow(img_orig)
            ax1.axis('off')
            if i == 0:
                ax1.set_title('Original', fontsize=14, fontweight='bold')
                
            # 重建图
            ax2 = plt.subplot(2, n_samples, n_samples + i + 1)
            img_recon = first_recon[i].permute(1, 2, 0).clamp(0, 1)
            ax2.imshow(img_recon)
            ax2.axis('off')
            if i == 0:
                ax2.set_title('Reconstruction', fontsize=14, fontweight='bold')
        
        # 添加epoch和损失信息作为标题
        epoch_num = save_path.split('epoch_')[1].split('.')[0] if 'epoch_' in save_path else 'best'
        fig.suptitle(f'Epoch {epoch_num} - Rec Loss: {total_rec_loss/n_batches:.4f}', 
                     fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return {
        'loss': total_loss / n_batches,
        'rec_loss': total_rec_loss / n_batches,
        'kl_loss': total_kl_loss / n_batches
    }


def analyze_latents(model: KL_VAE, dataloader: DataLoader, device: str):
    """Analyze latent space statistics"""
    model.eval()
    
    all_latents = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Analyzing latents'):
            batch = batch.to(device)
            # Get latents
            z = model.encode_images(batch)
            all_latents.append(z.cpu())
            
    all_latents = torch.cat(all_latents, dim=0)
    
    # Unscaled statistics
    unscaled_latents = all_latents / model.scale_factor
    
    print("\nLatent statistics:")
    print(f"Shape: {all_latents.shape}")
    print("\nWith scale_factor = {:.5f}:".format(model.scale_factor))
    print(f"  Mean: {all_latents.mean().item():.4f}")
    print(f"  Std: {all_latents.std().item():.4f}")
    print(f"  Min: {all_latents.min().item():.4f}")
    print(f"  Max: {all_latents.max().item():.4f}")
    
    print("\nWithout scale_factor:")
    print(f"  Mean: {unscaled_latents.mean().item():.4f}")
    print(f"  Std: {unscaled_latents.std().item():.4f}")
    
    # Suggest scale factor
    current_std = unscaled_latents.std().item()
    suggested_scale = 1.0 / current_std
    print(f"\nFor unit variance, suggested scale_factor: {suggested_scale:.5f}")
    print(f"Current scale_factor: {model.scale_factor:.5f}")


def main():
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=18)  # Reduced for 16GB GPU
    parser.add_argument('--num_workers', type=int, default=0)
    
    # Model
    parser.add_argument('--embed_dim', type=int, default=4)  # SD standard
    parser.add_argument('--ch', type=int, default=128)
    parser.add_argument('--ch_mult', type=int, nargs='+', default=[1, 2, 2, 4])
    parser.add_argument('--scale_factor', type=float, default=0.18215)
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=4.5e-6)  # SD's learning rate
    parser.add_argument('--kl_weight', type=float, default=1e-6)  # Very small KL weight
    parser.add_argument('--warmup_epochs', type=int, default=5)
    
    # Perceptual loss settings
    parser.add_argument('--use_perceptual', action='store_true', 
                       help='Use perceptual loss after warmup')
    parser.add_argument('--perceptual_weight', type=float, default=0.05,
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
    
    # Mode
    parser.add_argument('--analyze_only', action='store_true')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
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
    
    print(f"\nDataset: {len(full_dataset)} images")
    print(f"Train: {n_train}, Val: {n_val}")
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
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
    
    print(f"\nModel configuration:")
    print(f"  Latent channels: {args.embed_dim}")
    print(f"  Channel multipliers: {args.ch_mult}")
    print(f"  Downsampling: {2**len(args.ch_mult)}x")
    print(f"  Scale factor: {args.scale_factor}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Analyze mode
    if args.analyze_only:
        if args.resume:
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from {args.resume}")
        analyze_latents(model, train_loader, device)
        return
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.9))
    
    # Resume
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resumed from epoch {start_epoch}")
    
    # Initialize perceptual loss if requested
    perceptual_loss_fn = None
    if args.use_perceptual:
        if not PERCEPTUAL_AVAILABLE:
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
            print(f"\n感知损失配置:")
            print(f"  - 将在第 {args.perceptual_start_epoch} epoch后启用")
            print(f"  - 感知损失权重: {args.perceptual_weight}")
    
    # Training
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"\n训练配置:")
    print(f"  - 每个epoch生成可视化对比图")
    print(f"  - 早停patience: {args.patience}")
    print(f"  - 最小改进阈值: {args.min_delta}")
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        
        # KL weight warmup
        if epoch < args.warmup_epochs:
            kl_weight = args.kl_weight * (epoch + 1) / args.warmup_epochs
        else:
            kl_weight = args.kl_weight
            
        print(f"KL weight: {kl_weight:.2e}")
        
        # Decide whether to use perceptual loss
        current_perceptual_fn = None
        if args.use_perceptual and epoch >= args.perceptual_start_epoch:
            current_perceptual_fn = perceptual_loss_fn
            print(f"感知损失: 已启用 (权重={args.perceptual_weight})")
        elif args.use_perceptual:
            epochs_until_perceptual = args.perceptual_start_epoch - epoch
            print(f"感知损失: {epochs_until_perceptual} epoch后启用")
        
        # Train
        train_losses = train_epoch(model, train_loader, optimizer, kl_weight, device, current_perceptual_fn)
        print(f"Train - Loss: {train_losses['loss']:.4f}, "
              f"Rec: {train_losses['rec_loss']:.4f}, "
              f"KL: {train_losses['kl_loss']:.4f}", end='')
        if train_losses.get('perceptual_loss', 0) > 0:
            print(f", Perceptual: {train_losses['perceptual_loss']:.4f}")
        else:
            print()
        
        # Validate - 每个epoch都生成可视化
        sample_path = os.path.join(args.checkpoint_dir, f'samples_epoch_{epoch+1}.png')
        
        val_losses = validate(model, val_loader, kl_weight, device,
                            save_samples=True,  # 每个epoch都保存
                            save_path=sample_path)
        
        print(f"Val - Loss: {val_losses['loss']:.4f}, "
              f"Rec: {val_losses['rec_loss']:.4f}, "
              f"KL: {val_losses['kl_loss']:.4f}")
        
        # Create checkpoint data (always create it for potential best model saving)
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_losses,
            'val_loss': val_losses,
            'scale_factor': model.scale_factor,
            'embed_dim': model.embed_dim,
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
        if val_losses['loss'] < best_val_loss:
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
            
            # 保存最佳模型的可视化
            best_sample_path = os.path.join(args.checkpoint_dir, 'best_samples.png')
            _ = validate(model, val_loader, kl_weight, device,
                        save_samples=True,
                        save_path=best_sample_path)
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
    
    # Final analysis
    print("\n=== Final latent analysis ===")
    analyze_latents(model, train_loader, device)
    
    # 训练总结
    print("\n" + "="*60)
    print("训练总结")
    print("="*60)
    print(f"总训练轮数: {epoch + 1 - start_epoch}")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"\n最佳模型保存在: {args.checkpoint_dir}/kl_vae_best.pt")
    print(f"可视化对比图保存在: {args.checkpoint_dir}/")
    print("="*60)
    
    print("\n✅ 训练完成!")


if __name__ == '__main__':
    main()
