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


class MicroDopplerDataset(Dataset):
    """Dataset for micro-Doppler spectrograms"""
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Collect all images
        self.images = []
        for user_dir in sorted(self.root_dir.iterdir()):
            if user_dir.is_dir() and user_dir.name.startswith('ID_'):
                for img_path in user_dir.glob('*.jpg'):
                    self.images.append(img_path)
                    
        print(f"Found {len(self.images)} images")
        
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
                device: str) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    total_rec_loss = 0
    total_kl_loss = 0
    
    progress_bar = tqdm(dataloader, desc='Training')
    for batch in progress_bar:
        batch = batch.to(device)
        
        # Compute loss
        losses = model.get_loss(batch, kl_weight=kl_weight)
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
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'rec': f"{losses['rec_loss'].item():.4f}",
            'kl': f"{losses['kl_loss'].item():.4f}"
        })
        
    n_batches = len(dataloader)
    return {
        'loss': total_loss / n_batches,
        'rec_loss': total_rec_loss / n_batches,
        'kl_loss': total_kl_loss / n_batches
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
    
    # Save reconstructions with improved visualization
    if save_samples and first_batch is not None:
        fig = plt.figure(figsize=(20, 8))
        
        # 创建3行显示：原图、重建、差异
        n_samples = min(8, first_batch.shape[0])
        
        for i in range(n_samples):
            # 原图
            ax1 = plt.subplot(3, n_samples, i + 1)
            img_orig = first_batch[i].permute(1, 2, 0).clamp(0, 1)
            ax1.imshow(img_orig)
            ax1.axis('off')
            if i == 0:
                ax1.set_title('Original', fontsize=12)
                
            # 重建
            ax2 = plt.subplot(3, n_samples, n_samples + i + 1)
            img_recon = first_recon[i].permute(1, 2, 0).clamp(0, 1)
            ax2.imshow(img_recon)
            ax2.axis('off')
            if i == 0:
                ax2.set_title('Reconstruction', fontsize=12)
                
            # 差异图
            ax3 = plt.subplot(3, n_samples, 2 * n_samples + i + 1)
            diff = torch.abs(img_orig - img_recon)
            # 放大差异以便观察
            diff_vis = (diff * 5).clamp(0, 1)
            ax3.imshow(diff_vis)
            ax3.axis('off')
            if i == 0:
                ax3.set_title('Difference (5x)', fontsize=12)
        
        # 添加损失信息
        fig.suptitle(f'Rec Loss: {total_rec_loss/n_batches:.4f}, KL Loss: {total_kl_loss/n_batches:.4f}', 
                     fontsize=14)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 保存损失曲线（如果有历史数据）
        if hasattr(model, 'loss_history'):
            plt.figure(figsize=(10, 6))
            plt.plot(model.loss_history['train'], label='Train Loss')
            plt.plot(model.loss_history['val'], label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Progress')
            plt.legend()
            plt.grid(True)
            loss_path = save_path.replace('.png', '_loss.png')
            plt.savefig(loss_path, dpi=150, bbox_inches='tight')
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
    parser.add_argument('--batch_size', type=int, default=8)  # Smaller for 4-channel latent
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
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--min_delta', type=float, default=1e-4, 
                       help='Minimum improvement for early stopping')
    
    # Checkpoints
    parser.add_argument('--checkpoint_dir', type=str, default='vae_checkpoints')
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--visualize_every', type=int, default=5, 
                       help='Generate visualization every N epochs')
    
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
    
    # Training
    best_val_loss = float('inf')
    train_losses_history = []
    val_losses_history = []
    patience_counter = 0
    
    # 为模型添加损失历史（用于可视化）
    model.loss_history = {'train': [], 'val': []}
    
    print(f"\n训练配置:")
    print(f"  - 可视化频率: 每{args.visualize_every}个epoch")
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
        
        # Train
        train_losses = train_epoch(model, train_loader, optimizer, kl_weight, device)
        print(f"Train - Loss: {train_losses['loss']:.4f}, "
              f"Rec: {train_losses['rec_loss']:.4f}, "
              f"KL: {train_losses['kl_loss']:.4f}")
        
        # Validate
        save_samples = (epoch + 1) % args.visualize_every == 0
        sample_path = os.path.join(args.checkpoint_dir, f'samples_epoch_{epoch+1}.png')
        
        val_losses = validate(model, val_loader, kl_weight, device,
                            save_samples=save_samples,
                            save_path=sample_path)
        
        # 更新损失历史
        model.loss_history['train'].append(train_losses['loss'])
        model.loss_history['val'].append(val_losses['loss'])
        
        print(f"Val - Loss: {val_losses['loss']:.4f}, "
              f"Rec: {val_losses['rec_loss']:.4f}, "
              f"KL: {val_losses['kl_loss']:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
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
    print(f"总训练轮数: {len(model.loss_history['train'])}")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"最终训练损失: {model.loss_history['train'][-1]:.4f}")
    print(f"最终验证损失: {model.loss_history['val'][-1]:.4f}")
    print(f"\n最佳模型保存在: {args.checkpoint_dir}/kl_vae_best.pt")
    print(f"可视化保存在: {args.checkpoint_dir}/best_samples.png")
    print("="*60)
    
    print("\n✅ 训练完成!")


if __name__ == '__main__':
    main()
