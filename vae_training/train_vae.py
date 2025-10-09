"""
Train a simple VAE for DDPM
快速训练一个专门为DDPM设计的VAE
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
from typing import Dict, List
import numpy as np

from simple_vae import SimpleVAE


class MicroDopplerDataset(Dataset):
    """Simple dataset for micro-Doppler images"""
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


def train_epoch(model: SimpleVAE, 
                dataloader: DataLoader, 
                optimizer: torch.optim.Optimizer,
                device: str) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    
    for batch in tqdm(dataloader, desc='Training'):
        batch = batch.to(device)
        
        # Forward pass
        output = model(batch)
        loss = output['loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Record losses
        total_loss += loss.item()
        total_recon_loss += output['recon_loss'].item()
        total_kl_loss += output['kl_loss'].item()
        
    n_batches = len(dataloader)
    return {
        'loss': total_loss / n_batches,
        'recon_loss': total_recon_loss / n_batches,
        'kl_loss': total_kl_loss / n_batches
    }


@torch.no_grad()
def validate(model: SimpleVAE, 
             dataloader: DataLoader, 
             device: str,
             save_samples: bool = False,
             save_path: str = None) -> Dict[str, float]:
    """Validate model"""
    model.eval()
    
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    
    # Save first batch samples
    first_batch = None
    first_recon = None
    
    for i, batch in enumerate(tqdm(dataloader, desc='Validation')):
        batch = batch.to(device)
        
        # Forward pass
        output = model(batch)
        
        # Record losses
        total_loss += output['loss'].item()
        total_recon_loss += output['recon_loss'].item()
        total_kl_loss += output['kl_loss'].item()
        
        # Save first batch for visualization
        if i == 0 and save_samples:
            first_batch = batch[:8].cpu()
            first_recon = output['recon'][:8].cpu()
            
    n_batches = len(dataloader)
    
    # Save sample reconstructions
    if save_samples and first_batch is not None:
        fig, axes = plt.subplots(2, 8, figsize=(16, 4))
        for i in range(8):
            # Original
            axes[0, i].imshow(first_batch[i].permute(1, 2, 0).clamp(0, 1))
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Original')
                
            # Reconstruction
            axes[1, i].imshow(first_recon[i].permute(1, 2, 0).clamp(0, 1))
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Reconstruction')
                
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    return {
        'loss': total_loss / n_batches,
        'recon_loss': total_recon_loss / n_batches,
        'kl_loss': total_kl_loss / n_batches
    }


def analyze_latents(model: SimpleVAE, dataloader: DataLoader, device: str):
    """Analyze latent statistics for scale factor"""
    model.eval()
    
    all_latents = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Analyzing latents'):
            batch = batch.to(device)
            # Get unscaled latents
            mean, logvar = model.encoder(batch)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mean + eps * std
            all_latents.append(z.cpu())
            
    all_latents = torch.cat(all_latents, dim=0)
    
    print("\nLatent statistics (before scaling):")
    print(f"  Mean: {all_latents.mean().item():.4f}")
    print(f"  Std: {all_latents.std().item():.4f}")
    print(f"  Min: {all_latents.min().item():.4f}")
    print(f"  Max: {all_latents.max().item():.4f}")
    
    # Suggest scale factor
    current_std = all_latents.std().item()
    suggested_scale = 1.0 / current_std
    print(f"\nSuggested scale_factor for unit variance: {suggested_scale:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train VAE for DDPM')
    
    # Data
    parser.add_argument('--data_dir', type=str, 
                       default='D:\\Ysj\\新建文件夹\\VA-VAE\\dataset\\organized_gait_dataset\\Normal_line',
                       help='Path to training data')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=0)
    
    # Model
    parser.add_argument('--latent_channels', type=int, default=32)
    parser.add_argument('--scale_factor', type=float, default=0.18215,
                       help='Scale factor for DDPM (use analyze mode to find optimal)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--val_split', type=float, default=0.1)
    
    # Checkpoints
    parser.add_argument('--checkpoint_dir', type=str, default='vae_checkpoints')
    parser.add_argument('--save_every', type=int, default=5)
    
    # Mode
    parser.add_argument('--analyze_only', action='store_true',
                       help='Only analyze latent statistics, no training')
    
    args = parser.parse_args()
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # Dataset
    full_dataset = MicroDopplerDataset(args.data_dir, transform=transform)
    
    # Split into train/val
    n_val = int(len(full_dataset) * args.val_split)
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val]
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
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
    model = SimpleVAE(
        latent_channels=args.latent_channels,
        scale_factor=args.scale_factor
    ).to(device)
    
    print(f"\nModel configuration:")
    print(f"  Latent channels: {args.latent_channels}")
    print(f"  Scale factor: {args.scale_factor}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Analyze mode
    if args.analyze_only:
        analyze_latents(model, train_loader, device)
        return
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        
        # Train
        train_losses = train_epoch(model, train_loader, optimizer, device)
        print(f"Train - Loss: {train_losses['loss']:.4f}, "
              f"Recon: {train_losses['recon_loss']:.4f}, "
              f"KL: {train_losses['kl_loss']:.4f}")
        
        # Validate
        save_samples = (epoch + 1) % args.save_every == 0
        sample_path = os.path.join(args.checkpoint_dir, f'samples_epoch_{epoch+1}.png')
        
        val_losses = validate(model, val_loader, device, 
                            save_samples=save_samples,
                            save_path=sample_path)
        
        print(f"Val - Loss: {val_losses['loss']:.4f}, "
              f"Recon: {val_losses['recon_loss']:.4f}, "
              f"KL: {val_losses['kl_loss']:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(
                args.checkpoint_dir, 
                f'vae_epoch_{epoch+1}.pt'
            )
            model.save(checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
            
        # Save best model
        if val_losses['loss'] < best_val_loss:
            best_val_loss = val_losses['loss']
            best_path = os.path.join(args.checkpoint_dir, 'best_vae.pt')
            model.save(best_path)
            print(f"Saved best model: {best_path}")
            
    # Final latent analysis
    print("\n=== Final latent analysis ===")
    analyze_latents(model, train_loader, device)
    
    print("\n✅ Training complete!")
    print(f"Best model saved at: {os.path.join(args.checkpoint_dir, 'best_vae.pt')}")


if __name__ == '__main__':
    main()
