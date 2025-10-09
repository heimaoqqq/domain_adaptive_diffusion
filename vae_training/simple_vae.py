"""
Simple VAE for DDPM - Clean implementation without external dependencies
专门为DDPM设计的简单VAE，无需复杂依赖
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import math


class ResBlock(nn.Module):
    """Simple residual block"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        # Skip connection
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.norm1(F.silu(self.conv1(x)))
        x = self.norm2(F.silu(self.conv2(x)))
        return x + residual


class SimpleEncoder(nn.Module):
    """Simple encoder for 256x256 -> 16x16 (16x downsampling)"""
    def __init__(self, in_channels: int = 3, latent_channels: int = 32):
        super().__init__()
        
        # Encoder path: 256 -> 128 -> 64 -> 32 -> 16
        self.conv_in = nn.Conv2d(in_channels, 64, 3, padding=1)
        
        self.down1 = nn.Sequential(
            ResBlock(64, 128),
            nn.Conv2d(128, 128, 3, stride=2, padding=1)  # 256 -> 128
        )
        
        self.down2 = nn.Sequential(
            ResBlock(128, 256),
            nn.Conv2d(256, 256, 3, stride=2, padding=1)  # 128 -> 64
        )
        
        self.down3 = nn.Sequential(
            ResBlock(256, 512),
            nn.Conv2d(512, 512, 3, stride=2, padding=1)  # 64 -> 32
        )
        
        self.down4 = nn.Sequential(
            ResBlock(512, 512),
            nn.Conv2d(512, 512, 3, stride=2, padding=1)  # 32 -> 16
        )
        
        # Output mean and log variance
        self.conv_out = nn.Conv2d(512, latent_channels * 2, 3, padding=1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv_in(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        
        # Split into mean and log variance
        stats = self.conv_out(x)
        mean, logvar = torch.chunk(stats, 2, dim=1)
        return mean, logvar


class SimpleDecoder(nn.Module):
    """Simple decoder for 16x16 -> 256x256 (16x upsampling)"""
    def __init__(self, latent_channels: int = 32, out_channels: int = 3):
        super().__init__()
        
        self.conv_in = nn.Conv2d(latent_channels, 512, 3, padding=1)
        
        self.up1 = nn.Sequential(
            ResBlock(512, 512),
            nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1)  # 16 -> 32
        )
        
        self.up2 = nn.Sequential(
            ResBlock(512, 256),
            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1)  # 32 -> 64
        )
        
        self.up3 = nn.Sequential(
            ResBlock(256, 128),
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1)  # 64 -> 128
        )
        
        self.up4 = nn.Sequential(
            ResBlock(128, 64),
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1)  # 128 -> 256
        )
        
        self.conv_out = nn.Conv2d(64, out_channels, 3, padding=1)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(z)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return self.conv_out(x)


class SimpleVAE(nn.Module):
    """
    Simple VAE for DDPM training
    - 256x256 images -> 16x16 latents (16x downsampling)
    - 32 latent channels
    - Built-in scale factor for DDPM compatibility
    """
    def __init__(self, 
                 latent_channels: int = 32,
                 scale_factor: float = 0.18215,  # Standard for SD/DDPM
                 image_channels: int = 3):
        super().__init__()
        
        self.encoder = SimpleEncoder(image_channels, latent_channels)
        self.decoder = SimpleDecoder(latent_channels, image_channels)
        
        self.latent_channels = latent_channels
        self.scale_factor = scale_factor
        
        # Register scale factor as buffer so it's saved with the model
        self.register_buffer('_scale_factor', torch.tensor(scale_factor))
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images to latents"""
        mean, logvar = self.encoder(x)
        
        # Sample from the distribution
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        
        # Scale for DDPM (ensures unit variance)
        return z * self.scale_factor
    
    def encode_images(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for compatibility"""
        return self.encode(x)
        
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latents to images"""
        # Remove scale factor
        z = z / self.scale_factor
        return self.decoder(z)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with loss computation"""
        # Encode
        mean, logvar = self.encoder(x)
        
        # Sample
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        
        # Decode
        recon = self.decoder(z)
        
        # Compute losses
        recon_loss = F.mse_loss(recon, x)
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / x.shape[0]
        
        return {
            'recon': recon,
            'z': z * self.scale_factor,  # Scaled latent
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'loss': recon_loss + 0.001 * kl_loss  # Beta-VAE weight
        }
        
    def save(self, path: str):
        """Save model with metadata"""
        torch.save({
            'state_dict': self.state_dict(),
            'scale_factor': self.scale_factor,
            'latent_channels': self.latent_channels,
            'model_type': 'simple_vae_ddpm'
        }, path)
        
    @classmethod
    def load(cls, path: str, device: str = 'cuda'):
        """Load model from checkpoint"""
        checkpoint = torch.load(path, map_location=device)
        
        model = cls(
            latent_channels=checkpoint.get('latent_channels', 32),
            scale_factor=checkpoint.get('scale_factor', 0.18215)
        )
        
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        model.eval()
        
        return model
