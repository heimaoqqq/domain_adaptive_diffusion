"""
Modern VQ-VAE with FSQ (Finite Scalar Quantization)
基于最新的FSQ技术，更简单更稳定
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List
import numpy as np
from einops import rearrange


class ResnetBlock(nn.Module):
    """Residual block with GroupNorm"""
    def __init__(self, in_channels: int, out_channels: int = None, dropout: float = 0.0):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class FSQuantizer(nn.Module):
    """
    Finite Scalar Quantization (FSQ) - ICLR 2024
    更简单的量化方法，不需要学习codebook
    """
    def __init__(self, levels: List[int]):
        super().__init__()
        self.levels = levels
        self.n_codes = np.prod(levels)
        self.d = len(levels)
        
        # 预计算offset和scale
        self.register_buffer('offset', torch.tensor([l // 2 for l in levels], dtype=torch.float32))
        self.register_buffer('scale', torch.tensor([l - 1 for l in levels], dtype=torch.float32))
        
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        # Normalize to [-1, 1]
        z_normalized = torch.tanh(z)
        
        # Scale to [0, L-1]
        z_scaled = (z_normalized + 1) * 0.5 * self.scale.view(1, -1, 1, 1)
        
        # Round to nearest integer
        z_quantized = torch.round(z_scaled)
        
        # Straight-through estimator
        z_quantized = z_scaled + (z_quantized - z_scaled).detach()
        
        # Scale back to [-1, 1]
        z_out = z_quantized / self.scale.view(1, -1, 1, 1) * 2 - 1
        
        # Compute indices for discrete diffusion
        indices = z_quantized.long()
        
        return z_out, {
            'indices': indices,
            'quantized': z_quantized,
        }


class VectorQuantizer(nn.Module):
    """
    Classic VQ-VAE quantizer with improved training stability
    """
    def __init__(self, n_embed: int, embed_dim: int, beta: float = 0.25):
        super().__init__()
        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.beta = beta
        
        self.embedding = nn.Embedding(self.n_embed, self.embed_dim)
        self.embedding.weight.data.uniform_(-1/self.n_embed, 1/self.n_embed)
        
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        # z: [B, C, H, W]
        z_flattened = rearrange(z, 'b c h w -> b h w c')
        
        # Compute distances
        d = torch.sum(z_flattened ** 2, dim=-1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - \
            2 * torch.einsum('bhwc,nc->bhwn', z_flattened, self.embedding.weight)
        
        # Find closest embeddings
        indices = torch.argmin(d, dim=-1)
        z_q = self.embedding(indices)
        
        # Compute loss
        loss = self.beta * torch.mean((z_q.detach() - z_flattened) ** 2) + \
               torch.mean((z_q - z_flattened.detach()) ** 2)
        
        # Straight through estimator
        z_q = z_flattened + (z_q - z_flattened).detach()
        z_q = rearrange(z_q, 'b h w c -> b c h w')
        
        return z_q, {
            'indices': indices,
            'loss': loss,
            'perplexity': self._compute_perplexity(indices)
        }
        
    def _compute_perplexity(self, indices):
        # Calculate perplexity
        encodings = F.one_hot(indices.reshape(-1), self.n_embed).float()
        avg_probs = encodings.mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return perplexity


class Encoder(nn.Module):
    """VQ-VAE Encoder"""
    def __init__(self, 
                 in_channels: int = 3,
                 ch: int = 128,
                 ch_mult: Tuple[int] = (1, 2, 4, 8),
                 num_res_blocks: int = 2,
                 z_channels: int = 256,
                 dropout: float = 0.0):
        super().__init__()
        
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, padding=1)
        
        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        block_in = ch
        for i, mult in enumerate(ch_mult):
            block_out = ch * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResnetBlock(block_in, block_out, dropout))
                block_in = block_out
            if i < len(ch_mult) - 1:
                self.down_blocks.append(nn.Conv2d(block_in, block_in, 3, stride=2, padding=1))
        
        # Middle
        self.mid_block1 = ResnetBlock(block_in, block_in, dropout)
        self.mid_block2 = ResnetBlock(block_in, block_in, dropout)
        
        # Output
        self.norm_out = nn.GroupNorm(32, block_in)
        self.conv_out = nn.Conv2d(block_in, z_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        h = self.conv_in(x)
        
        # Downsample
        for block in self.down_blocks:
            if isinstance(block, nn.Conv2d):
                h = block(h)
            else:
                h = block(h)
        
        # Middle
        h = self.mid_block1(h)
        h = self.mid_block2(h)
        
        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        return h


class Decoder(nn.Module):
    """VQ-VAE Decoder"""
    def __init__(self,
                 out_channels: int = 3,
                 ch: int = 128,
                 ch_mult: Tuple[int] = (1, 2, 4, 8),
                 num_res_blocks: int = 2,
                 z_channels: int = 256,
                 dropout: float = 0.0):
        super().__init__()
        
        # Input
        block_in = ch * ch_mult[-1]
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, padding=1)
        
        # Middle
        self.mid_block1 = ResnetBlock(block_in, block_in, dropout)
        self.mid_block2 = ResnetBlock(block_in, block_in, dropout)
        
        # Upsampling blocks
        self.up_blocks = nn.ModuleList()
        for i, mult in enumerate(reversed(ch_mult)):
            block_out = ch * mult
            for _ in range(num_res_blocks):
                self.up_blocks.append(ResnetBlock(block_in, block_out, dropout))
                block_in = block_out
            if i < len(ch_mult) - 1:
                self.up_blocks.append(nn.ConvTranspose2d(block_in, block_in, 4, stride=2, padding=1))
        
        # Output
        self.norm_out = nn.GroupNorm(32, block_in)
        self.conv_out = nn.Conv2d(block_in, out_channels, kernel_size=3, padding=1)
        
    def forward(self, z):
        h = self.conv_in(z)
        
        # Middle
        h = self.mid_block1(h)
        h = self.mid_block2(h)
        
        # Upsample
        for block in self.up_blocks:
            if isinstance(block, nn.ConvTranspose2d):
                h = block(h)
            else:
                h = block(h)
        
        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        return h


class VQ_VAE(nn.Module):
    """
    Modern VQ-VAE with choice of quantization method
    支持FSQ和经典VQ两种量化方法
    """
    def __init__(self,
                 quantizer_type: str = 'fsq',  # 'fsq' or 'vq'
                 ch_mult: Tuple[int] = (1, 2, 4, 8),
                 z_channels: int = 8,  # For FSQ
                 n_embed: int = 1024,  # For VQ
                 fsq_levels: List[int] = None):  # e.g., [8,8,8,5,5,5]
        super().__init__()
        
        self.encoder = Encoder(ch_mult=ch_mult, z_channels=z_channels)
        self.decoder = Decoder(ch_mult=ch_mult, z_channels=z_channels)
        
        # Quantizer
        self.quantizer_type = quantizer_type
        if quantizer_type == 'fsq':
            if fsq_levels is None:
                # Default FSQ levels for 8 channels
                fsq_levels = [8, 8, 8, 5, 5, 5, 5, 5]  # Total codes = 8^3 * 5^5 ≈ 1.6M
            self.quantizer = FSQuantizer(fsq_levels)
            self.z_channels = len(fsq_levels)
        else:
            self.quantizer = VectorQuantizer(n_embed, z_channels)
            self.z_channels = z_channels
            
        # For compatibility with diffusion models
        self.scale_factor = 1.0  # VQ-VAE typically doesn't need scaling
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to quantized latent"""
        z = self.encoder(x)
        z_q, _ = self.quantizer(z)
        return z_q
        
    def encode_images(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for compatibility"""
        return self.encode(x)
        
    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decode from quantized latent"""
        return self.decoder(z_q)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Full forward pass"""
        # Encode
        z = self.encoder(x)
        z_q, quant_info = self.quantizer(z)
        
        # Decode
        x_recon = self.decoder(z_q)
        
        # Losses
        recon_loss = F.mse_loss(x_recon, x)
        
        output = {
            'recon': x_recon,
            'z_q': z_q,
            'recon_loss': recon_loss,
            'indices': quant_info.get('indices'),
        }
        
        # Add quantizer-specific losses
        if 'loss' in quant_info:
            output['vq_loss'] = quant_info['loss']
            output['loss'] = recon_loss + quant_info['loss']
        else:
            output['loss'] = recon_loss
            
        if 'perplexity' in quant_info:
            output['perplexity'] = quant_info['perplexity']
            
        return output
        
    def get_indices(self, x: torch.Tensor) -> torch.Tensor:
        """Get discrete indices for discrete diffusion"""
        z = self.encoder(x)
        _, quant_info = self.quantizer(z)
        return quant_info['indices']
