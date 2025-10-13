"""
KL-VAE for DDPM - Inspired by Stable Diffusion's AutoencoderKL
轻量级实现，保持核心功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import numpy as np


class ResnetBlock(nn.Module):
    """Residual block with GroupNorm"""
    def __init__(self, in_channels: int, out_channels: int = None, dropout: float = 0.0):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

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


class AttnBlock(nn.Module):
    """Self-attention block"""
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # Compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, h*w)
        w_ = torch.bmm(q, k)
        w_ = w_ * (int(c)**(-0.5))
        w_ = F.softmax(w_, dim=2)

        # Attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class Encoder(nn.Module):
    """VAE Encoder - 4x downsampling for efficiency"""
    def __init__(self, 
                 in_channels: int = 3,
                 ch: int = 128,
                 ch_mult: Tuple[int] = (1, 2, 2, 4),  # 4x downsampling
                 num_res_blocks: int = 2,
                 attn_resolutions: Tuple[int] = (16,),
                 dropout: float = 0.0,
                 z_channels: int = 4,
                 double_z: bool = True):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels

        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = 256  # Assuming 256x256 input
        in_ch_mult = (1,) + ch_mult
        self.down = nn.ModuleList()
        
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
                    
            down = nn.Module()
            down.block = block
            down.attn = attn
            
            if i_level != self.num_resolutions - 1:
                down.downsample = nn.Conv2d(block_in, block_in, kernel_size=3, stride=2, padding=1)
                curr_res = curr_res // 2
                
            self.down.append(down)

        # Middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)

        # End
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, 2*z_channels if double_z else z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Downsampling
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)

        # Middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # End
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    """VAE Decoder - 4x upsampling"""
    def __init__(self,
                 out_ch: int = 3,
                 ch: int = 128,
                 ch_mult: Tuple[int] = (1, 2, 2, 4),
                 num_res_blocks: int = 2,
                 attn_resolutions: Tuple[int] = (16,),
                 dropout: float = 0.0,
                 z_channels: int = 4):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        # Input
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = 256 // 2**(self.num_resolutions - 1)
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # Middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)

        # Upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = nn.ConvTranspose2d(block_in, block_in, kernel_size=4, stride=2, padding=1)
                curr_res = curr_res * 2
            self.up.insert(0, up)

        # End
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        # z to block_in
        h = self.conv_in(z)

        # Middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # Upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # End
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h


class DiagonalGaussianDistribution:
    """VAE分布"""
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])


class KL_VAE(nn.Module):
    """
    KL-VAE inspired by Stable Diffusion's AutoencoderKL
    - 4x downsampling (256->64) for efficiency  
    - 4 latent channels (standard for SD)
    - Built-in scale factor
    """
    def __init__(self,
                 ddconfig: Optional[Dict] = None,
                 embed_dim: int = 4,
                 scale_factor: float = 0.18215):  # SD's standard scale
        super().__init__()
        
        # Default config (4x downsampling)
        if ddconfig is None:
            encoder_config = dict(
                double_z=True,
                z_channels=embed_dim,
                in_channels=3,
                ch=128,
                ch_mult=(1, 2, 2, 4),  # 4x downsampling: 256->128->64->32->16->64
                num_res_blocks=2,
                attn_resolutions=(16,),
                dropout=0.0
            )
            
            decoder_config = dict(
                z_channels=embed_dim,
                out_ch=3,
                ch=128,
                ch_mult=(1, 2, 2, 4),
                num_res_blocks=2,
                attn_resolutions=(16,),
                dropout=0.0
            )
        else:
            # Extract encoder and decoder configs from ddconfig
            encoder_config = {k: v for k, v in ddconfig.items() if k != 'out_ch'}
            decoder_config = {k: v for k, v in ddconfig.items() if k != 'in_channels' and k != 'double_z'}
            
        self.encoder = Encoder(**encoder_config)
        self.decoder = Decoder(**decoder_config)
        
        # Convolutions for distribution parameters
        self.quant_conv = nn.Conv2d(2*embed_dim, 2*embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, embed_dim, 1)
        
        self.embed_dim = embed_dim
        self.scale_factor = scale_factor
        
    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        """Encode to distribution"""
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior
        
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent"""
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec
        
    def forward(self, input: torch.Tensor, sample_posterior: bool = True) -> Tuple[torch.Tensor, DiagonalGaussianDistribution]:
        """Full forward pass"""
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mean
        dec = self.decode(z)
        return dec, posterior
        
    def encode_images(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images to latents for diffusion training"""
        # 直接使用输入，因为VAE是用[0,1]训练的
        posterior = self.encode(x)
        z = posterior.sample()
        # Apply scale factor for stable diffusion training
        return z * self.scale_factor
        
    def decode_latents(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latents to images"""
        # Remove scale factor
        z = z / self.scale_factor
        x = self.decode(z)
        
        # VAE输出应该已经在[0,1]范围附近，只需要clamp
        x = torch.clamp(x, 0, 1)
        
        return x
        
    def get_loss(self, inputs: torch.Tensor, kl_weight: float = 1e-6, 
                 perceptual_loss_fn=None) -> Dict[str, torch.Tensor]:
        """Compute VAE loss with optional perceptual loss"""
        # 直接使用输入，因为VAE是用[0,1]训练的
        reconstructions, posterior = self(inputs)
        
        # Reconstruction loss
        if perceptual_loss_fn is not None:
            # Use perceptual loss
            loss_dict = perceptual_loss_fn(reconstructions, inputs)
            rec_loss = loss_dict['total']
            perceptual_loss = loss_dict.get('perceptual', torch.tensor(0.0))
        else:
            # Default MSE loss
            rec_loss = F.mse_loss(inputs, reconstructions)
            perceptual_loss = torch.tensor(0.0)
        
        # KL loss
        kl_loss = posterior.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        
        # Total loss
        loss = rec_loss + kl_weight * kl_loss
        
        return {
            'loss': loss,
            'rec_loss': rec_loss,
            'kl_loss': kl_loss,
            'perceptual_loss': perceptual_loss
        }
