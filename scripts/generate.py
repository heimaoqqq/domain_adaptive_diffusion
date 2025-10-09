"""
域适应生成脚本
使用训练好的域适应扩散模型生成目标域样本
支持多种生成策略和评估方法
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from models import DomainAdaptiveDiffusion
from utils import (
    set_seed, get_device, create_diffusion_config,
    compute_mmd, compute_fid_features
)
from utils.data_loader import LatentDataset

# denoising-diffusion-pytorch
from denoising_diffusion_pytorch import GaussianDiffusion


class DomainGuidedSampler:
    """
    域引导采样器
    实现多种域适应生成策略
    """
    
    def __init__(
        self,
        model: DomainAdaptiveDiffusion,
        device: str = 'cuda'
    ):
        """
        Args:
            model: 域适应扩散模型
            device: 设备
        """
        self.model = model
        self.device = device
        self.diffusion = model.diffusion
    
    @torch.no_grad()
    def sample_unconditional(
        self,
        batch_size: int = 16,
        return_all_steps: bool = False
    ) -> torch.Tensor:
        """
        无条件采样（基线）
        """
        return self.diffusion.sample(
            batch_size=batch_size,
            return_all_timesteps=return_all_steps
        )
    
    @torch.no_grad()
    def sample_domain_conditional(
        self,
        batch_size: int = 16,
        domain_label: int = 1,  # 0=源域, 1=目标域
        return_all_steps: bool = False
    ) -> torch.Tensor:
        """
        域条件采样
        """
        # 创建域标签
        domain_labels = torch.full(
            (batch_size,), domain_label,
            dtype=torch.long, device=self.device
        )
        
        # 条件采样
        return self.model.sample(
            batch_size=batch_size,
            domain_labels=domain_labels,
            return_all_timesteps=return_all_steps
        )
    
    @torch.no_grad()
    def sample_interpolated(
        self,
        batch_size: int = 16,
        source_weight: float = 0.5,
        target_weight: float = 0.5,
        return_all_steps: bool = False
    ) -> torch.Tensor:
        """
        插值域采样（生成介于源域和目标域之间的样本）
        """
        # 使用混合权重进行采样
        # 这需要模型支持连续域嵌入
        
        # 简化实现：分别生成然后插值
        source_samples = self.sample_domain_conditional(
            batch_size=batch_size,
            domain_label=0
        )
        
        target_samples = self.sample_domain_conditional(
            batch_size=batch_size,
            domain_label=1
        )
        
        # 在latent空间插值
        interpolated = (source_weight * source_samples + 
                       target_weight * target_samples)
        
        return interpolated
    
    @torch.no_grad()
    def sample_with_guidance(
        self,
        batch_size: int = 16,
        target_features: Optional[torch.Tensor] = None,
        guidance_scale: float = 3.0,
        domain_label: int = 1,
        return_all_steps: bool = False
    ) -> torch.Tensor:
        """
        特征引导采样（类似于Universal Guidance但更简单）
        
        Args:
            target_features: 目标特征（可选）
            guidance_scale: 引导强度
        """
        # 如果提供了目标特征，使用它们进行引导
        if target_features is not None:
            # 这里需要实现特征引导逻辑
            # 简化版本：直接使用域条件
            pass
        
        # 使用域条件和增强的引导尺度
        samples = self.sample_domain_conditional(
            batch_size=batch_size,
            domain_label=domain_label,
            return_all_steps=return_all_steps
        )
        
        return samples


def load_checkpoint(
    checkpoint_path: str,
    device: str = 'cuda'
) -> Tuple[DomainAdaptiveDiffusion, Dict]:
    """
    加载训练好的模型
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 从检查点恢复配置
    config = checkpoint.get('config', {})
    
    # 创建模型
    model = DomainAdaptiveDiffusion(
        input_channels=config.get('input_channels', 32),
        image_size=config.get('image_size', 16),
        num_domains=config.get('num_domains', 2),
        **config.get('model_config', {})
    ).to(device)
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 返回模型和元信息
    metadata = {
        'epoch': checkpoint.get('epoch', 0),
        'global_step': checkpoint.get('global_step', 0),
        'best_fid': checkpoint.get('best_fid', float('inf')),
        'config': config
    }
    
    return model, metadata


def decode_latents(
    latents: torch.Tensor,
    vae: nn.Module,
    denormalize: bool = True
) -> torch.Tensor:
    """
    解码latents到图像
    
    Args:
        latents: [B, C, H, W] latent tensors
        vae: VAE解码器
        denormalize: 是否反归一化到[0, 1]
    
    Returns:
        images: [B, 3, 256, 256] 图像
    """
    with torch.no_grad():
        # VAE解码
        if hasattr(vae, 'decode'):
            images = vae.decode(latents)
        else:
            # 如果VAE没有decode方法，尝试其他方法
            images = vae.decoder(latents) if hasattr(vae, 'decoder') else latents
        
        # 反归一化
        if denormalize:
            images = (images + 1.0) / 2.0
            images = torch.clamp(images, 0.0, 1.0)
        
        return images


def save_images(
    images: torch.Tensor,
    output_dir: Path,
    prefix: str = "sample",
    start_idx: int = 0
) -> List[str]:
    """
    保存图像到磁盘
    
    Returns:
        保存的文件路径列表
    """
    saved_paths = []
    
    for i, img in enumerate(images):
        # 转换为PIL图像
        img_np = img.cpu().permute(1, 2, 0).numpy()
        img_np = (img_np * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        
        # 保存
        filename = f"{prefix}_{start_idx + i:04d}.png"
        filepath = output_dir / filename
        img_pil.save(filepath)
        saved_paths.append(str(filepath))
    
    return saved_paths


def evaluate_generation_quality(
    generated_latents: torch.Tensor,
    target_latents: torch.Tensor,
    source_latents: Optional[torch.Tensor] = None,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    评估生成质量
    """
    metrics = {}
    
    # 计算与目标域的MMD距离
    mmd_to_target = compute_mmd(
        generated_latents.to(device),
        target_latents.to(device)
    )
    metrics['mmd_to_target'] = mmd_to_target.item()
    
    # 如果有源域数据，计算与源域的距离
    if source_latents is not None:
        mmd_to_source = compute_mmd(
            generated_latents.to(device),
            source_latents.to(device)
        )
        metrics['mmd_to_source'] = mmd_to_source.item()
        
        # 计算相对位置
        total_distance = compute_mmd(
            source_latents.to(device),
            target_latents.to(device)
        ).item()
        
        metrics['relative_position'] = (
            metrics['mmd_to_target'] / (total_distance + 1e-8)
        )
    
    # 计算多样性（生成样本内部的MMD）
    if len(generated_latents) > 10:
        half = len(generated_latents) // 2
        diversity = compute_mmd(
            generated_latents[:half].to(device),
            generated_latents[half:2*half].to(device)
        )
        metrics['diversity'] = diversity.item()
    
    return metrics


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Generate domain-adapted samples')
    
    # 模型配置
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--vae_path',
        type=str,
        default='../../../simplified_vavae.py',
        help='Path to VAE model'
    )
    
    # 数据配置
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/processed',
        help='Directory with processed latents'
    )
    
    # 生成配置
    parser.add_argument(
        '--num_samples',
        type=int,
        default=100,
        help='Number of samples to generate'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=50,
        help='Batch size for generation'
    )
    parser.add_argument(
        '--sampling_method',
        type=str,
        default='domain_conditional',
        choices=['unconditional', 'domain_conditional', 'interpolated', 'guided'],
        help='Sampling method'
    )
    parser.add_argument(
        '--domain_label',
        type=int,
        default=1,
        help='Domain label for conditional generation (0=source, 1=target)'
    )
    parser.add_argument(
        '--interpolation_weight',
        type=float,
        default=0.5,
        help='Interpolation weight for interpolated sampling'
    )
    
    # 输出配置
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/generated',
        help='Directory to save generated samples'
    )
    parser.add_argument(
        '--save_latents',
        action='store_true',
        help='Save latents in addition to images'
    )
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Evaluate generation quality'
    )
    
    # 其他
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    # 设置设备和种子
    device = args.device or get_device()
    set_seed(args.seed)
    print(f"Using device: {device}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    model, metadata = load_checkpoint(args.checkpoint, device)
    print(f"Model loaded from epoch {metadata['epoch']}")
    
    # 创建采样器
    sampler = DomainGuidedSampler(model, device)
    
    # 加载VAE（用于解码）
    print("\nLoading VAE for decoding...")
    # 这里简化处理，实际需要与prepare_data.py中的load_vae_model一致
    from scripts.prepare_data import load_vae_model
    vae = load_vae_model(args.vae_path, device)
    
    # 生成样本
    print(f"\nGenerating {args.num_samples} samples using {args.sampling_method} method...")
    
    all_latents = []
    all_images = []
    
    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Generating"):
        current_batch_size = min(
            args.batch_size,
            args.num_samples - batch_idx * args.batch_size
        )
        
        # 根据方法生成
        if args.sampling_method == 'unconditional':
            latents = sampler.sample_unconditional(current_batch_size)
        elif args.sampling_method == 'domain_conditional':
            latents = sampler.sample_domain_conditional(
                current_batch_size,
                domain_label=args.domain_label
            )
        elif args.sampling_method == 'interpolated':
            latents = sampler.sample_interpolated(
                current_batch_size,
                source_weight=1.0 - args.interpolation_weight,
                target_weight=args.interpolation_weight
            )
        elif args.sampling_method == 'guided':
            latents = sampler.sample_with_guidance(
                current_batch_size,
                domain_label=args.domain_label
            )
        
        all_latents.append(latents.cpu())
        
        # 解码为图像
        images = decode_latents(latents, vae, denormalize=True)
        all_images.append(images.cpu())
    
    # 合并所有批次
    all_latents = torch.cat(all_latents, dim=0)
    all_images = torch.cat(all_images, dim=0)
    
    print(f"Generated {len(all_latents)} samples")
    
    # 保存latents（如果需要）
    if args.save_latents:
        torch.save(all_latents, output_dir / 'generated_latents.pt')
        print(f"Saved latents to {output_dir / 'generated_latents.pt'}")
    
    # 保存图像
    print("\nSaving images...")
    saved_paths = save_images(
        all_images,
        output_dir,
        prefix=f"{args.sampling_method}_sample"
    )
    print(f"Saved {len(saved_paths)} images to {output_dir}")
    
    # 评估质量（如果需要）
    if args.evaluate:
        print("\nEvaluating generation quality...")
        
        # 加载目标域数据
        data_dir = Path(args.data_dir)
        target_latents = torch.load(data_dir / 'target_latents.pt')
        source_latents = None
        
        if (data_dir / 'source_latents.pt').exists():
            source_latents = torch.load(data_dir / 'source_latents.pt')
        
        # 计算指标
        metrics = evaluate_generation_quality(
            all_latents,
            target_latents,
            source_latents,
            device
        )
        
        # 打印结果
        print("\nGeneration Quality Metrics:")
        print("-" * 40)
        for key, value in metrics.items():
            print(f"{key:20s}: {value:.4f}")
        
        # 保存指标
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
    
    # 生成摘要
    summary = {
        'num_samples': len(all_latents),
        'sampling_method': args.sampling_method,
        'domain_label': args.domain_label,
        'checkpoint_epoch': metadata['epoch'],
        'output_dir': str(output_dir),
        'timestamp': str(np.datetime64('now'))
    }
    
    if args.evaluate and metrics:
        summary['metrics'] = metrics
    
    with open(output_dir / 'generation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*60)
    print("Generation completed!")
    print(f"Method: {args.sampling_method}")
    print(f"Samples: {len(all_latents)}")
    print(f"Output: {output_dir}")
    if args.evaluate and 'relative_position' in metrics:
        print(f"Position: {metrics['relative_position']:.1%} towards target domain")
    print("="*60)


if __name__ == '__main__':
    main()
