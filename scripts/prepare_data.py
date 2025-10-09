"""
数据准备脚本
将图像数据编码为VAE latent空间
严格匹配VAE接口，确保数据质量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))  # 访问上级VA-VAE

from utils import set_seed, get_device


class ImageDataset(Dataset):
    """
    图像数据集
    支持多种图像格式和目录结构
    """
    
    def __init__(
        self,
        data_dir: str,
        image_size: int = 256,
        extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp'],
        normalize: bool = True
    ):
        """
        Args:
            data_dir: 数据目录路径
            image_size: 图像大小
            extensions: 支持的图像扩展名
            normalize: 是否归一化到[-1, 1]
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.normalize = normalize
        
        # 收集所有图像路径
        self.image_paths = []
        self.labels = []
        
        # 假设目录结构: data_dir/class_name/image.jpg
        # 或者平面结构: data_dir/user_XX_sample_YY.jpg
        
        if self._is_class_folder_structure():
            self._load_class_folder_structure()
        else:
            self._load_flat_structure()
        
        print(f"Found {len(self.image_paths)} images from {len(set(self.labels))} classes")
    
    def _is_class_folder_structure(self) -> bool:
        """判断是否为类别文件夹结构"""
        subdirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        return len(subdirs) > 0
    
    def _load_class_folder_structure(self):
        """加载类别文件夹结构的数据"""
        class_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        
        for class_idx, class_dir in enumerate(class_dirs):
            class_name = class_dir.name
            
            # 收集该类别的所有图像
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                for img_path in class_dir.glob(f'*{ext}'):
                    self.image_paths.append(str(img_path))
                    self.labels.append(class_idx)
            
            print(f"  Class {class_idx} ({class_name}): {len([l for l in self.labels if l == class_idx])} images")
    
    def _load_flat_structure(self):
        """加载平面结构的数据（从文件名推断标签）"""
        all_images = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            all_images.extend(self.data_dir.glob(f'*{ext}'))
        
        all_images = sorted(all_images)
        
        for img_path in all_images:
            self.image_paths.append(str(img_path))
            
            # 尝试从文件名提取标签
            # 支持多种格式: ID_XX, user_XX, class_XX等
            filename = img_path.stem
            
            # 尝试不同的ID格式
            user_id = None
            
            # 格式1: ID_XX_sample_YY.jpg
            if 'ID_' in filename:
                try:
                    user_id = int(filename.split('ID_')[1].split('_')[0]) - 1  # 转为0-indexed
                except:
                    pass
            
            # 格式2: user_XX_sample_YY.jpg
            elif 'user_' in filename:
                try:
                    user_id = int(filename.split('user_')[1].split('_')[0]) - 1
                except:
                    pass
            
            # 格式3: class_XX_...
            elif 'class_' in filename:
                try:
                    user_id = int(filename.split('class_')[1].split('_')[0])
                except:
                    pass
            
            # 如果成功提取，使用该ID；否则使用默认值
            if user_id is not None and 0 <= user_id < 31:  # 确保在有效范围内
                self.labels.append(user_id)
            else:
                self.labels.append(0)  # 默认标签
                print(f"Warning: Cannot extract valid ID from {filename}, using default label 0")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        返回:
            image: [C, H, W] tensor
            label: 类别标签
        """
        # 加载图像
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # 调整大小
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        
        # 转换为tensor
        image = np.array(image).astype(np.float32) / 255.0  # [H, W, C]
        image = torch.from_numpy(image).permute(2, 0, 1)  # [C, H, W]
        
        # 归一化到[-1, 1]（如果需要）
        if self.normalize:
            image = 2.0 * image - 1.0
        
        return image, self.labels[idx]


def load_vae_model(vae_path: str = None, device: str = 'cuda', checkpoint_path: str = None) -> nn.Module:
    """
    加载VAE模型的包装器 - 使用与run_universal_guidance.py相同的方法
    
    Args:
        vae_path: 保留以兼容旧接口（未使用）
        device: 设备
        checkpoint_path: VAE checkpoint路径
    
    Returns:
        加载的VAE模型
    """
    # 使用新的加载方法
    from load_vae_from_checkpoint import load_vae_model as load_vae_impl
    return load_vae_impl(checkpoint_path, device)


def load_vae_model_old(vae_path: str, device: str = 'cuda', checkpoint_path: str = None) -> nn.Module:
    """
    加载您训练好的VA-VAE模型
    只支持simplified_vavae.py，确保正确加载
    
    Args:
        vae_path: simplified_vavae.py的路径
        device: 设备
        checkpoint_path: VAE checkpoint路径（如果不指定，会在默认位置查找）
    
    Returns:
        加载好的VAE模型
    """
    vae_path = Path(vae_path)
    
    if not vae_path.exists():
        raise FileNotFoundError(f"VAE文件不存在: {vae_path}")
    
    if vae_path.name != 'simplified_vavae.py':
        raise ValueError(f"只支持simplified_vavae.py，当前文件: {vae_path.name}")
    
    # 添加路径以便导入
    sys.path.insert(0, str(vae_path.parent))
    
    # 添加LightningDiT目录到路径
    lightningdit_path = vae_path.parent / "LightningDiT"
    if lightningdit_path.exists():
        sys.path.insert(0, str(lightningdit_path))
        sys.path.insert(0, str(lightningdit_path / "vavae"))
        sys.path.insert(0, str(lightningdit_path / "tokenizer"))
    
    try:
        # 先尝试直接从LightningDiT导入VA-VAE
        try:
            from tokenizer.vavae import VAVAE
            print(f"从LightningDiT导入VAVAE成功")
            use_lightning_vae = True
        except ImportError:
            # 如果失败，使用SimplifiedVAVAE
            from simplified_vavae import SimplifiedVAVAE
            print(f"导入SimplifiedVAVAE成功")
            use_lightning_vae = False
        
        import torch
        
        # 查找checkpoint
        if checkpoint_path is None:
            # 尝试默认位置
            possible_checkpoints = [
                vae_path.parent / 'vae_checkpoint.pt',
                vae_path.parent / 'checkpoints' / 'best_vae.pt',
                vae_path.parent / 'microdoppler_finetune' / 'vavae_stage3_best.ckpt',
            ]
            
            for ckpt_path in possible_checkpoints:
                if ckpt_path.exists():
                    checkpoint_path = str(ckpt_path)
                    print(f"找到checkpoint: {checkpoint_path}")
                    break
        
        # 创建VAE实例并加载checkpoint
        if use_lightning_vae:
            # 使用LightningDiT的VAVAE
            print(f"使用LightningDiT VAVAE")
            
            # 从checkpoint加载
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # 创建VAVAE实例（使用默认配置）
            vae = VAVAE(
                embed_dim=32,  # 你的VAE使用32通道
                n_embed=None,  # 不使用VQ
                double_z=False,
                # 其他参数使用默认值
            )
            
            # 加载权重
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                # 移除可能的前缀
                state_dict = {k.replace('vae.', '').replace('module.', ''): v 
                             for k, v in state_dict.items()}
                vae.load_state_dict(state_dict, strict=False)
            else:
                vae.load_state_dict(checkpoint, strict=False)
            
            vae = vae.to(device)
            vae.eval()
            
            # 不要硬编码scale_factor，从checkpoint中读取
            if 'scale_factor' in checkpoint:
                vae.scale_factor = float(checkpoint['scale_factor'])
            else:
                # 如果checkpoint中没有，使用默认值1.0而不是0.3989
                vae.scale_factor = 1.0
            
        else:
            # 使用SimplifiedVAVAE，它会从checkpoint自动读取scale_factor
            print(f"使用SimplifiedVAVAE")
            vae = SimplifiedVAVAE(checkpoint_path=checkpoint_path, use_vf='dinov2')
            vae = vae.to(device)
            vae.eval()
        
        print(f"✅ VA-VAE加载成功")
        print(f"   Latent channels: 32")
        print(f"   Downsample factor: 16")
        # 对于DDPM/LDM，scale_factor很重要
        actual_scale = getattr(vae, 'scale_factor', 1.0)
        print(f"   Scale factor: {actual_scale}")
        if actual_scale == 1.0:
            print(f"   ⚠️ 注意：scale_factor为1.0，可能未从checkpoint正确读取")
        
        return vae
        
    except ImportError as e:
        print(f"⚠️ 无法导入VAE模块: {e}")
        print(f"尝试使用简化的直接加载方式...")
        
        try:
            # 使用简化的直接加载
            from load_vavae_direct import load_vavae_checkpoint_direct
            
            if checkpoint_path is None:
                raise ValueError("使用简化加载方式必须提供checkpoint路径")
            
            vae = load_vavae_checkpoint_direct(checkpoint_path, device)
            return vae
            
        except Exception as fallback_error:
            error_msg = f"""
❌ 无法加载VAE！

原始错误: {e}
备用方案错误: {fallback_error}

可能的原因：
1. 缺少依赖库（如pytorch_lightning, taming, ldm等）
2. simplified_vavae.py中的导入路径问题

解决方案：
1. 确保安装了所有依赖：
   pip install pytorch-lightning omegaconf
   
2. 或使用提供的简化加载脚本

当前VAE路径: {vae_path}
当前checkpoint: {checkpoint_path}
"""
            print(error_msg)
            raise ImportError(error_msg)
    
    except Exception as e:
        error_msg = f"""
❌ VAE加载失败！

错误信息: {e}

当前VAE路径: {vae_path}
Checkpoint路径: {checkpoint_path}

请检查：
1. simplified_vavae.py是否完整
2. checkpoint文件是否存在且有效
3. 所有依赖是否已安装
"""
        print(error_msg)
        raise RuntimeError(error_msg)


@torch.no_grad()
def encode_dataset(
    dataloader: DataLoader,
    vae: nn.Module,
    device: str = 'cuda',
    desc: str = "Encoding"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    编码数据集到latent空间
    
    Returns:
        latents: [N, C, H, W] 编码后的latents
        labels: [N] 标签
    """
    all_latents = []
    all_labels = []
    
    for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc=desc)):
        images = images.to(device)
        
        # VAE编码 - 与step6_encode_official.py第161行保持一致
        # 直接使用encode_images，不涉及scale_factor
        if hasattr(vae, 'encode_images'):
            # VA-VAE的标准方法（step6_encode_official.py使用的方式）
            latents = vae.encode_images(images).detach()
        elif hasattr(vae, 'encode'):
            # 备用方法
            latents = vae.encode(images).detach()
            # 如果返回的是分布，取sample
            if hasattr(latents, 'sample'):
                latents = latents.sample()
        else:
            # 尝试直接调用
            raise AttributeError("VAE必须有encode_images或encode方法")
        
        all_latents.append(latents.cpu())
        all_labels.append(labels)
    
    # 合并所有batch
    all_latents = torch.cat(all_latents, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    return all_latents, all_labels


def compute_statistics(latents: torch.Tensor) -> Dict[str, torch.Tensor]:
    """计算latent统计量"""
    stats = {
        'mean': latents.mean(dim=0, keepdim=True),
        'std': latents.std(dim=0, keepdim=True),
        'min': latents.min(),
        'max': latents.max(),
        'shape': torch.tensor(latents.shape)
    }
    
    print("\nLatent statistics:")
    print(f"  Shape: {list(latents.shape)}")
    print(f"  Mean: {latents.mean():.4f}")
    print(f"  Std: {latents.std():.4f}")
    print(f"  Min: {latents.min():.4f}")
    print(f"  Max: {latents.max():.4f}")
    
    return stats


def validate_latents(latents: torch.Tensor) -> bool:
    """验证latent质量"""
    # 检查NaN和Inf
    if torch.isnan(latents).any():
        print("Warning: NaN values detected in latents!")
        return False
    
    if torch.isinf(latents).any():
        print("Warning: Inf values detected in latents!")
        return False
    
    # 检查值范围
    if latents.abs().max() > 100:
        print(f"Warning: Large values detected (max={latents.abs().max():.2f})")
    
    # 检查方差
    if latents.std() < 0.01:
        print(f"Warning: Low variance detected (std={latents.std():.4f})")
    
    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Prepare data for domain adaptive diffusion')
    
    # 数据路径
    parser.add_argument(
        '--source_dir',
        type=str,
        required=True,
        help='Path to source domain data directory'
    )
    parser.add_argument(
        '--target_dir',
        type=str,
        default=None,
        help='Path to target domain data directory (optional)'
    )
    
    # VAE配置
    parser.add_argument(
        '--vae_path',
        type=str,
        default='../simplified_vavae.py',
        help='Path to VAE model file'
    )
    
    # 输出路径
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/processed',
        help='Directory to save processed latents'
    )
    
    # 处理参数
    parser.add_argument(
        '--batch_size',
        type=int,
        default=50,
        help='Batch size for encoding'
    )
    parser.add_argument(
        '--image_size',
        type=int,
        default=256,
        help='Image size'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
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
    
    # 加载VAE
    print("\nLoading VAE model...")
    vae = load_vae_model(args.vae_path, device)
    
    # 处理源域数据
    print(f"\nProcessing source domain data from: {args.source_dir}")
    source_dataset = ImageDataset(
        args.source_dir,
        image_size=args.image_size,
        normalize=True
    )
    
    source_loader = DataLoader(
        source_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 编码源域数据
    source_latents, source_labels = encode_dataset(
        source_loader, vae, device,
        desc="Encoding source domain"
    )
    
    # 计算统计量
    source_stats = compute_statistics(source_latents)
    
    # 验证质量
    if not validate_latents(source_latents):
        print("Warning: Source latents validation failed!")
    
    # 保存源域数据
    torch.save(source_latents, output_dir / 'source_latents.pt')
    torch.save(source_labels, output_dir / 'source_labels.pt')
    torch.save(source_stats, output_dir / 'source_stats.pt')
    print(f"Saved source domain latents to {output_dir}")
    
    # 处理目标域数据（如果提供）
    if args.target_dir:
        print(f"\nProcessing target domain data from: {args.target_dir}")
        target_dataset = ImageDataset(
            args.target_dir,
            image_size=args.image_size,
            normalize=True
        )
        
        target_loader = DataLoader(
            target_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        # 编码目标域数据
        target_latents, target_labels = encode_dataset(
            target_loader, vae, device,
            desc="Encoding target domain"
        )
        
        # 计算统计量
        target_stats = compute_statistics(target_latents)
        
        # 验证质量
        if not validate_latents(target_latents):
            print("Warning: Target latents validation failed!")
        
        # 保存目标域数据
        torch.save(target_latents, output_dir / 'target_latents.pt')
        torch.save(target_labels, output_dir / 'target_labels.pt')
        torch.save(target_stats, output_dir / 'target_stats.pt')
        print(f"Saved target domain latents to {output_dir}")
        
        # 计算域差异
        print("\nDomain shift analysis:")
        mean_diff = (source_stats['mean'] - target_stats['mean']).abs().mean()
        std_diff = (source_stats['std'] - target_stats['std']).abs().mean()
        print(f"  Mean difference: {mean_diff:.4f}")
        print(f"  Std difference: {std_diff:.4f}")
    
    # 保存元数据
    metadata = {
        'source_samples': len(source_latents),
        'source_classes': len(torch.unique(source_labels)),
        'target_samples': len(target_latents) if args.target_dir else 0,
        'target_classes': len(torch.unique(target_labels)) if args.target_dir else 0,
        'image_size': args.image_size,
        'latent_shape': list(source_latents.shape[1:]),
        'vae_path': str(args.vae_path),
        'timestamp': str(np.datetime64('now'))
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "="*60)
    print("Data preparation completed!")
    print(f"Source: {metadata['source_samples']} samples, {metadata['source_classes']} classes")
    if args.target_dir:
        print(f"Target: {metadata['target_samples']} samples, {metadata['target_classes']} classes")
    print(f"Latent shape: {metadata['latent_shape']}")
    print(f"Output directory: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()

