"""
简化版微多普勒数据准备脚本
专门为小数据集和KL-VAE设计
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
from tqdm import tqdm
import random
import os
import sys

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))
from utils import set_seed


class MicroDopplerDataset(Dataset):
    """微多普勒数据集"""
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        # 加载图像
        img = Image.open(self.image_paths[idx]).convert('RGB')
        
        # 转换为tensor并归一化到[-1, 1]
        img = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
        img = 2.0 * img - 1.0
        
        return img, self.labels[idx]


def collect_data(root_dir: Path, gait_type: str) -> Dict[str, List[Tuple[str, int]]]:
    """收集数据并按用户ID组织"""
    data_dir = root_dir / gait_type
    if not data_dir.exists():
        raise ValueError(f"数据目录不存在: {data_dir}")
    
    user_data = {}
    
    # 遍历用户目录
    for user_dir in sorted(data_dir.iterdir()):
        if not user_dir.is_dir() or not user_dir.name.startswith('ID_'):
            continue
            
        user_id = int(user_dir.name.split('_')[1]) - 1  # 0-indexed
        images = list(user_dir.glob('*.jpg'))
        
        if images:
            user_data[user_id] = [(str(img), user_id) for img in images]
            print(f"  用户 {user_dir.name}: {len(images)} 张图像")
    
    return user_data


def split_data(user_data: Dict[str, List], train_ratio: float = 0.8) -> Tuple[List, List]:
    """按用户划分训练/验证集"""
    train_data = []
    val_data = []
    
    for user_id, data in user_data.items():
        # 打乱数据
        random.shuffle(data)
        
        # 划分
        n_train = int(len(data) * train_ratio)
        train_data.extend(data[:n_train])
        val_data.extend(data[n_train:])
    
    return train_data, val_data


def select_fewshot_samples(user_data: Dict[str, List], n_shot: int = 5) -> List:
    """为每个用户选择few-shot样本"""
    fewshot_data = []
    
    for user_id, data in user_data.items():
        # 随机选择n_shot个样本
        selected = random.sample(data, min(n_shot, len(data)))
        fewshot_data.extend(selected)
        
    return fewshot_data


def load_kl_vae(checkpoint_path: Path, device: str) -> nn.Module:
    """加载KL-VAE模型"""
    # 添加VAE模块路径
    vae_module_path = Path(__file__).parent.parent / 'vae'
    sys.path.insert(0, str(vae_module_path))
    from kl_vae import KL_VAE
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"VAE checkpoint不存在: {checkpoint_path}")
    
    # 加载模型
    checkpoint = torch.load(checkpoint_path, map_location=device)
    vae = KL_VAE()
    
    if 'model_state_dict' in checkpoint:
        vae.load_state_dict(checkpoint['model_state_dict'])
    else:
        vae.load_state_dict(checkpoint)
    
    vae = vae.to(device)
    vae.eval()
    
    print(f"✅ 加载KL-VAE成功")
    return vae


def encode_data(vae: nn.Module, dataloader: DataLoader, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """使用VAE编码数据"""
    all_latents = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="编码中"):
            images = images.to(device)
            
            # 编码
            posterior = vae.encode(images)
            
            # 从分布中采样
            if hasattr(posterior, 'mode'):
                # VAE返回的是分布对象，使用mode()获得确定性结果
                latents = posterior.mode()
            elif hasattr(posterior, 'sample'):
                # 备选：使用sample()
                latents = posterior.sample()
            else:
                # 如果直接返回张量
                latents = posterior
            
            # KL-VAE的scale_factor
            scale_factor = getattr(vae, 'scale_factor', 0.18215)
            latents = latents * scale_factor
            
            all_latents.append(latents.cpu())
            all_labels.append(labels)
    
    return torch.cat(all_latents, dim=0), torch.cat(all_labels, dim=0)


def main():
    parser = argparse.ArgumentParser(description='准备微多普勒数据')
    
    # 数据配置
    parser.add_argument('--dataset_root', type=str,
                       default=r'D:\Ysj\新建文件夹\VA-VAE\dataset\organized_gait_dataset',
                       help='数据集根目录')
    parser.add_argument('--source_domain', type=str, default='Normal_line',
                       help='源域步态类型')
    parser.add_argument('--target_domain', type=str, default='Normal_free',
                       help='目标域步态类型')
    
    # VAE配置
    parser.add_argument('--vae_checkpoint', type=str,
                       default='../vae/checkpoints/kl_vae_best.pt',
                       help='KL-VAE checkpoint路径')
    
    # 训练配置
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='训练集比例')
    parser.add_argument('--n_shot', type=int, default=5,
                       help='目标域每用户样本数')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批次大小')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    # 输出配置
    parser.add_argument('--output_dir', type=str, default='../data/processed',
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n准备数据集: {args.source_domain} -> {args.target_domain}")
    
    # 收集数据
    dataset_root = Path(args.dataset_root)
    source_data = collect_data(dataset_root, args.source_domain)
    target_data = collect_data(dataset_root, args.target_domain)
    
    print(f"\n源域: {len(source_data)} 用户")
    print(f"目标域: {len(target_data)} 用户")
    
    # 划分数据
    source_train, source_val = split_data(source_data, args.train_ratio)
    target_fewshot = select_fewshot_samples(target_data, args.n_shot)
    
    print(f"\n数据划分:")
    print(f"源域 - 训练: {len(source_train)}, 验证: {len(source_val)}")
    print(f"目标域 - Few-shot: {len(target_fewshot)} ({args.n_shot}张/用户)")
    
    # 加载VAE
    print(f"\n加载VAE模型...")
    vae = load_kl_vae(Path(args.vae_checkpoint), device)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 编码数据
    datasets = {
        'source_train': source_train,
        'source_val': source_val,
        'target_fewshot': target_fewshot
    }
    
    for name, data in datasets.items():
        if not data:
            continue
            
        print(f"\n处理 {name}...")
        
        # 创建数据集和加载器
        paths = [d[0] for d in data]
        labels = [d[1] for d in data]
        dataset = MicroDopplerDataset(paths, labels)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, 
                              shuffle=False, num_workers=0)
        
        # 编码
        latents, labels = encode_data(vae, dataloader, device)
        
        # 保存
        torch.save({
            'latents': latents,
            'labels': labels,
            'paths': paths
        }, output_dir / f'{name}.pt')
        
        print(f"  保存: {name}.pt (形状: {latents.shape})")
    
    # 保存数据统计
    stats = {
        'source_domain': args.source_domain,
        'target_domain': args.target_domain,
        'n_users': len(source_data),
        'n_shot': args.n_shot,
        'latent_shape': [4, 64, 64],  # KL-VAE输出
        'scale_factor': getattr(vae, 'scale_factor', 0.18215)
    }
    
    torch.save(stats, output_dir / 'data_stats.pt')
    
    print("\n✅ 数据准备完成!")
    print(f"输出目录: {output_dir}")


if __name__ == '__main__':
    main()
