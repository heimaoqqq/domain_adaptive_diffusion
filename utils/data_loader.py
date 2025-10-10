"""
数据加载模块
处理源域和目标域的latent数据
严格匹配模型接口要求
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import random


class DomainAdaptiveDataset(Dataset):
    """
    域适应数据集
    同时处理源域和目标域数据，支持多种采样策略
    """
    
    def __init__(
        self,
        source_latents: torch.Tensor,
        source_labels: torch.Tensor,
        target_latents: Optional[torch.Tensor] = None,
        target_labels: Optional[torch.Tensor] = None,
        phase: str = 'pretrain',  # 'pretrain' | 'align' | 'finetune'
        domain_balance_ratio: float = 0.8,
        augmentation: bool = False,
        augmentation_config: Optional[Dict] = None
    ):
        """
        Args:
            source_latents: 源域latent [N_s, C, H, W]
            source_labels: 源域标签 [N_s]
            target_latents: 目标域latent [N_t, C, H, W]
            target_labels: 目标域标签 [N_t]
            phase: 训练阶段
            domain_balance_ratio: 源域数据比例（用于align阶段）
            augmentation: 是否使用数据增强
            augmentation_config: 增强配置
        """
        self.source_latents = source_latents
        self.source_labels = source_labels
        self.target_latents = target_latents
        self.target_labels = target_labels
        self.phase = phase
        self.domain_balance_ratio = domain_balance_ratio
        self.augmentation = augmentation
        self.augmentation_config = augmentation_config or {}
        
        # 计算数据集大小
        self.source_size = len(source_latents)
        self.target_size = len(target_latents) if target_latents is not None else 0
        
        # 根据训练阶段确定数据集大小
        if phase == 'pretrain':
            self.total_size = self.source_size
        elif phase == 'align':
            # 混合采样，保证每个epoch都能看到所有数据
            self.total_size = max(self.source_size, self.target_size * 10)
        elif phase == 'finetune':
            # 主要使用目标域，但循环采样
            self.total_size = max(self.target_size * 20, 1000)
        else:
            raise ValueError(f"Unknown phase: {phase}")
    
    def __len__(self) -> int:
        return self.total_size
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        返回格式匹配模型接口：
        - latent: [C, H, W]
        - class_label: 标量
        - domain_label: 标量 (0=源域, 1=目标域)
        """
        if self.phase == 'pretrain':
            # 只使用源域
            real_idx = idx % self.source_size
            latent = self.source_latents[real_idx]
            label = self.source_labels[real_idx]
            domain = 0
            
        elif self.phase == 'align':
            # 混合采样
            if random.random() < self.domain_balance_ratio:
                # 源域
                real_idx = idx % self.source_size
                latent = self.source_latents[real_idx]
                label = self.source_labels[real_idx]
                domain = 0
            else:
                # 目标域（循环采样）
                real_idx = idx % self.target_size
                latent = self.target_latents[real_idx]
                label = self.target_labels[real_idx]
                domain = 1
                
        elif self.phase == 'finetune':
            # 主要使用目标域
            if self.target_size > 0 and random.random() < 0.9:
                # 90%概率采样目标域
                real_idx = idx % self.target_size
                latent = self.target_latents[real_idx]
                label = self.target_labels[real_idx]
                domain = 1
            else:
                # 10%概率采样源域（保持多样性）
                real_idx = idx % self.source_size
                latent = self.source_latents[real_idx]
                label = self.source_labels[real_idx]
                domain = 0
        
        # 数据增强（在latent空间）
        if self.augmentation and domain == 1:  # 只对目标域增强
            latent = self.apply_augmentation(latent)
        
        return {
            'latent': latent,
            'class_label': label,
            'domain_label': torch.tensor(domain, dtype=torch.long)
        }
    
    def apply_augmentation(self, latent: torch.Tensor) -> torch.Tensor:
        """
        在latent空间应用数据增强
        保持物理意义的同时增加多样性
        """
        aug_latent = latent.clone()
        
        # 添加小幅噪声
        if self.augmentation_config.get('noise_level', 0) > 0:
            noise_level = self.augmentation_config['noise_level']
            noise = torch.randn_like(latent) * noise_level
            aug_latent = aug_latent + noise
        
        # Mixup（在同域内）
        if self.augmentation_config.get('use_mixup', False) and random.random() < 0.5:
            alpha = self.augmentation_config.get('mixup_alpha', 0.2)
            lam = np.random.beta(alpha, alpha)
            # 随机选择另一个目标域样本
            if self.target_size > 0:
                idx2 = random.randint(0, self.target_size - 1)
                latent2 = self.target_latents[idx2]
                aug_latent = lam * aug_latent + (1 - lam) * latent2
        
        return aug_latent


class BalancedDomainSampler(torch.utils.data.Sampler):
    """
    平衡的域采样器
    确保每个batch中源域和目标域的比例固定
    """
    
    def __init__(
        self,
        dataset: DomainAdaptiveDataset,
        batch_size: int,
        source_ratio: float = 0.8,
        drop_last: bool = True
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.source_ratio = source_ratio
        self.drop_last = drop_last
        
        # 计算每个batch中的源域和目标域样本数
        self.source_per_batch = int(batch_size * source_ratio)
        self.target_per_batch = batch_size - self.source_per_batch
        
        # 生成索引
        self.source_indices = list(range(dataset.source_size))
        if dataset.target_size > 0:
            self.target_indices = list(range(dataset.target_size))
        else:
            self.target_indices = []
    
    def __iter__(self):
        # 打乱索引
        random.shuffle(self.source_indices)
        if self.target_indices:
            random.shuffle(self.target_indices)
        
        # 生成batch
        source_ptr = 0
        target_ptr = 0
        
        while True:
            batch_indices = []
            
            # 添加源域样本
            for _ in range(self.source_per_batch):
                if source_ptr >= len(self.source_indices):
                    random.shuffle(self.source_indices)
                    source_ptr = 0
                batch_indices.append(self.source_indices[source_ptr])
                source_ptr += 1
            
            # 添加目标域样本
            if self.target_indices:
                for _ in range(self.target_per_batch):
                    if target_ptr >= len(self.target_indices):
                        random.shuffle(self.target_indices)
                        target_ptr = 0
                    # 加上源域大小的偏移
                    batch_indices.append(
                        len(self.source_indices) + self.target_indices[target_ptr]
                    )
                    target_ptr += 1
            
            # 打乱batch内顺序
            random.shuffle(batch_indices)
            
            for idx in batch_indices:
                yield idx
            
            # 检查是否结束
            if source_ptr >= len(self.source_indices):
                break
    
    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def load_latents(data_path: str, device: str = 'cpu') -> Dict[str, torch.Tensor]:
    """
    加载预处理的latent数据
    
    Returns:
        包含source_latents, source_labels, target_latents, target_labels的字典
    """
    data_path = Path(data_path)
    
    data = {}
    
    # 尝试新格式文件（prepare_microdoppler_data.py生成的）
    source_train_path = data_path / 'source_train.pt'
    source_val_path = data_path / 'source_val.pt'
    target_fewshot_path = data_path / 'target_fewshot.pt'
    
    # 兼容旧格式文件
    source_latents_path = data_path / 'source_latents.pt'
    source_labels_path = data_path / 'source_labels.pt'
    
    # 加载源域数据
    if source_train_path.exists():
        # 使用新格式：source_train.pt
        source_data = torch.load(source_train_path, map_location=device)
        
        # 检查是否为字典格式（包含latents和labels）
        if isinstance(source_data, dict) and 'latents' in source_data:
            data['source_latents'] = source_data['latents']
            data['source_labels'] = source_data.get('labels', None)
        else:
            # 纯张量格式
            data['source_latents'] = source_data
            data['source_labels'] = None
            
        print(f"Loaded source training data: {data['source_latents'].shape}")
        
        # 如果没有labels，自动生成（31个用户）
        if data['source_labels'] is None:
            n_samples = len(data['source_latents'])
            n_users = 31
            samples_per_user = n_samples // n_users
            labels = []
            for i in range(n_users):
                labels.extend([i] * samples_per_user)
            # 处理剩余样本
            for i in range(n_samples - len(labels)):
                labels.append(i % n_users)
            data['source_labels'] = torch.tensor(labels, dtype=torch.long, device=device)
            print(f"Generated source labels: {data['source_labels'].shape}")
        else:
            print(f"Loaded source labels: {data['source_labels'].shape}")
        
    elif source_latents_path.exists():
        # 使用旧格式
        data['source_latents'] = torch.load(source_latents_path, map_location=device)
        print(f"Loaded source latents: {data['source_latents'].shape}")
        
        if source_labels_path.exists():
            data['source_labels'] = torch.load(source_labels_path, map_location=device)
            print(f"Loaded source labels: {data['source_labels'].shape}")
        else:
            raise FileNotFoundError(f"Source labels not found: {source_labels_path}")
    else:
        raise FileNotFoundError(f"Source data not found. Looked for: {source_train_path} or {source_latents_path}")
    
    # 加载目标域数据
    if target_fewshot_path.exists():
        # 使用新格式：target_fewshot.pt
        target_data = torch.load(target_fewshot_path, map_location=device)
        
        # 检查是否为字典格式
        if isinstance(target_data, dict) and 'latents' in target_data:
            data['target_latents'] = target_data['latents']
            data['target_labels'] = target_data.get('labels', None)
        else:
            data['target_latents'] = target_data
            data['target_labels'] = None
            
        print(f"Loaded target few-shot data: {data['target_latents'].shape}")
        
        # 如果没有labels，生成labels（31用户，每用户5张）
        if data['target_labels'] is None:
            n_samples = len(data['target_latents'])
            n_users = 31
            samples_per_user = 5
            labels = []
            for i in range(n_users):
                for j in range(samples_per_user):
                    if len(labels) < n_samples:
                        labels.append(i)
            data['target_labels'] = torch.tensor(labels[:n_samples], dtype=torch.long, device=device)
            print(f"Generated target labels: {data['target_labels'].shape}")
        else:
            print(f"Loaded target labels: {data['target_labels'].shape}")
        
    else:
        # 尝试旧格式
        target_latents_path = data_path / 'target_latents.pt'
        target_labels_path = data_path / 'target_labels.pt'
        
        if target_latents_path.exists():
            data['target_latents'] = torch.load(target_latents_path, map_location=device)
            print(f"Loaded target latents: {data['target_latents'].shape}")
            if target_labels_path.exists():
                data['target_labels'] = torch.load(target_labels_path, map_location=device)
            else:
                data['target_labels'] = None
        else:
            print("Warning: Target data not found, using source domain only")
            data['target_latents'] = None
            data['target_labels'] = None
    
    # 验证数据
    assert data['source_latents'].dim() == 4, "Latents should be 4D: [N, C, H, W]"
    assert len(data['source_latents']) == len(data['source_labels']), "Mismatch in source data"
    
    if data['target_latents'] is not None:
        assert data['target_latents'].dim() == 4, "Target latents should be 4D"
        assert len(data['target_latents']) == len(data['target_labels']), "Mismatch in target data"
        # 检查维度匹配
        assert data['source_latents'].shape[1:] == data['target_latents'].shape[1:], \
            "Source and target latents should have same dimensions"
    
    return data


def create_dataloaders(
    data_path: str,
    phase: str,
    batch_size: int,
    num_workers: int = 4,
    domain_balance_ratio: float = 0.8,
    augmentation: bool = False,
    augmentation_config: Optional[Dict] = None,
    device: str = 'cpu'
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    创建数据加载器
    
    Returns:
        (train_loader, val_loader) - val_loader可能为None
    """
    # 加载数据
    data = load_latents(data_path, device)
    
    # 创建数据集
    dataset = DomainAdaptiveDataset(
        source_latents=data['source_latents'],
        source_labels=data['source_labels'],
        target_latents=data['target_latents'],
        target_labels=data['target_labels'],
        phase=phase,
        domain_balance_ratio=domain_balance_ratio,
        augmentation=augmentation,
        augmentation_config=augmentation_config
    )
    
    # 创建数据加载器
    if phase == 'align' and data['target_latents'] is not None:
        # 使用平衡采样器
        sampler = BalancedDomainSampler(
            dataset,
            batch_size=batch_size,
            source_ratio=domain_balance_ratio
        )
        train_loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        # 标准数据加载器
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True
        )
    
    # 创建验证集
    val_loader = None
    
    # 尝试加载专门的验证集数据
    source_val_path = Path(data_path) / 'source_val.pt'
    if source_val_path.exists():
        # 使用专门的验证集
        val_data_raw = torch.load(source_val_path, map_location=device)
        
        # 检查格式
        if isinstance(val_data_raw, dict) and 'latents' in val_data_raw:
            val_data = val_data_raw['latents']
        else:
            val_data = val_data_raw
            
        print(f"Loaded validation data: {val_data.shape}")
        
        # 生成验证集labels
        n_val_samples = len(val_data)
        n_users = 31
        val_samples_per_user = n_val_samples // n_users
        val_labels = []
        for i in range(n_users):
            val_labels.extend([i] * val_samples_per_user)
        for i in range(n_val_samples - len(val_labels)):
            val_labels.append(i % n_users)
        val_labels = torch.tensor(val_labels, dtype=torch.long, device=device)
        
        val_dataset = DomainAdaptiveDataset(
            source_latents=val_data,
            source_labels=val_labels,
            target_latents=data['target_latents'][:10] if data['target_latents'] is not None else None,
            target_labels=data['target_labels'][:10] if data['target_labels'] is not None else None,
            phase='align' if data['target_latents'] is not None else 'pretrain',
            augmentation=False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    print(f"Created dataloader for phase '{phase}':")
    print(f"  Training samples: {len(dataset)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of batches: {len(train_loader)}")
    if val_loader:
        print(f"  Validation batches: {len(val_loader)}")
    
    return train_loader, val_loader


if __name__ == "__main__":
    # 测试数据加载器
    print("Testing data loader...")
    
    # 创建模拟数据
    source_latents = torch.randn(100, 32, 16, 16)
    source_labels = torch.randint(0, 31, (100,))
    target_latents = torch.randn(10, 32, 16, 16)
    target_labels = torch.randint(0, 31, (10,))
    
    # 测试不同阶段
    for phase in ['pretrain', 'align', 'finetune']:
        dataset = DomainAdaptiveDataset(
            source_latents, source_labels,
            target_latents, target_labels,
            phase=phase
        )
        
        print(f"\n{phase} phase:")
        print(f"  Dataset size: {len(dataset)}")
        
        # 获取一个样本
        sample = dataset[0]
        print(f"  Sample keys: {sample.keys()}")
        print(f"  Latent shape: {sample['latent'].shape}")
        print(f"  Class label: {sample['class_label']}")
        print(f"  Domain label: {sample['domain_label']}")
    
    print("\n✅ Data loader test passed!")

