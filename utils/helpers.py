"""
辅助工具函数
包含训练过程中常用的工具函数
"""

import torch
import torch.nn as nn
import numpy as np
import random
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import json
import yaml
from datetime import datetime
import shutil


def set_seed(seed: int = 42):
    """
    设置随机种子，确保实验可重复性
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 确定性操作
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to {seed}")


def count_parameters(model: nn.Module) -> int:
    """
    计算模型参数量
    
    Args:
        model: PyTorch模型
    
    Returns:
        总参数量
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total:,} ({total/1e6:.2f}M)")
    print(f"Trainable parameters: {trainable:,} ({trainable/1e6:.2f}M)")
    
    return total


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    losses: Dict[str, float],
    save_path: str,
    ema_model: Optional[nn.Module] = None,
    additional_info: Optional[Dict] = None
):
    """
    保存模型检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前epoch
        step: 当前步数
        losses: 损失字典
        save_path: 保存路径
        ema_model: EMA模型（如果有）
        additional_info: 额外信息
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'step': step,
        'losses': losses,
        'timestamp': datetime.now().isoformat()
    }
    
    if ema_model is not None:
        checkpoint['ema_model_state_dict'] = ema_model.state_dict()
    
    if additional_info is not None:
        checkpoint.update(additional_info)
    
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint to {save_path}")


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    ema_model: Optional[nn.Module] = None,
    strict: bool = True,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    加载模型检查点
    
    Args:
        model: 模型
        checkpoint_path: 检查点路径
        optimizer: 优化器（可选）
        ema_model: EMA模型（可选）
        strict: 是否严格匹配state dict
        device: 设备
    
    Returns:
        检查点信息字典
    """
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    print(f"Loaded model from {checkpoint_path}")
    
    # 加载优化器状态
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Loaded optimizer state")
    
    # 加载EMA模型
    if ema_model is not None and 'ema_model_state_dict' in checkpoint:
        ema_model.load_state_dict(checkpoint['ema_model_state_dict'], strict=strict)
        print("Loaded EMA model")
    
    # 返回其他信息
    info = {
        'epoch': checkpoint.get('epoch', 0),
        'step': checkpoint.get('step', 0),
        'losses': checkpoint.get('losses', {}),
        'timestamp': checkpoint.get('timestamp', 'unknown')
    }
    
    return info


class EMA:
    """
    指数移动平均（Exponential Moving Average）
    用于稳定训练和提升生成质量
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        """
        Args:
            model: 原始模型
            decay: EMA衰减率
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # 初始化shadow参数
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """更新EMA参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """应用EMA参数到模型"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        """恢复原始参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
    
    def state_dict(self):
        """返回EMA状态字典"""
        return self.shadow
    
    def load_state_dict(self, state_dict):
        """加载EMA状态"""
        self.shadow = state_dict


def create_exp_dir(
    base_dir: str,
    exp_name: Optional[str] = None,
    config: Optional[Dict] = None,
    use_timestamp: bool = False  # 新增参数，控制是否使用时间戳
) -> str:
    """
    创建实验目录
    
    Args:
        base_dir: 基础目录
        exp_name: 实验名称
        config: 配置字典（会保存到目录中）
        use_timestamp: 是否在目录名中添加时间戳
    
    Returns:
        实验目录路径
    """
    if use_timestamp:
        # 生成时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 创建目录名
        if exp_name:
            dir_name = f"{timestamp}_{exp_name}"
        else:
            dir_name = timestamp
        
        # 创建目录
        exp_dir = Path(base_dir) / dir_name
    else:
        # 直接使用base_dir，不创建子文件夹
        exp_dir = Path(base_dir)
    
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    if config:
        config_path = exp_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Saved config to {config_path}")
    
    print(f"Created experiment directory: {exp_dir}")
    
    return str(exp_dir)


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """获取当前学习率"""
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return 0.0


def adjust_learning_rate(
    optimizer: torch.optim.Optimizer,
    step: int,
    warmup_steps: int = 1000,
    base_lr: float = 1e-4,
    min_lr: float = 1e-6,
    schedule: str = 'cosine',
    max_steps: int = 100000
):
    """
    调整学习率
    
    Args:
        optimizer: 优化器
        step: 当前步数
        warmup_steps: 预热步数
        base_lr: 基础学习率
        min_lr: 最小学习率
        schedule: 调度策略 ('constant', 'linear', 'cosine')
        max_steps: 最大步数
    """
    if step < warmup_steps:
        # 线性预热
        lr = base_lr * step / warmup_steps
    else:
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        progress = min(1.0, progress)
        
        if schedule == 'constant':
            lr = base_lr
        elif schedule == 'linear':
            lr = base_lr * (1 - progress) + min_lr * progress
        elif schedule == 'cosine':
            lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * progress))
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr


def clip_grad_norm(
    model: nn.Module,
    max_norm: float,
    norm_type: float = 2.0
) -> float:
    """
    梯度裁剪
    
    Args:
        model: 模型
        max_norm: 最大梯度范数
        norm_type: 范数类型
    
    Returns:
        总梯度范数
    """
    total_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm,
        norm_type
    )
    return total_norm.item()


class AverageMeter:
    """
    计算和存储平均值和当前值
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


def get_device() -> str:
    """获取可用设备"""
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def move_to_device(
    data: Union[torch.Tensor, Dict, list],
    device: str
) -> Union[torch.Tensor, Dict, list]:
    """
    将数据移动到指定设备
    
    Args:
        data: 数据（tensor、字典或列表）
        device: 目标设备
    
    Returns:
        移动后的数据
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    else:
        return data


def save_config(config: Dict, save_path):
    """保存配置文件"""
    # 转换为Path对象以统一处理
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 获取文件扩展名
    suffix = save_path.suffix.lower()
    
    if suffix == '.json':
        with open(save_path, 'w') as f:
            json.dump(config, f, indent=2)
    elif suffix in ['.yaml', '.yml']:
        with open(save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    else:
        raise ValueError(f"Unsupported config format: {suffix}")
    
    print(f"Saved config to {save_path}")


def load_config(config_path) -> Dict:
    """加载配置文件"""
    # 转换为Path对象以统一处理
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    # 获取文件扩展名
    suffix = config_path.suffix.lower()
    
    if suffix == '.json':
        with open(config_path, 'r') as f:
            config = json.load(f)
    elif suffix in ['.yaml', '.yml']:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported config format: {suffix}")
    
    return config


if __name__ == "__main__":
    # 测试辅助函数
    print("Testing helper functions...")
    
    # 测试随机种子
    set_seed(42)
    
    # 测试模型参数计数
    model = nn.Linear(100, 10)
    count_parameters(model)
    
    # 测试EMA
    ema = EMA(model, decay=0.999)
    ema.update()
    
    # 测试平均计
    meter = AverageMeter()
    for i in range(10):
        meter.update(i)
    print(f"Average: {meter.avg}")
    
    print("\n✅ Helper functions test passed!")

