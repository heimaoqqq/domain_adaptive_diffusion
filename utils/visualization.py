"""
可视化模块
用于生成训练过程和结果的可视化
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Union
from pathlib import Path
import torchvision
from torchvision.utils import make_grid
import warnings
warnings.filterwarnings('ignore')


def visualize_samples(
    samples: torch.Tensor,
    nrow: int = 8,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    normalize: bool = True,
    value_range: Optional[tuple] = None
) -> None:
    """
    可视化生成的样本
    
    Args:
        samples: [N, C, H, W] tensor
        nrow: 每行显示的图像数
        title: 图表标题
        save_path: 保存路径
        show: 是否显示图像
        normalize: 是否归一化到[0,1]
        value_range: 值范围，如(-1, 1)
    """
    if samples.dim() == 3:
        samples = samples.unsqueeze(1)  # 添加通道维度
    
    # 确保是CPU tensor
    samples = samples.cpu()
    
    # 归一化
    if normalize:
        if value_range is not None:
            samples = (samples - value_range[0]) / (value_range[1] - value_range[0])
        else:
            samples = (samples - samples.min()) / (samples.max() - samples.min())
    
    # 创建网格
    grid = make_grid(samples, nrow=nrow, padding=2, pad_value=1)
    
    # 转换为numpy
    grid_np = grid.permute(1, 2, 0).numpy()
    
    # 如果是单通道，复制为3通道用于显示
    if grid_np.shape[2] == 1:
        grid_np = np.repeat(grid_np, 3, axis=2)
    
    # 创建图像
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(grid_np)
    ax.axis('off')
    
    if title:
        ax.set_title(title, fontsize=16)
    
    plt.tight_layout()
    
    # 保存
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    # 显示
    if show:
        plt.show()
    else:
        plt.close()


def plot_training_curves(
    losses: Dict[str, List[float]],
    save_path: Optional[str] = None,
    show: bool = True,
    smooth_factor: float = 0.9
) -> None:
    """
    绘制训练曲线
    
    Args:
        losses: 损失字典，如 {'total_loss': [...], 'mmd_loss': [...]}
        save_path: 保存路径
        show: 是否显示
        smooth_factor: 平滑因子
    """
    fig, axes = plt.subplots(len(losses), 1, figsize=(10, 4 * len(losses)))
    
    if len(losses) == 1:
        axes = [axes]
    
    for idx, (name, values) in enumerate(losses.items()):
        ax = axes[idx]
        
        # 原始值
        ax.plot(values, alpha=0.3, label='Raw')
        
        # 平滑值
        if smooth_factor > 0:
            smoothed = smooth_curve(values, smooth_factor)
            ax.plot(smoothed, label='Smoothed')
        
        ax.set_title(name.replace('_', ' ').title())
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def smooth_curve(values: List[float], factor: float = 0.9) -> List[float]:
    """指数移动平均平滑"""
    smoothed = []
    last = values[0] if values else 0
    
    for value in values:
        smoothed_value = last * factor + value * (1 - factor)
        smoothed.append(smoothed_value)
        last = smoothed_value
    
    return smoothed


def save_image_grid(
    images: torch.Tensor,
    save_path: str,
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = True,
    value_range: Optional[tuple] = None
) -> None:
    """
    保存图像网格
    
    Args:
        images: [N, C, H, W] tensor
        save_path: 保存路径
        nrow: 每行图像数
        padding: 间隔
        normalize: 是否归一化
        value_range: 值范围
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # 归一化
    if normalize:
        if value_range is not None:
            images = (images - value_range[0]) / (value_range[1] - value_range[0])
        else:
            images = (images - images.min()) / (images.max() - images.min())
    
    # 保存
    torchvision.utils.save_image(
        images,
        save_path,
        nrow=nrow,
        padding=padding,
        normalize=False  # 已经归一化了
    )
    print(f"Saved image grid to {save_path}")


def plot_latent_distribution(
    latents: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    title: str = "Latent Distribution",
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    可视化latent空间分布（使用t-SNE）
    
    Args:
        latents: [N, ...] latent vectors
        labels: [N] 标签（用于着色）
        title: 标题
        save_path: 保存路径
        show: 是否显示
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        warnings.warn("sklearn not available, skipping t-SNE visualization")
        return
    
    # 展平latent
    latents_flat = latents.view(latents.shape[0], -1)
    
    # 如果维度太高，先用PCA降维
    if latents_flat.shape[1] > 50:
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=50)
            latents_flat = pca.fit_transform(latents_flat.cpu().numpy())
        except:
            latents_flat = latents_flat.cpu().numpy()
    else:
        latents_flat = latents_flat.cpu().numpy()
    
    # t-SNE
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latents_flat)-1))
    latents_2d = tsne.fit_transform(latents_flat)
    
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if labels is not None:
        # 按标签着色
        unique_labels = torch.unique(labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(
                latents_2d[mask.cpu(), 0],
                latents_2d[mask.cpu(), 1],
                c=[colors[i]],
                label=f'Class {label}',
                alpha=0.7,
                s=30
            )
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax.scatter(latents_2d[:, 0], latents_2d[:, 1], alpha=0.7, s=30)
    
    ax.set_title(title)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved t-SNE plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_domain_comparison(
    source_samples: torch.Tensor,
    target_samples: torch.Tensor,
    generated_samples: Optional[torch.Tensor] = None,
    nrow: int = 8,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    比较不同域的样本
    
    Args:
        source_samples: 源域样本
        target_samples: 目标域样本
        generated_samples: 生成的样本（可选）
        nrow: 每行图像数
        save_path: 保存路径
        show: 是否显示
    """
    # 限制样本数量
    n = min(nrow * 4, len(source_samples), len(target_samples))
    source_samples = source_samples[:n]
    target_samples = target_samples[:n]
    
    if generated_samples is not None:
        generated_samples = generated_samples[:n]
        n_rows = 3
    else:
        n_rows = 2
    
    # 创建子图
    fig, axes = plt.subplots(n_rows, 1, figsize=(15, 5 * n_rows))
    if n_rows == 2:
        axes = list(axes) + [None]
    
    # 源域
    grid_source = make_grid(source_samples, nrow=nrow, padding=2, normalize=True)
    axes[0].imshow(grid_source.permute(1, 2, 0).cpu())
    axes[0].set_title('Source Domain', fontsize=14)
    axes[0].axis('off')
    
    # 目标域
    grid_target = make_grid(target_samples, nrow=nrow, padding=2, normalize=True)
    axes[1].imshow(grid_target.permute(1, 2, 0).cpu())
    axes[1].set_title('Target Domain', fontsize=14)
    axes[1].axis('off')
    
    # 生成样本
    if generated_samples is not None:
        grid_gen = make_grid(generated_samples, nrow=nrow, padding=2, normalize=True)
        axes[2].imshow(grid_gen.permute(1, 2, 0).cpu())
        axes[2].set_title('Generated (Target Domain)', fontsize=14)
        axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved domain comparison to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def create_interpolation_grid(
    model,
    latent1: torch.Tensor,
    latent2: torch.Tensor,
    steps: int = 10,
    decode_fn: Optional[callable] = None
) -> torch.Tensor:
    """
    创建两个latent之间的插值网格
    
    Args:
        model: 生成模型
        latent1: 起始latent
        latent2: 结束latent
        steps: 插值步数
        decode_fn: 解码函数（如VAE decoder）
    
    Returns:
        插值结果 [steps, C, H, W]
    """
    # 线性插值
    alphas = torch.linspace(0, 1, steps)
    interpolated = []
    
    for alpha in alphas:
        inter = (1 - alpha) * latent1 + alpha * latent2
        interpolated.append(inter)
    
    interpolated = torch.stack(interpolated)
    
    # 如果需要解码
    if decode_fn is not None:
        with torch.no_grad():
            interpolated = decode_fn(interpolated)
    
    return interpolated


def plot_metrics_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    比较不同方法的指标
    
    Args:
        metrics_dict: {'method1': {'fid': ..., 'mmd': ...}, ...}
        save_path: 保存路径
        show: 是否显示
    """
    methods = list(metrics_dict.keys())
    metric_names = list(next(iter(metrics_dict.values())).keys())
    
    n_metrics = len(metric_names)
    n_methods = len(methods)
    
    # 创建柱状图
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(n_metrics)
    width = 0.8 / n_methods
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_methods))
    
    for i, method in enumerate(methods):
        values = [metrics_dict[method].get(m, 0) for m in metric_names]
        offset = (i - n_methods / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=method, color=colors[i])
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Value')
    ax.set_title('Method Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved metrics comparison to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    # 测试可视化函数
    print("Testing visualization functions...")
    
    # 生成测试数据
    samples = torch.randn(16, 1, 32, 32)
    
    # 测试样本可视化
    visualize_samples(samples, title="Test Samples", show=False)
    
    # 测试训练曲线
    losses = {
        'total_loss': [np.random.random() for _ in range(100)],
        'mmd_loss': [np.random.random() * 0.1 for _ in range(100)]
    }
    plot_training_curves(losses, show=False)
    
    print("✅ Visualization test passed!")

