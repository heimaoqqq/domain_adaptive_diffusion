"""
综合评估脚本
评估域适应扩散模型的生成质量和域适应效果
支持多种评估指标和可视化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import argparse
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm
import json
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils import (
    set_seed, get_device,
    compute_mmd, compute_fid_features, compute_cosine_similarity
)
from utils.visualization import plot_tsne_domains, plot_feature_distributions


class DomainAdaptationEvaluator:
    """
    域适应评估器
    全面评估生成样本的质量和域适应效果
    """
    
    def __init__(
        self,
        device: str = 'cuda',
        feature_extractor: Optional[nn.Module] = None
    ):
        """
        Args:
            device: 设备
            feature_extractor: 特征提取器（如分类器的特征层）
        """
        self.device = device
        self.feature_extractor = feature_extractor
        
        # 如果没有提供特征提取器，使用简单的CNN
        if self.feature_extractor is None:
            self.feature_extractor = self._create_simple_feature_extractor()
    
    def _create_simple_feature_extractor(self) -> nn.Module:
        """创建简单的特征提取器"""
        class SimpleFeatureExtractor(nn.Module):
            def __init__(self):
                super().__init__()
                # 简单的CNN特征提取器
                self.features = nn.Sequential(
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(128, 64)
                )
            
            def forward(self, x):
                return self.features(x)
        
        return SimpleFeatureExtractor().to(self.device)
    
    @torch.no_grad()
    def extract_features(
        self,
        latents: torch.Tensor,
        batch_size: int = 100
    ) -> torch.Tensor:
        """
        从latents提取特征
        """
        self.feature_extractor.eval()
        
        features = []
        num_batches = (len(latents) + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(latents))
            
            batch = latents[start_idx:end_idx].to(self.device)
            feat = self.feature_extractor(batch)
            
            # L2归一化
            feat = F.normalize(feat, p=2, dim=1)
            features.append(feat.cpu())
        
        return torch.cat(features, dim=0)
    
    def compute_domain_metrics(
        self,
        source_latents: torch.Tensor,
        target_latents: torch.Tensor,
        generated_latents: torch.Tensor
    ) -> Dict[str, float]:
        """
        计算域适应相关指标
        """
        metrics = {}
        
        # 提取特征
        print("Extracting features...")
        source_features = self.extract_features(source_latents)
        target_features = self.extract_features(target_latents)
        generated_features = self.extract_features(generated_latents)
        
        # 1. MMD距离
        print("Computing MMD distances...")
        metrics['mmd_source_target'] = compute_mmd(
            source_features, target_features
        ).item()
        
        metrics['mmd_generated_target'] = compute_mmd(
            generated_features, target_features
        ).item()
        
        metrics['mmd_generated_source'] = compute_mmd(
            generated_features, source_features
        ).item()
        
        # 相对位置（0=源域, 1=目标域）
        metrics['relative_position'] = (
            metrics['mmd_generated_target'] / 
            (metrics['mmd_source_target'] + 1e-8)
        )
        
        # 2. 余弦相似度
        print("Computing cosine similarities...")
        
        # 计算中心点
        source_center = source_features.mean(dim=0, keepdim=True)
        target_center = target_features.mean(dim=0, keepdim=True)
        generated_center = generated_features.mean(dim=0, keepdim=True)
        
        # 中心相似度
        metrics['cosine_center_gen_target'] = F.cosine_similarity(
            generated_center, target_center
        ).item()
        
        metrics['cosine_center_gen_source'] = F.cosine_similarity(
            generated_center, source_center
        ).item()
        
        metrics['cosine_center_source_target'] = F.cosine_similarity(
            source_center, target_center
        ).item()
        
        # 3. 分布统计
        print("Computing distribution statistics...")
        
        # 特征范数
        metrics['feat_norm_source'] = source_features.norm(dim=1).mean().item()
        metrics['feat_norm_target'] = target_features.norm(dim=1).mean().item()
        metrics['feat_norm_generated'] = generated_features.norm(dim=1).mean().item()
        
        # 特征方差
        metrics['feat_std_source'] = source_features.std().item()
        metrics['feat_std_target'] = target_features.std().item()
        metrics['feat_std_generated'] = generated_features.std().item()
        
        # 4. 多样性指标
        if len(generated_features) > 20:
            # 内部多样性（随机分两组计算MMD）
            perm = torch.randperm(len(generated_features))
            half = len(generated_features) // 2
            
            group1 = generated_features[perm[:half]]
            group2 = generated_features[perm[half:2*half]]
            
            metrics['diversity_mmd'] = compute_mmd(group1, group2).item()
            
            # 最近邻多样性
            distances = torch.cdist(generated_features, generated_features)
            # 排除自身（对角线）
            distances.fill_diagonal_(float('inf'))
            min_distances = distances.min(dim=1)[0]
            metrics['diversity_nn_mean'] = min_distances.mean().item()
            metrics['diversity_nn_std'] = min_distances.std().item()
        
        return metrics, {
            'source': source_features,
            'target': target_features,
            'generated': generated_features
        }
    
    def compute_classifier_metrics(
        self,
        generated_latents: torch.Tensor,
        labels: torch.Tensor,
        classifier: nn.Module
    ) -> Dict[str, float]:
        """
        使用分类器评估生成质量
        """
        metrics = {}
        classifier.eval()
        
        # 如果需要从latent解码到图像
        # 这里假设分类器直接处理latents
        
        all_preds = []
        all_probs = []
        
        batch_size = 100
        num_batches = (len(generated_latents) + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(generated_latents))
                
                batch = generated_latents[start_idx:end_idx].to(self.device)
                logits = classifier(batch)
                probs = F.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
                
                all_preds.append(preds.cpu())
                all_probs.append(probs.cpu())
        
        all_preds = torch.cat(all_preds)
        all_probs = torch.cat(all_probs)
        
        # 计算准确率
        if labels is not None and len(labels) == len(all_preds):
            accuracy = (all_preds == labels).float().mean().item()
            metrics['classifier_accuracy'] = accuracy
        
        # 计算置信度
        max_probs = all_probs.max(dim=1)[0]
        metrics['classifier_confidence_mean'] = max_probs.mean().item()
        metrics['classifier_confidence_std'] = max_probs.std().item()
        
        # 计算熵（不确定性）
        entropy = -(all_probs * torch.log(all_probs + 1e-8)).sum(dim=1)
        metrics['classifier_entropy_mean'] = entropy.mean().item()
        metrics['classifier_entropy_std'] = entropy.std().item()
        
        return metrics
    
    def visualize_domains(
        self,
        features: Dict[str, torch.Tensor],
        output_path: str,
        method: str = 'tsne',
        perplexity: int = 30
    ):
        """
        可视化域分布
        """
        # 合并所有特征
        all_features = []
        all_labels = []
        all_domains = []
        
        for domain_idx, (domain_name, domain_features) in enumerate(features.items()):
            all_features.append(domain_features)
            all_labels.extend([0] * len(domain_features))  # 假设同一类
            all_domains.extend([domain_idx] * len(domain_features))
        
        all_features = torch.cat(all_features, dim=0).numpy()
        all_domains = np.array(all_domains)
        
        # 降维
        print(f"Computing {method.upper()} visualization...")
        if method == 'tsne':
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        elif method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        embeddings = reducer.fit_transform(all_features)
        
        # 绘图
        plt.figure(figsize=(10, 8))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        domain_names = list(features.keys())
        
        for domain_idx, domain_name in enumerate(domain_names):
            mask = all_domains == domain_idx
            plt.scatter(
                embeddings[mask, 0],
                embeddings[mask, 1],
                c=colors[domain_idx],
                label=domain_name,
                alpha=0.6,
                s=50
            )
        
        plt.title(f'Domain Distribution ({method.upper()})', fontsize=16)
        plt.xlabel(f'{method.upper()} Component 1', fontsize=12)
        plt.ylabel(f'{method.upper()} Component 2', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to {output_path}")
    
    def generate_report(
        self,
        metrics: Dict[str, float],
        output_path: str
    ):
        """
        生成评估报告
        """
        report = []
        report.append("="*60)
        report.append("Domain Adaptation Evaluation Report")
        report.append("="*60)
        report.append("")
        
        # 域偏移分析
        report.append("1. Domain Shift Analysis")
        report.append("-"*40)
        report.append(f"Source ↔ Target distance: {metrics.get('mmd_source_target', 0):.4f}")
        report.append(f"Generated → Target distance: {metrics.get('mmd_generated_target', 0):.4f}")
        report.append(f"Generated → Source distance: {metrics.get('mmd_generated_source', 0):.4f}")
        
        rel_pos = metrics.get('relative_position', 0)
        report.append(f"\nRelative position: {rel_pos:.1%}")
        if rel_pos < 0.3:
            report.append("✅ Excellent: Very close to target domain")
        elif rel_pos < 0.5:
            report.append("✓ Good: Closer to target than source")
        elif rel_pos < 0.7:
            report.append("⚠ Fair: In transition region")
        else:
            report.append("❌ Poor: Still close to source domain")
        
        # 相似度分析
        report.append("\n2. Similarity Analysis")
        report.append("-"*40)
        cos_target = metrics.get('cosine_center_gen_target', 0)
        cos_source = metrics.get('cosine_center_gen_source', 0)
        report.append(f"Generated ↔ Target similarity: {cos_target:.4f}")
        report.append(f"Generated ↔ Source similarity: {cos_source:.4f}")
        
        if cos_target > cos_source:
            report.append("✓ Generated samples are more similar to target")
        else:
            report.append("⚠ Generated samples are more similar to source")
        
        # 多样性分析
        if 'diversity_mmd' in metrics:
            report.append("\n3. Diversity Analysis")
            report.append("-"*40)
            report.append(f"Internal diversity (MMD): {metrics['diversity_mmd']:.4f}")
            report.append(f"Nearest neighbor distance: {metrics.get('diversity_nn_mean', 0):.4f} ± {metrics.get('diversity_nn_std', 0):.4f}")
        
        # 分类器评估（如果有）
        if 'classifier_accuracy' in metrics:
            report.append("\n4. Classifier Evaluation")
            report.append("-"*40)
            report.append(f"Accuracy: {metrics['classifier_accuracy']:.1%}")
            report.append(f"Confidence: {metrics['classifier_confidence_mean']:.3f} ± {metrics['classifier_confidence_std']:.3f}")
            report.append(f"Entropy: {metrics['classifier_entropy_mean']:.3f} ± {metrics['classifier_entropy_std']:.3f}")
        
        # 保存报告
        report_text = "\n".join(report)
        
        with open(output_path, 'w') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\nReport saved to {output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Evaluate domain adaptation quality')
    
    # 数据路径
    parser.add_argument(
        '--source_latents',
        type=str,
        required=True,
        help='Path to source domain latents'
    )
    parser.add_argument(
        '--target_latents',
        type=str,
        required=True,
        help='Path to target domain latents'
    )
    parser.add_argument(
        '--generated_latents',
        type=str,
        required=True,
        help='Path to generated latents'
    )
    
    # 可选：分类器评估
    parser.add_argument(
        '--classifier_path',
        type=str,
        default=None,
        help='Path to classifier for evaluation'
    )
    parser.add_argument(
        '--generated_labels',
        type=str,
        default=None,
        help='Path to labels for generated samples'
    )
    
    # 输出配置
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/evaluation',
        help='Directory to save evaluation results'
    )
    
    # 可视化配置
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization plots'
    )
    parser.add_argument(
        '--vis_method',
        type=str,
        default='tsne',
        choices=['tsne', 'pca'],
        help='Visualization method'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=1000,
        help='Maximum samples for visualization'
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
    
    # 加载数据
    print("\nLoading latents...")
    source_latents = torch.load(args.source_latents)
    target_latents = torch.load(args.target_latents)
    generated_latents = torch.load(args.generated_latents)
    
    # 限制样本数量（用于可视化）
    if args.max_samples and len(source_latents) > args.max_samples:
        indices = torch.randperm(len(source_latents))[:args.max_samples]
        source_latents = source_latents[indices]
    
    if args.max_samples and len(target_latents) > args.max_samples:
        indices = torch.randperm(len(target_latents))[:args.max_samples]
        target_latents = target_latents[indices]
    
    if args.max_samples and len(generated_latents) > args.max_samples:
        indices = torch.randperm(len(generated_latents))[:args.max_samples]
        generated_latents = generated_latents[indices]
    
    print(f"Source samples: {len(source_latents)}")
    print(f"Target samples: {len(target_latents)}")
    print(f"Generated samples: {len(generated_latents)}")
    
    # 创建评估器
    evaluator = DomainAdaptationEvaluator(device=device)
    
    # 计算域适应指标
    print("\nComputing domain adaptation metrics...")
    metrics, features = evaluator.compute_domain_metrics(
        source_latents,
        target_latents,
        generated_latents
    )
    
    # 分类器评估（如果提供）
    if args.classifier_path:
        print("\nEvaluating with classifier...")
        # 这里需要实际加载分类器
        # classifier = load_classifier(args.classifier_path, device)
        # labels = torch.load(args.generated_labels) if args.generated_labels else None
        # classifier_metrics = evaluator.compute_classifier_metrics(
        #     generated_latents, labels, classifier
        # )
        # metrics.update(classifier_metrics)
    
    # 保存指标
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # 生成报告
    evaluator.generate_report(metrics, output_dir / 'evaluation_report.txt')
    
    # 可视化（如果需要）
    if args.visualize:
        print("\nGenerating visualizations...")
        
        # t-SNE/PCA可视化
        evaluator.visualize_domains(
            features,
            output_dir / f'domain_distribution_{args.vis_method}.png',
            method=args.vis_method
        )
        
        # 特征分布直方图
        plt.figure(figsize=(12, 4))
        
        for idx, (name, feat) in enumerate(features.items()):
            plt.subplot(1, 3, idx + 1)
            feat_norms = feat.norm(dim=1).numpy()
            plt.hist(feat_norms, bins=30, alpha=0.7, density=True)
            plt.title(f'{name.capitalize()} Feature Norms')
            plt.xlabel('L2 Norm')
            plt.ylabel('Density')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_distributions.png', dpi=150)
        plt.close()
    
    print("\n" + "="*60)
    print("Evaluation completed!")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
