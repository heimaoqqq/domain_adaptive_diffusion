"""
Diversity选择策略实现
使用源域分类器选择最具代表性的few-shot样本
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from typing import List, Optional


def load_classifier(checkpoint_path: str, device: str = 'cuda') -> nn.Module:
    """
    加载源域分类器（根据improved_classifier_training.py的实际架构）
    
    Args:
        checkpoint_path: 分类器checkpoint路径
        device: 计算设备
    
    Returns:
        classifier: 加载的分类器模型
    """
    print(f"  📦 加载分类器: {checkpoint_path}")
    
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 定义与训练时完全一致的分类器架构
    class ImprovedClassifier(nn.Module):
        def __init__(self, num_classes=31, dropout_rate=0.3):
            super().__init__()
            from torchvision import models
            
            # ResNet-18 backbone（与训练时一致）
            self.backbone = models.resnet18(pretrained=False)
            # 移除最后的分类层
            self.backbone.fc = nn.Identity()
            feature_dim = 512
            
            # 分类头（与improved_classifier_training.py第517-524行一致）
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(feature_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=False),
                nn.Dropout(dropout_rate * 0.5),
                nn.Linear(256, num_classes)
            )
            
            # 对比学习投影头（与训练时一致）
            self.projection_head = nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.ReLU(inplace=False),
                nn.Linear(128, 64)
            )
        
        def forward(self, x, return_features=False):
            features = self.backbone(x)  # [B, 512]
            
            if return_features:
                projected = self.projection_head(features)
                return features, projected
            
            logits = self.classifier(features)
            return logits
    
    # 创建模型
    model = ImprovedClassifier(num_classes=31)
    
    # 加载权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"  ✅ 分类器加载成功")
    return model


def select_diverse_samples(
    image_paths: List[Path],
    num_samples: int,
    classifier: Optional[nn.Module] = None,
    classifier_path: Optional[str] = None,
    device: str = 'cuda',
    seed: int = 42
) -> List[Path]:
    """
    使用diversity策略选择最具代表性的样本
    
    Args:
        image_paths: 所有可选图像路径列表
        num_samples: 要选择的样本数
        classifier: 已加载的分类器（可选）
        classifier_path: 分类器路径（如果没有提供classifier）
        device: 计算设备
        seed: 随机种子
    
    Returns:
        selected_paths: 选中的图像路径列表
    """
    
    if len(image_paths) <= num_samples:
        return image_paths[:num_samples]
    
    # 加载分类器
    if classifier is None:
        if classifier_path is None:
            # 使用默认路径
            classifier_path = r'D:\Ysj\新建文件夹\VA-VAE\improved_classifier\best_improved_classifier.pth'
        classifier = load_classifier(classifier_path, device)
    
    print(f"  🎯 使用K-means diversity策略选择{num_samples}个样本...")
    
    # 图像预处理（与训练分类器时一致）
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 提取特征
    features = []
    classifier.eval()
    
    with torch.no_grad():
        for img_path in image_paths:
            # 加载图像
            image = Image.open(img_path).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)
            
            # 提取特征（使用完整模型的backbone）
            # backbone已包含全局平均池化，输出是512维向量
            feature = classifier.backbone(image)  # [1, 512]
            
            # 转换为numpy
            feature = feature.squeeze().cpu().numpy()  # [512]
            features.append(feature)
    
    features = np.array(features)
    print(f"    特征维度: {features.shape}")
    
    # K-means聚类
    n_clusters = min(num_samples, len(features))
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    kmeans.fit(features)
    
    # 每个簇选择最接近中心的样本
    selected_indices = []
    for i in range(n_clusters):
        cluster_center = kmeans.cluster_centers_[i]
        
        # 计算到簇中心的距离
        distances = np.linalg.norm(features - cluster_center, axis=1)
        
        # 找到该簇中的样本
        cluster_mask = (kmeans.labels_ == i)
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_indices) > 0:
            # 选择最接近簇中心的样本
            closest_idx = cluster_indices[np.argmin(distances[cluster_indices])]
            selected_indices.append(closest_idx)
    
    # 如果选中的样本不足，补充
    if len(selected_indices) < num_samples:
        remaining_indices = [i for i in range(len(image_paths)) 
                            if i not in selected_indices]
        additional = remaining_indices[:num_samples - len(selected_indices)]
        selected_indices.extend(additional)
    
    selected_indices = sorted(selected_indices[:num_samples])
    selected_paths = [image_paths[i] for i in selected_indices]
    
    print(f"    ✅ 选中样本索引: {selected_indices}")
    
    return selected_paths
