"""
微多普勒数据准备脚本
支持随机和基于分类器的diversity样本选择策略
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
from sklearn.cluster import KMeans
import torchvision.transforms as transforms

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


def load_classifier(checkpoint_path: str, device: str) -> nn.Module:
    """加载源域分类器"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 尝试从domain_adaptation_experiment导入
    sys.path.append(str(Path(__file__).parent.parent.parent / 'domain_adaptation_experiment'))
    try:
        # 根据checkpoint判断模型类型
        if 'resnet18' in checkpoint_path.lower() or 'improved' in checkpoint_path.lower():
            # 使用标准ResNet18
            import torchvision.models as models
            model = models.resnet18(pretrained=False)
            num_features = model.fc.in_features
            
            # 根据checkpoint获取类别数
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                # 查找fc层的输出维度
                fc_weight_key = 'fc.weight' if 'fc.weight' in state_dict else 'module.fc.weight'
                if fc_weight_key in state_dict:
                    num_classes = state_dict[fc_weight_key].shape[0]
                else:
                    num_classes = 31  # 默认31个用户
            else:
                num_classes = 31
                
            model.fc = nn.Linear(num_features, num_classes)
            
            # 加载权重
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            raise ValueError("未知的分类器类型")
            
    except Exception as e:
        print(f"加载分类器失败: {e}")
        print("将使用随机选择策略")
        return None
    
    model.to(device)
    model.eval()
    return model


def extract_features(model: nn.Module, image_paths: List[str], device: str) -> np.ndarray:
    """使用分类器提取特征"""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    features = []
    model.eval()
    
    # 临时移除fc层以获取特征
    if hasattr(model, 'fc'):
        original_fc = model.fc
        model.fc = nn.Identity()
    
    with torch.no_grad():
        for img_path in image_paths:
            # 加载和预处理图像
            image = Image.open(img_path).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)
            
            # 提取特征
            feature = model(image)
            features.append(feature.cpu().numpy().flatten())
    
    # 恢复fc层
    if hasattr(model, 'fc'):
        model.fc = original_fc
    
    return np.array(features)


def select_fewshot_samples(user_data: Dict[str, List], n_shot: int = 5, 
                         strategy: str = 'random', classifier: nn.Module = None,
                         device: str = 'cuda') -> List:
    """为每个用户选择few-shot样本
    
    Args:
        user_data: 用户数据字典
        n_shot: 每个用户选择的样本数
        strategy: 选择策略 ('random' 或 'diversity')
        classifier: 源域分类器（用于diversity策略）
        device: 设备
    """
    fewshot_data = []
    
    for user_id, data in tqdm(user_data.items(), desc="选择few-shot样本"):
        if strategy == 'diversity' and classifier is not None:
            # 使用diversity策略
            if len(data) <= n_shot:
                selected = data
            else:
                # 提取所有样本的特征
                image_paths = [d[0] for d in data]
                features = extract_features(classifier, image_paths, device)
                
                # K-means聚类
                n_clusters = min(n_shot, len(features))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                kmeans.fit(features)
                
                # 每个簇选择最接近中心的样本
                selected_indices = []
                for i in range(n_clusters):
                    cluster_mask = (kmeans.labels_ == i)
                    cluster_indices = np.where(cluster_mask)[0]
                    
                    if len(cluster_indices) > 0:
                        # 计算到簇中心的距离
                        center = kmeans.cluster_centers_[i]
                        distances = np.sum((features[cluster_indices] - center) ** 2, axis=1)
                        closest_idx = cluster_indices[np.argmin(distances)]
                        selected_indices.append(closest_idx)
                
                # 根据索引选择样本
                selected = [data[i] for i in selected_indices[:n_shot]]
        else:
            # 随机选择
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
    parser.add_argument('--selection_strategy', type=str, default='random',
                       choices=['random', 'diversity'],
                       help='目标域样本选择策略')
    parser.add_argument('--classifier_checkpoint', type=str, 
                       default='/kaggle/input/best-improved-classifier-pth/best_improved_classifier.pth',
                       help='源域分类器路径（用于diversity策略）')
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
    
    # 加载分类器（如果使用diversity策略）
    classifier = None
    if args.selection_strategy == 'diversity':
        print(f"\n加载源域分类器用于diversity选择...")
        if os.path.exists(args.classifier_checkpoint):
            classifier = load_classifier(args.classifier_checkpoint, device)
            if classifier is not None:
                print("✅ 分类器加载成功，使用diversity策略")
            else:
                print("⚠️ 分类器加载失败，退回到随机选择")
                args.selection_strategy = 'random'
        else:
            print(f"⚠️ 分类器文件不存在: {args.classifier_checkpoint}")
            print("退回到随机选择策略")
            args.selection_strategy = 'random'
    
    # 划分数据
    source_train, source_val = split_data(source_data, args.train_ratio)
    target_fewshot = select_fewshot_samples(
        target_data, 
        args.n_shot,
        strategy=args.selection_strategy,
        classifier=classifier,
        device=device
    )
    
    print(f"\n数据划分:")
    print(f"源域 - 训练: {len(source_train)}, 验证: {len(source_val)}")
    print(f"目标域 - Few-shot: {len(target_fewshot)} ({args.n_shot}张/用户)")
    print(f"选择策略: {args.selection_strategy}")
    
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
        'selection_strategy': args.selection_strategy,
        'latent_shape': [4, 64, 64],  # KL-VAE输出
        'scale_factor': getattr(vae, 'scale_factor', 0.18215)
    }
    
    torch.save(stats, output_dir / 'data_stats.pt')
    
    print("\n✅ 数据准备完成!")
    print(f"输出目录: {output_dir}")


if __name__ == '__main__':
    main()
