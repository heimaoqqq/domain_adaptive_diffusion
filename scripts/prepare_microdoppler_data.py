"""
专门为微多普勒数据集设计的数据准备脚本
处理organized_gait_dataset的特定结构
"""

import torch
import argparse
import sys
from pathlib import Path
import json
from typing import Dict, List, Tuple

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from prepare_data import ImageDataset, load_vae_model, encode_dataset, compute_statistics, validate_latents
from torch.utils.data import DataLoader
import numpy as np


def prepare_gait_datasets(
    dataset_root: str,
    source_gait: str,
    target_gait: str,
    users: List[int] = None,
    train_ratio: float = 0.8
) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    """
    为步态数据集准备源域和目标域
    
    Args:
        dataset_root: organized_gait_dataset路径
        source_gait: 源域步态类型（如'Normal_line'）
        target_gait: 目标域步态类型（如'Normal_free'）
        users: 要包含的用户列表（None表示所有用户）
        train_ratio: 训练集比例
    
    Returns:
        source_paths: {train: Path, val: Path}
        target_paths: {train: Path, val: Path}
    """
    dataset_root = Path(dataset_root)
    source_dir = dataset_root / source_gait
    target_dir = dataset_root / target_gait
    
    # 验证路径并提供详细信息
    print(f"\n检查数据集路径:")
    print(f"  源域路径: {source_dir}")
    print(f"  存在: {source_dir.exists()}")
    
    if source_dir.exists():
        # 列出目录内容
        contents = list(source_dir.iterdir())
        print(f"  目录内容（前10项）:")
        for item in contents[:10]:
            print(f"    - {item.name} ({'DIR' if item.is_dir() else 'FILE'})")
    else:
        raise ValueError(f"Source directory not found: {source_dir}")
    
    print(f"\n  目标域路径: {target_dir}")
    print(f"  存在: {target_dir.exists()}")
    
    if target_dir.exists():
        contents = list(target_dir.iterdir())
        print(f"  目录内容（前10项）:")
        for item in contents[:10]:
            print(f"    - {item.name} ({'DIR' if item.is_dir() else 'FILE'})")
    else:
        raise ValueError(f"Target directory not found: {target_dir}")
    
    # 收集用户数据
    source_users = {}
    target_users = {}
    
    # 遍历源域 - 图像在子文件夹中
    print("\n收集源域数据（仅.jpg文件）...")
    for user_dir in source_dir.iterdir():
        if user_dir.is_dir() and user_dir.name.startswith("ID_"):
            # 从文件夹名提取用户ID，如 ID_1 -> 1
            try:
                user_id = int(user_dir.name.split("_")[1])
                if 1 <= user_id <= 31:  # 验证范围
                    if users is None or user_id in users:
                        # 收集该用户文件夹中的所有图像（只收集jpg格式）
                        user_images = list(user_dir.glob("*.jpg"))
                        if user_images:
                            source_users[user_id] = user_images
                            print(f"  用户 ID_{user_id}: {len(user_images)} 张图像")
            except Exception as e:
                print(f"  警告: 无法处理文件夹 {user_dir.name}: {e}")
    
    print(f"\n源域共找到 {len(source_users)} 个用户，{sum(len(imgs) for imgs in source_users.values())} 张图像")
    
    # 遍历目标域 - 图像在子文件夹中
    print("\n收集目标域数据（仅.jpg文件）...")
    for user_dir in target_dir.iterdir():
        if user_dir.is_dir() and user_dir.name.startswith("ID_"):
            try:
                user_id = int(user_dir.name.split("_")[1])
                if 1 <= user_id <= 31:  # 验证范围
                    if users is None or user_id in users:
                        # 收集该用户文件夹中的所有图像（只收集jpg格式）
                        user_images = list(user_dir.glob("*.jpg"))
                        if user_images:
                            target_users[user_id] = user_images
                            print(f"  用户 ID_{user_id}: {len(user_images)} 张图像")
            except Exception as e:
                print(f"  警告: 无法处理文件夹 {user_dir.name}: {e}")
    
    print(f"\n目标域共找到 {len(target_users)} 个用户，{sum(len(imgs) for imgs in target_users.values())} 张图像")
    
    # Few-shot设置：源域使用所有数据，目标域只使用少量数据
    SHOTS_PER_USER = 5  # Few-shot设置
    SELECTION_STRATEGY = 'diversity'  # 使用diversity策略选择最具代表性的样本
    CLASSIFIER_PATH = r'D:\Ysj\新建文件夹\VA-VAE\improved_classifier\best_improved_classifier.pth'
    
    print(f"\n🔬 域自适应扩散模型训练数据准备:")
    print(f"  - 源域: 所有数据用于训练扩散模型")
    print(f"  - 目标域: 每用户{SHOTS_PER_USER}张用于MMD域对齐")
    print(f"  - 选择策略: {SELECTION_STRATEGY} (使用源域分类器)")
    
    # 加载分类器（只加载一次）
    from diversity_selector import load_classifier
    print(f"\n加载源域分类器用于diversity选择...")
    classifier = load_classifier(CLASSIFIER_PATH, device='cuda')
    
    source_train = []
    source_val = []
    target_train = []
    target_val = []
    
    for user_id in sorted(source_users.keys()):
        # 源域：使用所有数据
        user_images = sorted(source_users[user_id])
        split_idx = int(len(user_images) * train_ratio)
        source_train.extend(user_images[:split_idx])
        source_val.extend(user_images[split_idx:])
        
        # 目标域：Few-shot设置（每用户只用5张）
        if user_id in target_users:
            user_images = sorted(target_users[user_id])
            
            # 使用diversity策略选择few-shot样本
            from diversity_selector import select_diverse_samples
            
            few_shot_images = select_diverse_samples(
                user_images, 
                SHOTS_PER_USER,
                classifier=classifier,  # 使用已加载的分类器
                device='cuda',
                seed=42 + user_id
            )
            
            # 这些数据用于计算MMD损失
            target_train.extend(few_shot_images)
    
    print(f"\n数据划分统计:")
    print(f"源域 - 训练: {len(source_train)}, 验证: {len(source_val)}")
    print(f"目标域 - MMD对齐: {len(target_train)} ({len(target_train)//31 if len(target_train) > 0 else 0}张/用户)")
    print(f"  说明：源域数据用于训练扩散模型主损失")
    print(f"       目标域数据用于计算MMD损失实现域对齐")
    
    return {
        'source_train': source_train,
        'source_val': source_val,
        'target_train': target_train,  # Few-shot样本
        'target_val': []  # Few-shot设置下不需要目标域验证集
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Prepare micro-Doppler data for domain adaptation')
    
    # 数据集配置 - 与run_universal_guidance.py保持一致
    parser.add_argument(
        '--dataset_root',
        type=str,
        default=r'D:\Ysj\新建文件夹\VA-VAE\dataset\organized_gait_dataset\kaggle\working\organized_gait_dataset',
        help='Root path to organized_gait_dataset (will append gait type)'
    )
    parser.add_argument(
        '--source_gait',
        type=str,
        default='Normal_line',
        help='Source domain gait type'
    )
    parser.add_argument(
        '--target_gait',
        type=str,
        default='Normal_free',
        help='Target domain gait type'
    )
    parser.add_argument(
        '--users',
        type=int,
        nargs='+',
        default=None,
        help='Specific users to include (default: all)'
    )
    
    # VAE配置
    parser.add_argument(
        '--vae_checkpoint',
        type=str,
        default=r'D:\Ysj\新建文件夹\VA-VAE\VAE\vavae-stage3-epoch26-val_rec_loss0.0000.ckpt',
        help='Path to VAE checkpoint'
    )
    parser.add_argument(
        '--vae_script',
        type=str,
        default=r'D:\Ysj\新建文件夹\VA-VAE\simplified_vavae.py',
        help='Path to simplified_vavae.py script'
    )
    
    # 输出配置
    parser.add_argument(
        '--output_dir',
        type=str,
        default='../data/microdoppler_processed',
        help='Directory to save processed latents'
    )
    
    # 处理参数
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for encoding'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.8,
        help='Training set ratio'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    # 设置种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 准备数据集路径
    print(f"准备数据集: {args.source_gait} -> {args.target_gait}")
    data_splits = prepare_gait_datasets(
        args.dataset_root,
        args.source_gait,
        args.target_gait,
        args.users,
        args.train_ratio
    )
    
    # 加载VAE
    print("\n加载VAE模型...")
    vae = load_vae_model(args.vae_script, args.device, checkpoint_path=args.vae_checkpoint)
    
    # 处理每个数据分割
    all_statistics = {}
    
    for split_name, image_paths in data_splits.items():
        if len(image_paths) == 0:
            print(f"\n跳过空分割: {split_name}")
            continue
            
        print(f"\n处理 {split_name}: {len(image_paths)} 张图像")
        
        # 创建临时数据集
        class PathDataset(torch.utils.data.Dataset):
            def __init__(self, paths, transform=None):
                self.paths = paths
                self.transform = transform
                
            def __len__(self):
                return len(self.paths)
                
            def __getitem__(self, idx):
                from PIL import Image
                img_path = self.paths[idx]
                image = Image.open(img_path).convert('RGB')
                
                # 调整大小
                image = image.resize((256, 256), Image.LANCZOS)
                
                # 转换为tensor并归一化
                image = np.array(image).astype(np.float32) / 255.0
                image = torch.from_numpy(image).permute(2, 0, 1)
                image = 2.0 * image - 1.0  # 归一化到[-1, 1]
                
                # 提取用户ID作为标签
                filename = Path(img_path).stem
                user_id = 0  # 默认值
                
                if 'ID_' in filename:
                    try:
                        user_id = int(filename.split('ID_')[1].split('_')[0]) - 1  # ID_1 -> 0 (0-indexed)
                    except:
                        pass
                elif 'user_' in filename:
                    try:
                        user_id = int(filename.split('user_')[1].split('_')[0]) - 1  # 0-indexed
                    except:
                        pass
                
                return image, user_id
        
        dataset = PathDataset(image_paths)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,  # Windows下必须设为0避免序列化错误
            pin_memory=True
        )
        
        # 编码数据
        latents, labels = encode_dataset(
            dataloader, vae, args.device,
            desc=f"编码 {split_name}"
        )
        
        # 计算统计量
        stats = compute_statistics(latents)
        all_statistics[split_name] = stats
        
        # 验证质量
        if not validate_latents(latents):
            print(f"警告: {split_name} latents验证失败!")
        
        # 保存数据
        torch.save(latents, output_dir / f'{split_name}_latents.pt')
        torch.save(labels, output_dir / f'{split_name}_labels.pt')
        torch.save(stats, output_dir / f'{split_name}_stats.pt')
        print(f"保存 {split_name}: {latents.shape}")
    
    # 计算域偏移
    if 'source_train' in all_statistics and 'target_train' in all_statistics:
        print("\n域偏移分析:")
        source_mean = all_statistics['source_train']['mean']
        target_mean = all_statistics['target_train']['mean']
        mean_diff = (source_mean - target_mean).abs().mean()
        print(f"  均值差异: {mean_diff:.4f}")
        
        source_std = all_statistics['source_train']['std']
        target_std = all_statistics['target_train']['std']
        std_diff = (source_std - target_std).abs().mean()
        print(f"  标准差差异: {std_diff:.4f}")
    
    # 保存元数据
    metadata = {
        'source_gait': args.source_gait,
        'target_gait': args.target_gait,
        'users': args.users or 'all',
        'train_ratio': args.train_ratio,
        'splits': {
            split_name: len(paths) 
            for split_name, paths in data_splits.items()
        },
        'image_size': 256,
        'latent_shape': list(latents.shape[1:]) if 'latents' in locals() else None,
        'vae_checkpoint': str(args.vae_checkpoint),
        'timestamp': str(np.datetime64('now'))
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # 创建简化的链接（用于训练脚本）
    if (output_dir / 'source_train_latents.pt').exists():
        # 创建标准命名的符号链接或复制
        import shutil
        shutil.copy(output_dir / 'source_train_latents.pt', output_dir / 'source_latents.pt')
        shutil.copy(output_dir / 'source_train_labels.pt', output_dir / 'source_labels.pt')
        shutil.copy(output_dir / 'target_train_latents.pt', output_dir / 'target_latents.pt')
        shutil.copy(output_dir / 'target_train_labels.pt', output_dir / 'target_labels.pt')
    
    print("\n" + "="*60)
    print("数据准备完成!")
    print(f"输出目录: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
