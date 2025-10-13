"""
支持VAE的图像数据集
基于官方image_datasets.py修改，添加VAE编码支持
"""

import math
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import os

from vae_wrapper import VAEInterface


def load_data_vae(
    *,
    data_dir,
    batch_size,
    image_size,
    vae_interface=None,
    use_cached_latents=False,
    cached_latents_dir=None,
    class_cond=False,
    deterministic=False,
    random_flip=True,
):
    """
    创建数据生成器，支持VAE编码
    
    Args:
        data_dir: 数据目录
        batch_size: 批大小
        image_size: 图像尺寸
        vae_interface: VAE接口实例
        use_cached_latents: 是否使用预编码的latents
        cached_latents_dir: 预编码latents目录
        class_cond: 是否使用类别条件
        deterministic: 是否确定性顺序
        random_flip: 是否随机翻转
    """
    if not data_dir:
        raise ValueError("未指定数据目录")
    
    # 获取所有图像文件
    all_files = _list_image_files_recursively(data_dir)
    
    # 处理类别
    classes = None
    if class_cond:
        # 从文件夹名提取类别（微多普勒数据集格式）
        class_names = []
        for path in all_files:
            # 获取父文件夹名作为类别
            parent_dir = os.path.basename(os.path.dirname(path))
            class_names.append(parent_dir)
        
        # 创建类别映射
        unique_classes = sorted(set(class_names))
        class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        classes = [class_to_idx[cls] for cls in class_names]
        
        print(f"发现 {len(unique_classes)} 个类别: {unique_classes}")
    
    # 创建数据集
    dataset = ImageDatasetVAE(
        resolution=image_size,
        image_paths=all_files,
        classes=classes,
        vae_interface=vae_interface,
        use_cached_latents=use_cached_latents,
        cached_latents_dir=cached_latents_dir,
        random_flip=random_flip,
    )
    
    # 创建数据加载器
    if deterministic:
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0,  # VAE编码时使用0避免多进程问题
            drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0,
            drop_last=True
        )
    
    # 无限循环生成
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    """递归列出所有图像文件"""
    results = []
    
    # 支持的图像格式
    valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    
    for root, dirs, files in os.walk(data_dir):
        for file in sorted(files):
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_extensions:
                full_path = os.path.join(root, file)
                results.append(full_path)
    
    return results


class ImageDatasetVAE(Dataset):
    """
    支持VAE编码的图像数据集
    """
    
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        vae_interface=None,
        use_cached_latents=False,
        cached_latents_dir=None,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.image_paths = image_paths
        self.classes = classes
        self.vae_interface = vae_interface
        self.use_cached_latents = use_cached_latents
        self.cached_latents_dir = cached_latents_dir
        self.random_flip = random_flip
        
        # 如果使用VAE但没有提供接口，创建一个
        if not use_cached_latents and vae_interface is None:
            print("创建默认VAE接口...")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.vae_interface = VAEInterface(device=device)
        
        # 计算latent尺寸
        if self.vae_interface:
            self.latent_size = self.vae_interface.get_latent_size(resolution)
            print(f"Latent尺寸: {self.latent_size}x{self.latent_size} (图像: {resolution}x{resolution})")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """获取一个样本"""
        
        # 准备输出字典
        out_dict = {}
        if self.classes is not None:
            out_dict["y"] = np.array(self.classes[idx], dtype=np.int64)
        
        # 如果使用缓存的latents
        if self.use_cached_latents:
            # 构建latent文件路径
            img_path = self.image_paths[idx]
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            latent_path = os.path.join(self.cached_latents_dir, f"{img_name}.npy")
            
            if os.path.exists(latent_path):
                # 加载预编码的latent
                latent = np.load(latent_path).astype(np.float32)
                
                # 标准化到std≈1.0用于扩散
                latent = latent / 0.18215
                
                # 随机翻转
                if self.random_flip and random.random() < 0.5:
                    latent = latent[:, :, ::-1].copy()
                
                return latent, out_dict
            else:
                print(f"警告: 缺少缓存latent: {latent_path}")
                # 继续使用实时编码
        
        # 实时编码路径
        # 加载图像
        path = self.image_paths[idx]
        with open(path, 'rb') as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")
        
        # 调整大小
        pil_image = center_crop_arr(pil_image, self.resolution)
        
        # 转换为tensor [H, W, C] -> [C, H, W]
        arr = np.array(pil_image).astype(np.float32)
        
        # 随机翻转（在编码前）
        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]
        
        # 归一化到[0, 1] for VAE
        arr = arr / 255.0
        arr = np.transpose(arr, [2, 0, 1])
        
        # 使用VAE编码
        if self.vae_interface is not None:
            # 转为tensor并添加batch维度
            img_tensor = torch.from_numpy(arr).unsqueeze(0)
            
            # 编码到latent
            with torch.no_grad():
                latent = self.vae_interface.encode_batch(img_tensor)
                
                # 标准化到std≈1.0用于扩散
                latent = latent / 0.18215
                
                # 转回numpy并移除batch维度
                latent = latent.squeeze(0).cpu().numpy()
            
            return latent.astype(np.float32), out_dict
        else:
            # 如果没有VAE，返回原始图像（用于测试）
            # 归一化到[-1, 1]（官方Guided-Diffusion的范围）
            arr = arr * 2.0 - 1.0
            return arr, out_dict


def center_crop_arr(pil_image, image_size):
    """中心裁剪并调整图像大小"""
    # 如果是numpy数组，直接返回
    if isinstance(pil_image, np.ndarray):
        return pil_image
    
    # 逐步缩小到目标尺寸附近
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    
    # 缩放到目标尺寸
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    
    # 中心裁剪
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def test_data_loading():
    """测试数据加载"""
    print("=" * 60)
    print("测试VAE数据加载")
    print("=" * 60)
    
    # 设置测试参数
    data_dir = "dataset/organized_gait_dataset/Normal_free"  # 使用一个类别测试
    
    if not os.path.exists(data_dir):
        print(f"⚠️ 测试数据目录不存在: {data_dir}")
        print("   使用模拟数据")
        return
    
    # 创建VAE接口
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vae_interface = VAEInterface(device=device)
    
    # 创建数据加载器
    data_gen = load_data_vae(
        data_dir=data_dir,
        batch_size=4,
        image_size=64,
        vae_interface=vae_interface,
        class_cond=False,  # 单类别测试
        random_flip=True,
    )
    
    # 获取一个批次
    print("\n获取一个批次...")
    batch, cond = next(data_gen)
    
    print(f"✓ 批次形状: {batch.shape}")
    print(f"  数据范围: [{batch.min():.2f}, {batch.max():.2f}]")
    print(f"  数据std: {batch.std():.4f}")
    
    if cond:
        if "y" in cond:
            print(f"  类别标签: {cond['y']}")
    
    # 测试解码回图像
    print("\n测试解码回图像...")
    batch_tensor = torch.from_numpy(batch).to(device)
    
    # 反标准化到VAE尺度
    batch_scaled = batch_tensor * 0.18215
    
    # 解码
    with torch.no_grad():
        images = vae_interface.decode_batch(batch_scaled)
    
    print(f"✓ 解码图像形状: {images.shape}")
    print(f"  图像范围: [{images.min():.2f}, {images.max():.2f}]")
    
    print("\n" + "=" * 60)
    print("数据加载测试完成")
    print("=" * 60)


if __name__ == "__main__":
    test_data_loading()

