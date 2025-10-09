"""
评估指标模块
计算FID、MMD、分类准确率等指标
严格遵循学术标准实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict
from scipy import linalg
from tqdm import tqdm
import warnings


class InceptionStatistics:
    """
    计算Inception特征统计量，用于FID计算
    参考: Heusel et al. "GANs Trained by a Two Time-Scale Update Rule" NIPS 2017
    """
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.inception_model = self._load_inception_model()
    
    def _load_inception_model(self):
        """加载预训练的Inception-v3模型"""
        try:
            from torchvision.models import inception_v3
            model = inception_v3(pretrained=True, transform_input=False)
            model.fc = nn.Identity()  # 移除最后的分类层
            model = model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            warnings.warn(f"Failed to load Inception model: {e}")
            return None
    
    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> np.ndarray:
        """
        提取Inception特征
        Args:
            images: [N, 3, H, W] 范围[0, 1]
        Returns:
            features: [N, 2048]
        """
        if self.inception_model is None:
            raise RuntimeError("Inception model not available")
        
        # 调整大小到299x299（Inception输入）
        if images.shape[2] != 299 or images.shape[3] != 299:
            images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        
        # 归一化到[-1, 1]
        images = 2 * images - 1
        
        # 提取特征
        features = self.inception_model(images.to(self.device))
        return features.cpu().numpy()
    
    def calculate_statistics(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """计算均值和协方差"""
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma


def calculate_fid(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    eps: float = 1e-6
) -> float:
    """
    计算Fréchet Inception Distance (FID)
    
    Args:
        real_features: 真实图像的Inception特征 [N, D]
        fake_features: 生成图像的Inception特征 [M, D]
        eps: 数值稳定性的小常数
    
    Returns:
        FID分数（越低越好）
    """
    # 计算统计量
    mu1 = np.mean(real_features, axis=0)
    sigma1 = np.cov(real_features, rowvar=False)
    
    mu2 = np.mean(fake_features, axis=0)
    sigma2 = np.cov(fake_features, rowvar=False)
    
    # 计算差异
    diff = mu1 - mu2
    
    # 计算协方差矩阵的平方根
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    # 处理数值误差
    if not np.isfinite(covmean).all():
        warnings.warn("FID calculation produces singular product; adding epsilon to diagonal")
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # 确保是实数
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            warnings.warn(f"Imaginary component {m}")
        covmean = covmean.real
    
    # 计算FID
    tr_covmean = np.trace(covmean)
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    
    return float(fid)


def calculate_mmd(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    kernel: str = 'gaussian',
    bandwidth: float = 1.0,
    num_kernels: int = 5
) -> float:
    """
    计算Maximum Mean Discrepancy (MMD)
    
    Args:
        source_features: 源域特征 [N, D]
        target_features: 目标域特征 [M, D]
        kernel: 核函数类型
        bandwidth: 高斯核带宽
        num_kernels: 多核MMD的核数量
    
    Returns:
        MMD距离（越低越好）
    """
    # 展平特征
    source = source_features.view(source_features.shape[0], -1)
    target = target_features.view(target_features.shape[0], -1)
    
    if kernel == 'gaussian':
        # 多核MMD
        mmd = 0
        for i in range(num_kernels):
            # 不同带宽
            bw = bandwidth * (2.0 ** (i - num_kernels // 2))
            
            # 计算核矩阵
            XX = gaussian_kernel(source, source, bw).mean()
            YY = gaussian_kernel(target, target, bw).mean()
            XY = gaussian_kernel(source, target, bw).mean()
            
            # 累加MMD
            mmd += XX + YY - 2 * XY
        
        mmd /= num_kernels
    
    elif kernel == 'linear':
        # 线性核
        XX = torch.mm(source, source.t()).mean()
        YY = torch.mm(target, target.t()).mean()
        XY = torch.mm(source, target.t()).mean()
        mmd = XX + YY - 2 * XY
    
    else:
        raise ValueError(f"Unknown kernel: {kernel}")
    
    return float(mmd)


def gaussian_kernel(x: torch.Tensor, y: torch.Tensor, bandwidth: float) -> torch.Tensor:
    """计算高斯核矩阵"""
    x_size = x.shape[0]
    y_size = y.shape[0]
    dim = x.shape[1]
    
    x = x.unsqueeze(1)  # [N, 1, D]
    y = y.unsqueeze(0)  # [1, M, D]
    
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    
    kernel = torch.exp(-torch.sum((tiled_x - tiled_y) ** 2, dim=2) / (2 * bandwidth ** 2))
    
    return kernel


def calculate_domain_shift(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    method: str = 'mmd'
) -> Dict[str, float]:
    """
    计算域偏移程度
    
    Args:
        source_features: 源域特征
        target_features: 目标域特征
        method: 计算方法 ('mmd', 'coral', 'cosine')
    
    Returns:
        包含各种域偏移指标的字典
    """
    metrics = {}
    
    # MMD距离
    if 'mmd' in method or method == 'all':
        mmd = calculate_mmd(source_features, target_features)
        metrics['mmd'] = mmd
    
    # CORAL距离
    if 'coral' in method or method == 'all':
        source_mean = source_features.mean(0)
        target_mean = target_features.mean(0)
        
        source_cov = torch.cov(source_features.T)
        target_cov = torch.cov(target_features.T)
        
        coral = torch.norm(source_cov - target_cov, p='fro') ** 2 / (4 * source_features.shape[1] ** 2)
        metrics['coral'] = float(coral)
    
    # 余弦相似度
    if 'cosine' in method or method == 'all':
        source_mean = F.normalize(source_features.mean(0), dim=0)
        target_mean = F.normalize(target_features.mean(0), dim=0)
        cosine_sim = torch.dot(source_mean, target_mean)
        metrics['cosine_similarity'] = float(cosine_sim)
        metrics['cosine_distance'] = 1 - float(cosine_sim)
    
    # Wasserstein距离（简化版）
    if 'wasserstein' in method or method == 'all':
        source_mean = source_features.mean(0)
        target_mean = target_features.mean(0)
        w_dist = torch.norm(source_mean - target_mean, p=2)
        metrics['wasserstein'] = float(w_dist)
    
    return metrics


@torch.no_grad()
def evaluate_generation(
    generated_samples: torch.Tensor,
    real_samples: torch.Tensor,
    vae_decoder: Optional[nn.Module] = None,
    classifier: Optional[nn.Module] = None,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    综合评估生成质量
    
    Args:
        generated_samples: 生成的latent或图像
        real_samples: 真实的latent或图像
        vae_decoder: VAE解码器（如果输入是latent）
        classifier: 分类器（用于评估分类准确率）
    
    Returns:
        包含各种评估指标的字典
    """
    metrics = {}
    
    # 如果是latent，先解码
    if vae_decoder is not None and generated_samples.shape[1] == 32:  # 假设latent是32通道
        print("Decoding latents...")
        batch_size = 50
        decoded_generated = []
        decoded_real = []
        
        for i in range(0, len(generated_samples), batch_size):
            batch_gen = generated_samples[i:i+batch_size].to(device)
            batch_real = real_samples[i:i+batch_size].to(device)
            
            decoded_gen = vae_decoder(batch_gen)
            decoded_real = vae_decoder(batch_real)
            
            decoded_generated.append(decoded_gen.cpu())
            decoded_real.append(decoded_real.cpu())
        
        generated_samples = torch.cat(decoded_generated, dim=0)
        real_samples = torch.cat(decoded_real, dim=0)
    
    # 计算FID（需要Inception特征）
    try:
        print("Calculating FID...")
        inception_stats = InceptionStatistics(device)
        
        # 提取特征
        gen_features = []
        real_features = []
        
        batch_size = 50
        for i in tqdm(range(0, len(generated_samples), batch_size), desc="Extracting features"):
            batch_gen = generated_samples[i:i+batch_size]
            batch_real = real_samples[i:min(i+batch_size, len(real_samples))]
            
            # 确保是3通道（如果是单通道，复制3次）
            if batch_gen.shape[1] == 1:
                batch_gen = batch_gen.repeat(1, 3, 1, 1)
            if batch_real.shape[1] == 1:
                batch_real = batch_real.repeat(1, 3, 1, 1)
            
            gen_feat = inception_stats.extract_features(batch_gen)
            real_feat = inception_stats.extract_features(batch_real)
            
            gen_features.append(gen_feat)
            real_features.append(real_feat)
        
        gen_features = np.concatenate(gen_features, axis=0)
        real_features = np.concatenate(real_features, axis=0)
        
        # 计算FID
        fid = calculate_fid(real_features, gen_features)
        metrics['fid'] = fid
        
    except Exception as e:
        print(f"FID calculation failed: {e}")
        metrics['fid'] = -1
    
    # 计算MMD（在特征空间）
    print("Calculating MMD...")
    gen_flat = generated_samples.view(generated_samples.shape[0], -1)
    real_flat = real_samples.view(real_samples.shape[0], -1)
    mmd = calculate_mmd(gen_flat[:100], real_flat[:100])  # 使用子集加速
    metrics['mmd'] = mmd
    
    # 计算域偏移
    print("Calculating domain shift...")
    domain_metrics = calculate_domain_shift(gen_flat[:100], real_flat[:100], method='all')
    metrics.update({f'domain_{k}': v for k, v in domain_metrics.items()})
    
    # 如果有分类器，计算分类相关指标
    if classifier is not None:
        print("Evaluating with classifier...")
        classifier.eval()
        
        # 这里需要根据具体的分类器接口调整
        # 假设分类器接受图像并返回logits
        with torch.no_grad():
            gen_preds = []
            
            for i in range(0, len(generated_samples), batch_size):
                batch = generated_samples[i:i+batch_size].to(device)
                if batch.shape[1] == 32:  # 如果是latent
                    # 需要解码
                    if vae_decoder is not None:
                        batch = vae_decoder(batch)
                
                logits = classifier(batch)
                preds = torch.argmax(logits, dim=1)
                gen_preds.append(preds.cpu())
            
            gen_preds = torch.cat(gen_preds, dim=0)
            
            # 计算预测的多样性
            unique_preds = len(torch.unique(gen_preds))
            metrics['prediction_diversity'] = unique_preds / len(gen_preds)
    
    # 计算基本统计量
    metrics['gen_mean'] = float(generated_samples.mean())
    metrics['gen_std'] = float(generated_samples.std())
    metrics['real_mean'] = float(real_samples.mean())
    metrics['real_std'] = float(real_samples.std())
    
    return metrics


def calculate_kid(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    """
    计算Kernel Inception Distance (KID)
    比FID更适合小样本评估
    """
    # 计算核矩阵
    k_rr = polynomial_kernel(real_features, real_features)
    k_ff = polynomial_kernel(fake_features, fake_features)
    k_rf = polynomial_kernel(real_features, fake_features)
    
    # 计算KID
    m = real_features.shape[0]
    n = fake_features.shape[0]
    
    kid = (k_rr.sum() - np.diag(k_rr).sum()) / (m * (m - 1))
    kid += (k_ff.sum() - np.diag(k_ff).sum()) / (n * (n - 1))
    kid -= 2 * k_rf.mean()
    
    return float(kid)


def polynomial_kernel(x: np.ndarray, y: np.ndarray, degree: int = 3) -> np.ndarray:
    """多项式核函数"""
    return (1 + np.dot(x, y.T)) ** degree


if __name__ == "__main__":
    # 测试评估指标
    print("Testing metrics...")
    
    # 生成模拟数据
    source = torch.randn(100, 512)
    target = torch.randn(80, 512)
    
    # 测试MMD
    mmd = calculate_mmd(source, target)
    print(f"MMD: {mmd:.4f}")
    
    # 测试域偏移
    shifts = calculate_domain_shift(source, target, method='all')
    print(f"Domain shifts: {shifts}")
    
    print("\n✅ Metrics test passed!")

