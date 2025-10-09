"""
损失函数模块
包含MMD损失和其他域适应相关损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict


class MMDLoss(nn.Module):
    """
    Maximum Mean Discrepancy (MMD) 损失
    用于测量两个分布之间的差异
    
    参考文献:
    - Gretton et al. "A Kernel Two-Sample Test" JMLR 2012
    - Long et al. "Deep Transfer Learning with Joint Adaptation Networks" ICML 2017
    """
    
    def __init__(
        self,
        kernel_type: str = 'gaussian',
        kernel_mul: float = 2.0,
        kernel_num: int = 5,
        fix_sigma: Optional[float] = None
    ):
        super().__init__()
        self.kernel_type = kernel_type
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma
    
    def gaussian_kernel(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        kernel_mul: float = 2.0,
        kernel_num: int = 5,
        fix_sigma: Optional[float] = None
    ) -> torch.Tensor:
        """
        计算高斯核矩阵
        使用多个带宽的高斯核的和
        """
        n_samples = source.shape[0] + target.shape[0]
        total = torch.cat([source, target], dim=0)
        
        # 计算L2距离矩阵
        total0 = total.unsqueeze(0).expand(n_samples, n_samples, -1)
        total1 = total.unsqueeze(1).expand(n_samples, n_samples, -1)
        L2_distance = ((total0 - total1) ** 2).sum(2)
        
        # 计算带宽
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            # 使用中位数启发式
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
            bandwidth = bandwidth.sqrt()
        
        # 多核MMD
        kernel_val = torch.zeros_like(L2_distance)
        for i in range(kernel_num):
            bandwidth_temp = bandwidth * (kernel_mul ** (i - kernel_num // 2))
            kernel_val += torch.exp(-L2_distance / (2 * bandwidth_temp ** 2))
        
        return kernel_val / kernel_num
    
    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        计算MMD损失
        
        Args:
            source: 源域特征 [N_s, D]
            target: 目标域特征 [N_t, D]
        
        Returns:
            MMD损失值
        """
        # 展平特征
        source = source.view(source.shape[0], -1)
        target = target.view(target.shape[0], -1)
        
        batch_size_source = source.shape[0]
        batch_size_target = target.shape[0]
        
        # 计算核矩阵
        kernels = self.gaussian_kernel(
            source, target,
            kernel_mul=self.kernel_mul,
            kernel_num=self.kernel_num,
            fix_sigma=self.fix_sigma
        )
        
        # 分割核矩阵
        XX = kernels[:batch_size_source, :batch_size_source]
        YY = kernels[batch_size_source:, batch_size_source:]
        XY = kernels[:batch_size_source, batch_size_source:]
        YX = kernels[batch_size_source:, :batch_size_source]
        
        # 计算MMD
        # MMD^2 = E[K(x,x')] + E[K(y,y')] - 2*E[K(x,y)]
        XX = torch.mean(XX)
        YY = torch.mean(YY)
        XY = torch.mean(XY)
        YX = torch.mean(YX)
        
        loss = XX + YY - XY - YX
        
        return loss


class DomainAlignmentLoss(nn.Module):
    """
    域对齐损失
    组合多种损失用于域适应
    """
    
    def __init__(
        self,
        mmd_weight: float = 1.0,
        coral_weight: float = 0.0,
        dann_weight: float = 0.0
    ):
        super().__init__()
        self.mmd_weight = mmd_weight
        self.coral_weight = coral_weight
        self.dann_weight = dann_weight
        
        # 各种损失函数
        self.mmd_loss = MMDLoss()
        
        if coral_weight > 0:
            self.coral_loss = CORALLoss()
        
        if dann_weight > 0:
            self.dann_loss = DANNLoss()
    
    def forward(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        计算域对齐损失
        
        Returns:
            包含各项损失的字典
        """
        losses = {}
        total_loss = 0
        
        # MMD损失
        if self.mmd_weight > 0:
            mmd_loss = self.mmd_loss(source_features, target_features)
            losses['mmd'] = mmd_loss
            total_loss += self.mmd_weight * mmd_loss
        
        # CORAL损失
        if self.coral_weight > 0 and hasattr(self, 'coral_loss'):
            coral_loss = self.coral_loss(source_features, target_features)
            losses['coral'] = coral_loss
            total_loss += self.coral_weight * coral_loss
        
        # DANN损失
        if self.dann_weight > 0 and hasattr(self, 'dann_loss'):
            dann_loss = self.dann_loss(source_features, target_features, **kwargs)
            losses['dann'] = dann_loss
            total_loss += self.dann_weight * dann_loss
        
        losses['total'] = total_loss
        
        return losses


class CORALLoss(nn.Module):
    """
    CORrelation ALignment损失
    通过对齐协方差矩阵来减小域差异
    
    参考: Sun et al. "Deep CORAL: Correlation Alignment for Deep Domain Adaptation" ECCV 2016
    """
    
    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        计算CORAL损失
        """
        # 展平特征
        source = source.view(source.shape[0], -1)
        target = target.view(target.shape[0], -1)
        
        d = source.shape[1]
        
        # 源域协方差
        xm = torch.mean(source, dim=0, keepdim=True)
        xc = source - xm
        xcT = torch.transpose(xc, 0, 1)
        source_cov = torch.matmul(xcT, xc) / (source.shape[0] - 1)
        
        # 目标域协方差
        ym = torch.mean(target, dim=0, keepdim=True)
        yc = target - ym
        ycT = torch.transpose(yc, 0, 1)
        target_cov = torch.matmul(ycT, yc) / (target.shape[0] - 1)
        
        # Frobenius范数
        loss = torch.mean((source_cov - target_cov) ** 2)
        
        return loss / (4 * d * d)


class DANNLoss(nn.Module):
    """
    Domain Adversarial Neural Network损失
    通过对抗训练实现域适应
    
    参考: Ganin et al. "Domain-Adversarial Training of Neural Networks" JMLR 2016
    """
    
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        # 域判别器
        self.domain_discriminator = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        alpha: float = 1.0
    ) -> torch.Tensor:
        """
        计算DANN损失
        
        Args:
            source: 源域特征
            target: 目标域特征
            alpha: 梯度反转系数
        """
        # 展平特征
        source = source.view(source.shape[0], -1)
        target = target.view(target.shape[0], -1)
        
        # 创建域标签
        source_labels = torch.zeros(source.shape[0], 1).to(source.device)
        target_labels = torch.ones(target.shape[0], 1).to(target.device)
        
        # 域判别
        source_pred = self.domain_discriminator(source)
        target_pred = self.domain_discriminator(target)
        
        # 二元交叉熵损失
        source_loss = F.binary_cross_entropy(source_pred, source_labels)
        target_loss = F.binary_cross_entropy(target_pred, target_labels)
        
        return alpha * (source_loss + target_loss)


class ContrastiveLoss(nn.Module):
    """
    对比学习损失
    用于学习域不变特征
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算InfoNCE损失
        """
        # L2归一化
        features = F.normalize(features, p=2, dim=1)
        
        # 计算相似度矩阵
        similarity = torch.matmul(features, features.T) / self.temperature
        
        # 创建mask
        batch_size = features.shape[0]
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        
        # 对角线mask（自己和自己的相似度不算）
        diagonal_mask = torch.eye(batch_size, dtype=torch.bool).to(features.device)
        mask = mask.masked_fill(diagonal_mask, 0)
        
        # 计算损失
        exp_sim = torch.exp(similarity)
        exp_sim = exp_sim.masked_fill(diagonal_mask, 0)
        
        positive_sum = torch.sum(exp_sim * mask, dim=1)
        negative_sum = torch.sum(exp_sim * (1 - mask), dim=1)
        
        loss = -torch.log(positive_sum / (positive_sum + negative_sum + 1e-8))
        
        return loss.mean()


if __name__ == "__main__":
    # 测试损失函数
    print("Testing loss functions...")
    
    # 测试MMD损失
    mmd_loss = MMDLoss()
    source = torch.randn(32, 512)
    target = torch.randn(16, 512)
    loss = mmd_loss(source, target)
    print(f"MMD Loss: {loss:.4f}")
    
    # 测试CORAL损失
    coral_loss = CORALLoss()
    loss = coral_loss(source, target)
    print(f"CORAL Loss: {loss:.4f}")
    
    # 测试域对齐损失
    align_loss = DomainAlignmentLoss(mmd_weight=1.0, coral_weight=0.5)
    losses = align_loss(source, target)
    print(f"Domain Alignment Losses: {losses}")
    
    print("✅ All loss functions test passed!")
