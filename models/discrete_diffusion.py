"""
Discrete Diffusion for VQ-VAE
基于VQ-Diffusion和MaskGIT的思想
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import numpy as np
from einops import rearrange, repeat
from tqdm import tqdm


class DiscreteGaussianDiffusion(nn.Module):
    """
    离散扩散模型 - 适用于VQ-VAE的离散latent
    基于Multinomial Diffusion的思想
    """
    
    def __init__(
        self,
        model,  # 预测模型（如Transformer）
        *,
        num_classes: int,  # 离散token数量
        timesteps: int = 1000,
        loss_type: str = 'ce',  # 'ce' or 'focal'
        objective: str = 'pred_x0',  # 'pred_x0' or 'pred_noise'
        schedule_type: str = 'cosine',
        # 域适应相关
        mmd_loss_weight: float = 0.1,
        use_discrete_guidance: bool = True
    ):
        super().__init__()
        
        self.model = model
        self.num_classes = num_classes
        self.timesteps = timesteps
        self.loss_type = loss_type
        self.objective = objective
        self.mmd_loss_weight = mmd_loss_weight
        self.use_discrete_guidance = use_discrete_guidance
        
        # 创建扩散schedule
        if schedule_type == 'cosine':
            alphas = self._cosine_schedule(timesteps)
        else:
            alphas = self._linear_schedule(timesteps)
            
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 转换到离散概率
        self.register_buffer('transition_probs', alphas_cumprod)
        self.register_buffer('transition_probs_prev', alphas_cumprod_prev)
        
        # 用于采样
        posterior_variance = (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod) * (1.0 - alphas)
        self.register_buffer('posterior_variance', posterior_variance)
        
    def _linear_schedule(self, timesteps: int) -> torch.Tensor:
        """线性schedule"""
        return torch.linspace(0.9999, 0.0001, timesteps)
        
    def _cosine_schedule(self, timesteps: int) -> torch.Tensor:
        """余弦schedule"""
        s = 0.008
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
        return torch.clip(alphas, 0.0001, 0.9999)
        
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        前向扩散过程 - 添加离散噪声
        x_start: [B, H, W] 离散indices
        """
        batch_size = x_start.shape[0]
        
        # 获取转换概率
        transition_prob = self.transition_probs[t]  # [B]
        
        # 创建mask - 哪些token要被替换
        mask = torch.rand_like(x_start.float()) > transition_prob.view(-1, 1, 1)
        
        # 随机采样新的token
        random_indices = torch.randint(0, self.num_classes, x_start.shape, device=x_start.device)
        
        # 应用mask
        x_noisy = torch.where(mask, random_indices, x_start)
        
        return x_noisy, mask
        
    def p_losses(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        domain_labels: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """计算训练损失"""
        
        # 前向扩散
        x_noisy, mask = self.q_sample(x_start, t)
        
        # 模型预测
        if self.objective == 'pred_x0':
            # 预测原始token
            model_out = self.model(x_noisy, t, domain_labels, class_labels)
            target = x_start
        else:
            # 预测噪声（被mask的位置）
            model_out = self.model(x_noisy, t, domain_labels, class_labels)
            target = mask.long()
            
        # 计算损失
        if self.loss_type == 'ce':
            # Cross entropy loss
            loss = F.cross_entropy(
                model_out.reshape(-1, self.num_classes),
                target.reshape(-1),
                reduction='none'
            )
            # 只计算被mask位置的损失
            loss = (loss.reshape_as(mask) * mask).sum() / mask.sum()
        else:
            # Focal loss for class imbalance
            ce_loss = F.cross_entropy(
                model_out.reshape(-1, self.num_classes),
                target.reshape(-1),
                reduction='none'
            )
            p = torch.exp(-ce_loss)
            loss = ((1 - p) ** 2 * ce_loss).mean()
            
        return {
            'loss': loss,
            'x_noisy': x_noisy,
            'mask': mask
        }
        
    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        t: int,
        domain_labels: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0
    ) -> torch.Tensor:
        """单步去噪"""
        
        batched_t = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        
        # 模型预测
        model_out = self.model(x, batched_t, domain_labels, class_labels)
        
        if self.use_discrete_guidance and guidance_scale > 1.0:
            # Classifier-free guidance for discrete case
            uncond_out = self.model(x, batched_t, None, None)
            model_out = uncond_out + guidance_scale * (model_out - uncond_out)
            
        # 获取概率分布
        probs = F.softmax(model_out, dim=1)  # [B, num_classes, H, W]
        
        # 采样新的token
        if t > 0:
            # 添加噪声
            noise_level = self.posterior_variance[t]
            probs = probs + noise_level * torch.randn_like(probs)
            
        # 采样
        x_new = torch.argmax(probs, dim=1)
        
        return x_new
        
    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int],
        domain_labels: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None,
        guidance_scale: float = 3.0
    ) -> torch.Tensor:
        """生成样本"""
        
        device = next(self.model.parameters()).device
        batch_size = shape[0]
        
        # 从完全随机开始
        x = torch.randint(0, self.num_classes, shape, device=device)
        
        # 逐步去噪
        for t in tqdm(reversed(range(self.timesteps)), desc='Sampling', total=self.timesteps):
            x = self.p_sample(x, t, domain_labels, class_labels, guidance_scale)
            
        return x
        
    def forward(
        self,
        x: torch.Tensor,
        domain_labels: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """训练前向传播"""
        
        b = x.shape[0]
        device = x.device
        
        # 随机采样时间步
        t = torch.randint(0, self.timesteps, (b,), device=device).long()
        
        # 计算损失
        losses = self.p_losses(x, t, domain_labels, class_labels)
        
        return losses


class MaskGitDiffusion(nn.Module):
    """
    MaskGIT风格的离散扩散
    更简单高效，适合VQ-VAE
    """
    
    def __init__(
        self,
        model,
        *,
        num_classes: int,
        mask_schedule: str = 'cosine',
        T: int = 10  # 采样步数（比DDPM少很多）
    ):
        super().__init__()
        
        self.model = model
        self.num_classes = num_classes
        self.T = T
        
        # Mask schedule
        if mask_schedule == 'cosine':
            self.mask_schedule = lambda r: np.cos(r * np.pi / 2)
        else:
            self.mask_schedule = lambda r: 1 - r
            
    def forward(
        self,
        indices: torch.Tensor,
        domain_labels: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """训练"""
        
        b, h, w = indices.shape
        device = indices.device
        
        # 随机mask比例
        r = torch.rand(1).item()
        num_masked = int(self.mask_schedule(r) * h * w)
        
        # 随机选择要mask的位置
        mask_indices = torch.randperm(h * w)[:num_masked]
        mask = torch.zeros(b, h * w, dtype=torch.bool, device=device)
        mask[:, mask_indices] = True
        mask = mask.reshape(b, h, w)
        
        # Mask tokens
        masked_indices = indices.clone()
        masked_indices[mask] = self.num_classes  # Special mask token
        
        # 预测
        logits = self.model(masked_indices, domain_labels, class_labels)
        
        # 损失
        loss = F.cross_entropy(
            logits.permute(0, 2, 3, 1)[mask],
            indices[mask]
        )
        
        return {'loss': loss}
        
    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int],
        domain_labels: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """并行采样 - 非常快"""
        
        b, h, w = shape
        device = next(self.model.parameters()).device
        
        # 从全mask开始
        indices = torch.full(shape, self.num_classes, device=device)
        
        for t in range(self.T):
            # 预测所有masked位置
            logits = self.model(indices, domain_labels, class_labels)
            
            # 获取概率
            probs = F.softmax(logits / temperature, dim=1)
            
            # 采样
            sampled = torch.multinomial(
                probs.permute(0, 2, 3, 1).reshape(-1, self.num_classes),
                1
            ).reshape(b, h, w)
            
            # 计算confidence
            confidence = probs.max(dim=1)[0]
            
            # 确定本轮要生成的数量
            n = int((1 - t / self.T) * h * w)
            
            # 选择confidence最高的n个位置
            _, idx = confidence.reshape(b, -1).topk(n, dim=1)
            
            # 更新这些位置
            for i in range(b):
                indices[i].reshape(-1)[idx[i]] = sampled[i].reshape(-1)[idx[i]]
                
        return indices
