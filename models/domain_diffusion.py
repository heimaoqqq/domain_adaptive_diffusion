"""
Domain Adaptive Diffusion Model
扩展denoising-diffusion-pytorch的GaussianDiffusion
添加域适应能力和MMD损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Any
from denoising_diffusion_pytorch import GaussianDiffusion
from tqdm import tqdm
import numpy as np
from einops import reduce


class DomainAdaptiveDiffusion(GaussianDiffusion):
    """
    域适应扩散模型
    在标准高斯扩散基础上添加:
    1. 域条件生成
    2. MMD对齐损失
    3. 三阶段训练策略
    """
    
    def __init__(
        self,
        model,
        *,
        image_size: int = 64,  # latent size (256/4 for KL-VAE)
        timesteps: int = 1000,
        sampling_timesteps: Optional[int] = None,
        objective: str = 'pred_v',  # 'pred_noise' | 'pred_x0' | 'pred_v'
        beta_schedule: str = 'cosine',
        schedule_fn_kwargs: dict = dict(),
        ddim_sampling_eta: float = 0.0,
        auto_normalize: bool = True,
        offset_noise_strength: float = 0.0,
        min_snr_loss_weight: bool = False,
        min_snr_gamma: float = 5,
        # 域适应特定参数
        mmd_loss_weight: float = 0.1,
        mmd_kernel_bandwidth: float = 1.0,
        domain_balance_ratio: float = 0.8,  # 源域数据比例
        use_ema: bool = True,
        # 兼容参数（不传递给父类）
        loss_type: str = 'l2',  # 保留接口但不使用
    ):
        # GaussianDiffusion不支持loss_type参数，直接移除
        super().__init__(
            model,
            image_size=image_size,
            timesteps=timesteps,
            sampling_timesteps=sampling_timesteps,
            objective=objective,
            beta_schedule=beta_schedule,
            schedule_fn_kwargs=schedule_fn_kwargs,
            ddim_sampling_eta=ddim_sampling_eta,
            auto_normalize=auto_normalize,
            offset_noise_strength=offset_noise_strength,
            min_snr_loss_weight=min_snr_loss_weight,
            min_snr_gamma=min_snr_gamma
        )
        
        # 保存loss_type供自定义使用
        self.loss_type = loss_type
        
        # 保存父类可能不暴露的属性
        self.min_snr_loss_weight = min_snr_loss_weight
        self.min_snr_gamma = min_snr_gamma
        self.auto_normalize = auto_normalize
        self.sampling_timesteps = sampling_timesteps or timesteps // 4  # 默认为1/4的时间步
        self.objective = objective
        self.offset_noise_strength = offset_noise_strength
        
        # 保存必要的属性以供采样使用
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.channels = model.channels if hasattr(model, 'channels') else 4  # VAE latent channels
        self.image_size = image_size
        
        # 域适应参数
        self.mmd_loss_weight = mmd_loss_weight
        self.mmd_kernel_bandwidth = mmd_kernel_bandwidth
        self.domain_balance_ratio = domain_balance_ratio
        
        # EMA模型（可选）
        self.use_ema = use_ema
        if use_ema:
            self.ema_model = self._create_ema_model()
            
        # 训练阶段跟踪
        self.training_phase = 'pretrain'  # 'pretrain' | 'align' | 'finetune'
        
    def _create_ema_model(self):
        """创建EMA模型用于更稳定的生成"""
        import copy
        ema_model = copy.deepcopy(self.model)
        ema_model.requires_grad_(False)
        return ema_model
    
    def update_ema(self, decay: float = 0.999):
        """更新EMA模型"""
        if not self.use_ema:
            return
            
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)
    
    def compute_mmd_loss(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        kernel: str = 'gaussian'
    ) -> torch.Tensor:
        """
        计算Maximum Mean Discrepancy损失
        用于域对齐
        
        参考: Gretton et al. "A Kernel Two-Sample Test" JMLR 2012
        """
        if target_features.shape[0] == 0:
            return torch.tensor(0.0, device=source_features.device)
        
        # 展平特征
        source_features = source_features.view(source_features.shape[0], -1)
        target_features = target_features.view(target_features.shape[0], -1)
        
        # 高斯核函数
        def gaussian_kernel(x, y, bandwidth=1.0):
            """计算高斯核矩阵"""
            n_x = x.shape[0]
            n_y = y.shape[0]
            
            x = x.unsqueeze(1)  # [n_x, 1, d]
            y = y.unsqueeze(0)  # [1, n_y, d]
            
            distances = torch.sum((x - y) ** 2, dim=-1)  # [n_x, n_y]
            kernel_matrix = torch.exp(-distances / (2 * bandwidth ** 2))
            
            return kernel_matrix
        
        # 计算MMD
        if kernel == 'gaussian':
            # 源域内核矩阵
            k_ss = gaussian_kernel(source_features, source_features, self.mmd_kernel_bandwidth)
            k_ss = k_ss.mean()
            
            # 目标域内核矩阵
            k_tt = gaussian_kernel(target_features, target_features, self.mmd_kernel_bandwidth)
            k_tt = k_tt.mean()
            
            # 跨域核矩阵
            k_st = gaussian_kernel(source_features, target_features, self.mmd_kernel_bandwidth)
            k_st = k_st.mean()
            
            # MMD^2 = E[k(s,s')] + E[k(t,t')] - 2*E[k(s,t)]
            mmd_squared = k_ss + k_tt - 2 * k_st
            
        else:
            raise ValueError(f"Unknown kernel type: {kernel}")
        
        # 返回MMD (not squared for gradient stability)
        return torch.sqrt(torch.clamp(mmd_squared, min=1e-10))
    
    def p_losses(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        *,
        noise: Optional[torch.Tensor] = None,
        offset_noise_strength: Optional[float] = None,
        class_labels: Optional[torch.Tensor] = None,
        domain_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算训练损失
        扩展父类方法，添加域对齐损失
        """
        b, c, h, w = x_start.shape
        noise = torch.randn_like(x_start) if noise is None else noise
        
        # 添加offset noise（可选）
        offset_noise_strength = self.offset_noise_strength if offset_noise_strength is None else offset_noise_strength
        if offset_noise_strength > 0:
            offset_noise = torch.randn(b, c, 1, 1, device=x_start.device)
            noise = noise + offset_noise_strength * offset_noise
        
        # 前向扩散过程 q(x_t | x_0)
        x = self.q_sample(x_start=x_start, t=t, noise=noise)
        
        # 模型预测
        if hasattr(self, 'self_condition') and self.self_condition and np.random.random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()
        else:
            x_self_cond = None
        
        # 前向传播（带条件）
        model_out = self.model(x, t, class_labels, domain_labels, x_self_cond)
        
        # 计算目标
        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')
        
        # 重建损失
        if self.loss_type == 'l1':
            loss = F.l1_loss(model_out, target, reduction='none')
        elif self.loss_type == 'l2':
            loss = F.mse_loss(model_out, target, reduction='none')
        elif self.loss_type == 'huber':
            loss = F.smooth_l1_loss(model_out, target, reduction='none')
        else:
            raise ValueError(f'unknown loss type {self.loss_type}')
        
        loss = reduce(loss, 'b ... -> b', 'mean')
        
        # 应用min SNR加权（可选）
        if self.min_snr_loss_weight and hasattr(self, 'alphas_cumprod'):
            snr = self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])
            min_snr_gamma = self.min_snr_gamma
            weight = torch.clamp(snr, max=min_snr_gamma) / snr
            loss = loss * weight
        
        # 基础损失
        base_loss = loss.mean()
        
        # MMD域对齐损失（仅在align阶段）
        mmd_loss = torch.tensor(0.0, device=x_start.device)
        if self.training_phase == 'align' and domain_labels is not None:
            # 分离源域和目标域
            source_mask = (domain_labels == 0)
            target_mask = (domain_labels == 1)
            
            if source_mask.any() and target_mask.any():
                # 预测干净图像用于MMD
                with torch.no_grad():
                    pred_x0 = self.predict_start_from_noise(x, t, model_out)
                
                # 计算MMD损失
                mmd_loss = self.compute_mmd_loss(
                    pred_x0[source_mask],
                    pred_x0[target_mask]
                )
        
        # 总损失
        total_loss = base_loss + self.mmd_loss_weight * mmd_loss
        
        # 返回详细的损失信息（用于logging）
        self.last_loss_dict = {
            'total_loss': total_loss.item(),
            'base_loss': base_loss.item(),
            'mmd_loss': mmd_loss.item()
        }
        
        return total_loss
    
    def p_losses_with_details(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        *,
        noise: Optional[torch.Tensor] = None,
        offset_noise_strength: Optional[float] = None,
        class_labels: Optional[torch.Tensor] = None,
        domain_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算训练损失（详细版本）
        返回各项损失的字典而不是单一的总损失
        """
        # 调用原始的p_losses方法
        total_loss = self.p_losses(
            x_start, t,
            noise=noise,
            offset_noise_strength=offset_noise_strength,
            class_labels=class_labels,
            domain_labels=domain_labels
        )
        
        # 返回详细损失字典
        # 注意：last_loss_dict已经在p_losses中更新了
        return {
            'total_loss': total_loss,
            'base_loss': self.last_loss_dict.get('base_loss', total_loss.item()),
            'mmd_loss': self.last_loss_dict.get('mmd_loss', 0.0)
        }
    
    def set_training_phase(self, phase: str):
        """
        设置训练阶段
        'pretrain': 仅源域预训练
        'align': 域对齐训练
        'finetune': 目标域微调
        """
        assert phase in ['pretrain', 'align', 'finetune']
        self.training_phase = phase
        print(f"Training phase set to: {phase}")
    
    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 16,
        class_labels: Optional[torch.Tensor] = None,
        domain_labels: Optional[torch.Tensor] = None,
        guidance_scale: float = 3.0,
        return_all_timesteps: bool = False
    ) -> torch.Tensor:
        """
        生成样本
        支持类别和域条件，以及classifier-free guidance
        """
        device = self.device
        shape = (batch_size, self.channels, self.image_size, self.image_size)
        
        # 准备条件
        if class_labels is None:
            class_labels = torch.randint(0, 31, (batch_size,), device=device)
        
        if domain_labels is None:
            domain_labels = torch.ones(batch_size, dtype=torch.long, device=device)  # 默认目标域
        
        # 使用EMA模型（如果有）
        model = self.ema_model if self.use_ema and hasattr(self, 'ema_model') else self.model
        
        # DDIM采样
        return self.ddim_sample(
            model,
            shape,
            class_labels=class_labels,
            domain_labels=domain_labels,
            guidance_scale=guidance_scale,
            return_all_timesteps=return_all_timesteps
        )
    
    @torch.no_grad()
    def ddim_sample(
        self,
        model,
        shape,
        class_labels=None,
        domain_labels=None,
        guidance_scale=1.0,
        return_all_timesteps=False,
        eta=0.0
    ):
        """
        DDIM采样实现
        支持classifier-free guidance
        """
        batch, device = shape[0], self.device
        
        # 采样步数
        times = torch.linspace(-1, self.num_timesteps - 1, steps=self.sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        
        # 初始噪声
        img = torch.randn(shape, device=device)
        imgs = [img]
        
        for time, time_next in tqdm(time_pairs, desc='DDIM sampling'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            
            # Classifier-free guidance
            if guidance_scale > 1.0:
                # 有条件预测
                pred_noise_cond = model(img, time_cond, class_labels, domain_labels)
                
                # 无条件预测（null class，源域）
                null_class = torch.full_like(class_labels, model.class_embedding.num_embeddings - 1)
                null_domain = torch.zeros_like(domain_labels)
                pred_noise_uncond = model(img, time_cond, null_class, null_domain)
                
                # 混合
                pred_noise = pred_noise_uncond + guidance_scale * (pred_noise_cond - pred_noise_uncond)
            else:
                pred_noise = model(img, time_cond, class_labels, domain_labels)
            
            # DDIM更新步骤
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next] if time_next >= 0 else torch.tensor(1.0)
            
            # 预测x0
            if self.objective == 'pred_noise':
                pred_x0 = (img - torch.sqrt(1 - alpha) * pred_noise) / torch.sqrt(alpha)
            elif self.objective == 'pred_x0':
                pred_x0 = pred_noise
            elif self.objective == 'pred_v':
                pred_x0 = torch.sqrt(alpha) * img - torch.sqrt(1 - alpha) * pred_noise
            
            # 裁剪x0
            pred_x0 = torch.clamp(pred_x0, -1, 1)
            
            # 计算方差
            sigma = eta * torch.sqrt((1 - alpha_next) / (1 - alpha)) * torch.sqrt(1 - alpha / alpha_next)
            
            # 预测方向
            pred_dir = torch.sqrt(1 - alpha_next - sigma ** 2) * pred_noise
            
            # DDIM更新
            img = torch.sqrt(alpha_next) * pred_x0 + pred_dir
            
            if sigma > 0:
                noise = torch.randn_like(img)
                img = img + sigma * noise
            
            if return_all_timesteps:
                imgs.append(img)
        
        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)
        
        # 反归一化
        if self.auto_normalize:
            ret = self.unnormalize(ret)
        
        return ret


if __name__ == "__main__":
    # 测试
    from .conditional_unet import DomainConditionalUnet
    
    print("Testing Domain Adaptive Diffusion...")
    
    # 创建模型
    unet = DomainConditionalUnet(
        dim=64,
        dim_mults=(1, 2, 4, 4),
        channels=32
    )
    
    diffusion = DomainAdaptiveDiffusion(
        unet,
        image_size=16,
        timesteps=1000,
        sampling_timesteps=50,  # DDIM快速采样
        mmd_loss_weight=0.1
    )
    
    # 测试损失计算
    batch_size = 4
    x = torch.randn(batch_size, 32, 16, 16)
    t = torch.randint(0, 1000, (batch_size,))
    class_labels = torch.randint(0, 31, (batch_size,))
    domain_labels = torch.randint(0, 2, (batch_size,))
    
    # 测试不同训练阶段
    for phase in ['pretrain', 'align', 'finetune']:
        diffusion.set_training_phase(phase)
        loss = diffusion.p_losses(x, t, class_labels=class_labels, domain_labels=domain_labels)
        print(f"{phase} - Loss: {loss:.4f}")
        print(f"  Details: {diffusion.last_loss_dict}")
    
    print("✅ Diffusion model test passed!")

