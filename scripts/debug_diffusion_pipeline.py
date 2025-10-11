"""
系统调试扩散模型pipeline的各个环节
"""
import torch
import torch.nn.functional as F
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from vae.kl_vae import KL_VAE
from utils.data_loader import create_dataloaders
from diffusers import UNet2DConditionModel, DDPMScheduler, DDIMScheduler
import yaml

class DiffusionDebugger:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
    def debug_vae_details(self, vae_checkpoint):
        """深入调试VAE的细节"""
        print("\n" + "="*50)
        print("1. VAE调试")
        print("="*50)
        
        # 加载VAE
        vae = KL_VAE(embed_dim=4, scale_factor=0.18215)
        checkpoint = torch.load(vae_checkpoint, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            vae.load_state_dict(checkpoint['model_state_dict'])
        else:
            vae.load_state_dict(checkpoint)
        vae.to(self.device)
        vae.eval()
        
        print(f"VAE scale_factor: {vae.scale_factor}")
        
        # 测试1：检查VAE的编码解码一致性
        print("\n测试1: VAE编码解码一致性")
        test_img = torch.rand(1, 3, 256, 256).to(self.device)
        with torch.no_grad():
            # 编码
            latent = vae.encode_images(test_img)  # 已经包含scale_factor
            print(f"  输入图像: shape={test_img.shape}, range=[{test_img.min():.3f}, {test_img.max():.3f}]")
            print(f"  编码latent: shape={latent.shape}, mean={latent.mean():.3f}, std={latent.std():.3f}")
            print(f"  编码latent range: [{latent.min():.3f}, {latent.max():.3f}]")
            
            # 解码
            recon = vae.decode_latents(latent)
            print(f"  重建图像: shape={recon.shape}, range=[{recon.min():.3f}, {recon.max():.3f}]")
            
        # 测试2：检查不同scale下的表现
        print("\n测试2: 不同尺度的latent解码")
        for scale in [0.1, 0.5, 1.0, 5.0, 10.0]:
            test_latent = torch.randn(1, 4, 32, 32).to(self.device) * scale
            with torch.no_grad():
                decoded = vae.decode_latents(test_latent)
                print(f"  输入std={scale:.1f}: 解码range=[{decoded.min():.2f}, {decoded.max():.2f}]")
                
        return vae
        
    def debug_data_pipeline(self, data_path, vae):
        """调试数据加载和预处理流程"""
        print("\n" + "="*50)
        print("2. 数据流程调试")
        print("="*50)
        
        # 加载数据
        train_loader, _ = create_dataloaders(
            data_path=Path(data_path),
            batch_size=8,
            phase='pretrain',
            num_workers=0
        )
        
        # 检查一个batch
        batch = next(iter(train_loader))
        latents = batch['latent'].to(self.device)
        labels = batch['class_label'].to(self.device)
        
        print(f"\n原始数据:")
        print(f"  Latents shape: {latents.shape}")
        print(f"  Latents stats: mean={latents.mean():.4f}, std={latents.std():.4f}")
        print(f"  Latents range: [{latents.min():.4f}, {latents.max():.4f}]")
        print(f"  Labels: {labels}")
        
        # 测试：这些latents能正确解码吗？
        print(f"\n解码测试:")
        with torch.no_grad():
            decoded = vae.decode_latents(latents[:1])
            print(f"  解码图像range: [{decoded.min():.3f}, {decoded.max():.3f}]")
            
        # 测试除以scale_factor后的情况
        print(f"\n归一化测试 (除以scale_factor={vae.scale_factor}):")
        normalized_latents = latents / vae.scale_factor
        print(f"  归一化后stats: mean={normalized_latents.mean():.4f}, std={normalized_latents.std():.4f}")
        print(f"  归一化后range: [{normalized_latents.min():.4f}, {normalized_latents.max():.4f}]")
        
        return latents, labels
        
    def debug_noise_schedule(self):
        """调试噪声调度器"""
        print("\n" + "="*50)
        print("3. 噪声调度器调试")
        print("="*50)
        
        # 创建调度器
        ddpm_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            variance_type="fixed_small",
            clip_sample=False,
            prediction_type="epsilon",
        )
        
        ddim_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
            prediction_type="epsilon",
        )
        
        # 测试加噪过程
        print("\nDDPM加噪测试:")
        x0 = torch.randn(1, 4, 32, 32)  # 标准范围
        noise = torch.randn_like(x0)
        
        for t in [0, 250, 500, 750, 999]:
            timesteps = torch.tensor([t])
            noisy = ddpm_scheduler.add_noise(x0, noise, timesteps)
            print(f"  t={t}: noisy std={noisy.std():.3f}, range=[{noisy.min():.2f}, {noisy.max():.2f}]")
            
        # 测试DDIM采样
        print("\nDDIM采样测试:")
        ddim_scheduler.set_timesteps(50)
        x_t = torch.randn(1, 4, 32, 32) * 1.0  # 从标准高斯开始
        
        print(f"  初始: std={x_t.std():.3f}")
        for i, t in enumerate(ddim_scheduler.timesteps[:5]):  # 只看前5步
            noise_pred = torch.randn_like(x_t) * 0.1  # 模拟小的噪声预测
            x_t = ddim_scheduler.step(noise_pred, t, x_t, return_dict=False)[0]
            print(f"  Step {i}, t={t}: std={x_t.std():.3f}, range=[{x_t.min():.2f}, {x_t.max():.2f}]")
            
    def debug_unet_condition(self, config_path):
        """调试UNet的条件机制"""
        print("\n" + "="*50)
        print("4. UNet条件机制调试")
        print("="*50)
        
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        model_config = config.get('model', {})
        
        # 创建模型 - 使用与train_diffusers.py相同的配置
        unet = UNet2DConditionModel(
            sample_size=config['data']['latent_size'],  # 32
            in_channels=model_config.get('in_channels', 4),
            out_channels=model_config.get('out_channels', 4),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D"
            ),
            up_block_types=(
                "UpBlock2D",
                "UpBlock2D", 
                "UpBlock2D",
                "UpBlock2D"
            ),
            block_out_channels=tuple(model_config.get('block_out_channels', [128, 256, 512, 512])),
            layers_per_block=model_config.get('layers_per_block', 3),
            attention_head_dim=model_config.get('attention_head_dim', 16),
            num_class_embeds=model_config.get('num_class_embeds', 31),
            class_embed_type=model_config.get('class_embed_type', 'timestep'),
            class_embeddings_concat=model_config.get('class_embeddings_concat', False),
            # 重要：对于timestep类型的条件编码，不需要设置cross_attention_dim
            norm_num_groups=model_config.get('norm_num_groups', 32),
            norm_eps=float(model_config.get('norm_eps', 1e-6)),
            resnet_time_scale_shift="default",
            act_fn=model_config.get('act_fn', 'silu'),
        ).to(self.device)
        
        # 测试不同条件下的输出
        print("\n测试条件影响:")
        x = torch.randn(2, 4, 32, 32).to(self.device)
        t = torch.tensor([500, 500]).to(self.device)
        
        # 测试1：相同输入，不同类别
        labels1 = torch.tensor([0, 15]).to(self.device)  # 不同类别
        # 创建dummy的encoder_hidden_states（使用标准SD的1280维）
        dummy_encoder_hidden_states = torch.zeros(2, 1, 1280).to(self.device)
        
        with torch.no_grad():
            out1 = unet(x, t, class_labels=labels1, 
                       encoder_hidden_states=dummy_encoder_hidden_states,
                       return_dict=False)[0]
            
        print(f"  不同类别输出差异: {(out1[0] - out1[1]).abs().mean():.4f}")
        
        # 测试2：相同输入，相同类别
        labels2 = torch.tensor([0, 0]).to(self.device)  # 相同类别
        
        with torch.no_grad():
            out2 = unet(x, t, class_labels=labels2, 
                       encoder_hidden_states=dummy_encoder_hidden_states,
                       return_dict=False)[0]
            
        print(f"  相同类别输出差异: {(out2[0] - out2[1]).abs().mean():.4f}")
        
        # 测试3：有条件vs无条件（使用null class）
        with torch.no_grad():
            out_with_cond = unet(x[:1], t[:1], class_labels=labels1[:1], 
                               encoder_hidden_states=dummy_encoder_hidden_states[:1],
                               return_dict=False)[0]
            # 使用null class（通常是num_classes）
            null_label = torch.tensor([31]).to(self.device)  # 31是null class
            out_no_cond = unet(x[:1], t[:1], class_labels=null_label, 
                             encoder_hidden_states=dummy_encoder_hidden_states[:1],
                             return_dict=False)[0]
            
        print(f"  有/无条件输出差异: {(out_with_cond - out_no_cond).abs().mean():.4f}")
        
    def debug_training_prediction(self, model_checkpoint=None):
        """调试训练后的模型预测"""
        print("\n" + "="*50)
        print("5. 模型预测调试")
        print("="*50)
        
        if model_checkpoint and Path(model_checkpoint).exists():
            checkpoint = torch.load(model_checkpoint, map_location=self.device)
            print(f"加载checkpoint: epoch={checkpoint.get('epoch', 'unknown')}")
            
            # 这里可以加载模型并测试
            # 由于需要完整的模型初始化代码，暂时跳过
            print("TODO: 加载并测试训练后的模型")
        else:
            print("未提供模型checkpoint")

def main():
    debugger = DiffusionDebugger()
    
    # 配置路径（需要根据实际情况调整）
    vae_checkpoint = "/kaggle/input/kl-vae-best-pt/kl_vae_best.pt"
    data_path = "/kaggle/input/data-latent"
    config_path = "/kaggle/working/domain_adaptive_diffusion/configs/diffusers_baseline.yaml"
    
    # 1. 调试VAE
    vae = debugger.debug_vae_details(vae_checkpoint)
    
    # 2. 调试数据流程
    latents, labels = debugger.debug_data_pipeline(data_path, vae)
    
    # 3. 调试噪声调度
    debugger.debug_noise_schedule()
    
    # 4. 调试UNet条件
    debugger.debug_unet_condition(config_path)
    
    # 5. 调试模型预测（如果有checkpoint）
    # debugger.debug_training_prediction("path/to/checkpoint.pt")

if __name__ == "__main__":
    main()
