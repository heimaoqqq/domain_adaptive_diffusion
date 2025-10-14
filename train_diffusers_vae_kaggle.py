"""
Diffusers + VAE训练脚本 - Latent Diffusion版本
使用您已训练好的VAE，在latent空间训练diffusion
"""

import os
import torch
import torch.nn.functional as F
from diffusers import (
    AutoencoderKL,
    UNet2DModel, 
    DDPMScheduler,
    DDIMScheduler,
    StableDiffusionPipeline
)
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


# ============ 1. 加载您的VAE ============
def load_pretrained_vae(vae_path="/kaggle/input/kl-vae-best-pt/kl_vae_best.pt"):
    """
    加载您已训练好的VAE
    如果格式不兼容，可以用diffusers的VAE
    """
    try:
        # 尝试加载您的VAE
        print(f"加载VAE: {vae_path}")
        
        # 如果是您自己训练的simplified_vavae
        import sys
        sys.path.append('/kaggle/working')
        from simplified_vavae import VAVAE
        
        vae = VAVAE(
            in_channels=3,
            latent_dim=4,
            hidden_dims=[128, 256, 512, 512],
        )
        
        checkpoint = torch.load(vae_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            vae.load_state_dict(checkpoint['model_state_dict'])
        else:
            vae.load_state_dict(checkpoint)
        
        print("成功加载自定义VAE")
        return vae, "custom"
        
    except:
        # 备选：使用diffusers的预训练VAE
        print("使用Diffusers的VAE（与SD兼容）")
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",  # SD的VAE
            torch_dtype=torch.float16
        )
        return vae, "diffusers"


# ============ 2. VAE包装器（统一接口）============
class VAEWrapper:
    """统一不同VAE的接口"""
    
    def __init__(self, vae, vae_type="custom"):
        self.vae = vae
        self.vae_type = vae_type
        self.scaling_factor = 0.18215  # SD标准缩放
        
    def encode(self, x):
        """编码到latent"""
        if self.vae_type == "custom":
            # 您的VAE
            with torch.no_grad():
                mu, logvar = self.vae.encode(x)
                z = self.vae.reparameterize(mu, logvar)
                return z * self.scaling_factor
        else:
            # Diffusers VAE
            with torch.no_grad():
                latent = self.vae.encode(x).latent_dist.sample()
                return latent * self.scaling_factor
    
    def decode(self, z):
        """解码到图像"""
        z = z / self.scaling_factor
        
        if self.vae_type == "custom":
            with torch.no_grad():
                return self.vae.decode(z)
        else:
            with torch.no_grad():
                return self.vae.decode(z).sample


# ============ 3. Latent Diffusion模型配置 ============
def get_latent_model_config():
    """
    Latent空间的UNet配置
    注意：输入是32x32x4，不是256x256x3
    """
    
    configs = {
        "nano": {  # ~5M参数
            "layers_per_block": 1,
            "block_out_channels": (128, 256, 256),
            "attention_resolutions": [],
        },
        "tiny": {  # ~20M参数（推荐）
            "layers_per_block": 2,
            "block_out_channels": (128, 256, 512, 512),
            "attention_resolutions": [4],  # 在8x8分辨率
        },
        "small": {  # ~50M参数
            "layers_per_block": 2,
            "block_out_channels": (224, 448, 672, 896),
            "attention_resolutions": [4, 2],
        },
        "base": {  # ~860M参数（标准SD）
            "layers_per_block": 2,
            "block_out_channels": (320, 640, 1280, 1280),
            "attention_resolutions": [4, 2, 1],
        }
    }
    
    # 对latent训练，推荐tiny
    selected = "tiny"  # ← 修改这里选择大小
    config = configs[selected]
    
    print(f"使用 {selected} Latent Diffusion配置:")
    print(f"  - Latent尺寸: 32x32x4")
    print(f"  - 通道数: {config['block_out_channels']}")
    print(f"  - 注意力: {config['attention_resolutions']}")
    
    return config


# ============ 4. 数据集（直接加载预编码的latent）============
class LatentDataset(Dataset):
    """直接加载预编码的latent数据"""
    
    def __init__(self, latent_dir="/kaggle/input/data-latent2", split="train"):
        """
        latent_dir: 包含预编码latent的目录
        split: "train" 或 "val"
        """
        self.latent_dir = latent_dir
        self.split = split
        
        # 加载对应的数据文件
        if split == "train":
            data_path = os.path.join(latent_dir, "source_train.pt")
        elif split == "val":
            data_path = os.path.join(latent_dir, "source_val.pt")
        else:
            data_path = os.path.join(latent_dir, "target_fewshot.pt")
        
        print(f"加载预编码latent: {data_path}")
        
        # 加载数据
        if os.path.exists(data_path):
            data = torch.load(data_path, map_location='cpu')
            
            # 检查数据格式
            if isinstance(data, dict):
                self.latents = data.get('latents', data.get('data', None))
                self.labels = data.get('labels', None)
            elif isinstance(data, torch.Tensor):
                self.latents = data
                self.labels = None
            else:
                raise ValueError(f"未知数据格式: {type(data)}")
            
            print(f"加载 {len(self.latents)} 个latent样本")
            print(f"Latent形状: {self.latents[0].shape if len(self.latents) > 0 else 'N/A'}")
            
            # 加载统计信息（用于归一化）
            stats_path = os.path.join(latent_dir, "data_stats.pt")
            if os.path.exists(stats_path):
                self.stats = torch.load(stats_path, map_location='cpu')
                print(f"加载统计信息: mean={self.stats.get('mean', 'N/A')}, std={self.stats.get('std', 'N/A')}")
            else:
                self.stats = None
        else:
            raise FileNotFoundError(f"找不到数据文件: {data_path}")
    
    def __len__(self):
        return len(self.latents)
    
    def __getitem__(self, idx):
        latent = self.latents[idx]
        
        # 确保是4维张量 [C, H, W]
        if latent.dim() == 3:
            latent = latent.unsqueeze(0)  # 添加batch维度
        
        # 如果是5维 [1, C, H, W]，去掉第一维
        if latent.dim() == 4 and latent.shape[0] == 1:
            latent = latent.squeeze(0)
        
        return {"latents": latent}


# ============ 5. Latent Diffusion训练 ============
def train_latent_diffusion(
    latent_dir="/kaggle/input/data-latent2",  # 预编码的latent目录
    output_dir="/kaggle/working/latent_diffusion",
    num_epochs=50,
    batch_size=32,  # latent空间可以更大batch
    learning_rate=1e-4,
    vae_path=None,  # 可选，用于生成时解码
):
    # 加速器
    accelerator = Accelerator(mixed_precision="fp16")
    
    # 如果提供了VAE路径，加载用于生成时解码（可选）
    vae_wrapper = None
    if vae_path and os.path.exists(vae_path):
        vae_model, vae_type = load_pretrained_vae(vae_path)
        vae_model.cuda()
        vae_model.eval()
        vae_wrapper = VAEWrapper(vae_model, vae_type)
        print("加载VAE用于样本生成")
    
    # 获取latent模型配置
    config = get_latent_model_config()
    
    # 创建UNet（在latent空间）
    unet = UNet2DModel(
        sample_size=32,  # latent大小
        in_channels=4,   # VAE latent channels
        out_channels=4,
        layers_per_block=config["layers_per_block"],
        block_out_channels=config["block_out_channels"],
        down_block_types=tuple(
            ["DownBlock2D" if i not in config["attention_resolutions"] 
             else "AttnDownBlock2D" 
             for i in range(len(config["block_out_channels"]))]
        ),
        up_block_types=tuple(
            ["UpBlock2D" if i not in config["attention_resolutions"]
             else "AttnUpBlock2D"
             for i in range(len(config["block_out_channels"]))]
        ),
        attention_head_dim=8,
        norm_num_groups=32,
        act_fn="silu",
    )
    
    total_params = sum(p.numel() for p in unet.parameters()) / 1e6
    print(f"UNet参数: {total_params:.1f}M")
    
    # 数据集（使用预编码的latent）
    dataset = LatentDataset(
        latent_dir=latent_dir,
        split="train"  # 使用训练集
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # latent已缓存，不需要多进程
    )
    
    # 噪声调度器
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,  # SD标准值
        beta_end=0.012,
        beta_schedule="scaled_linear",
        prediction_type="epsilon",
    )
    
    # 优化器
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )
    
    # 学习率调度
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * num_epochs,
    )
    
    # 准备训练
    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler
    )
    
    # 训练循环
    global_step = 0
    for epoch in range(num_epochs):
        progress_bar = tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in dataloader:
            # 直接获取预编码的latent
            latents = batch["latents"]
            
            # 标准diffusion训练
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=latents.device
            ).long()
            
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # 预测噪声
            noise_pred = unet(noisy_latents, timesteps).sample
            loss = F.mse_loss(noise_pred, noise)
            
            # 更新
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item(), lr=lr_scheduler.get_last_lr()[0])
            
            global_step += 1
            
            # 定期生成样本（如果有VAE）
            if global_step % 500 == 0 and vae_wrapper is not None:
                generate_latent_samples(
                    unet, vae_wrapper, noise_scheduler, 
                    epoch, global_step
                )
    
    # 保存模型
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存UNet
    torch.save(
        accelerator.unwrap_model(unet).state_dict(),
        os.path.join(output_dir, "unet.pt")
    )
    
    # 保存完整配置（用于推理）
    pipeline_config = {
        "unet": accelerator.unwrap_model(unet).state_dict(),
        "scheduler": noise_scheduler,
        "latent_dir": latent_dir,
    }
    if vae_path:
        pipeline_config["vae_path"] = vae_path
    torch.save(pipeline_config, os.path.join(output_dir, "pipeline.pt"))
    
    print(f"模型已保存到 {output_dir}")
    
    return unet, vae_wrapper


# ============ 6. 生成样本 ============
def generate_latent_samples(unet, vae_wrapper, scheduler, epoch, step):
    """从latent生成样本"""
    unet.eval()
    
    with torch.no_grad():
        # 初始化latent噪声
        latents = torch.randn((4, 4, 32, 32), device=unet.device)
        
        # DDIM采样（更快）
        scheduler.set_timesteps(50)  # 只用50步
        
        for t in scheduler.timesteps:
            noise_pred = unet(latents, t).sample
            latents = scheduler.step(noise_pred, t, latents).prev_sample
        
        # 解码到图像
        images = vae_wrapper.decode(latents)
        images = (images + 1) / 2
        images = images.clamp(0, 1).cpu()
    
    unet.train()
    
    # 保存
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for i, ax in enumerate(axes):
        ax.imshow(images[i].permute(1, 2, 0).numpy())
        ax.axis('off')
    plt.suptitle(f'Latent Diffusion - Epoch {epoch}, Step {step}')
    plt.savefig(f'/kaggle/working/samples_step_{step}.png')
    plt.close()
    print(f"样本保存: samples_step_{step}.png")


# ============ 7. 推理pipeline ============
class LatentDiffusionPipeline:
    """简单的推理pipeline"""
    
    def __init__(self, unet, vae_wrapper, scheduler):
        self.unet = unet
        self.vae_wrapper = vae_wrapper
        self.scheduler = scheduler
    
    @torch.no_grad()
    def __call__(self, batch_size=4, num_inference_steps=50, ddim=True):
        # 使用DDIM加速
        if ddim:
            self.scheduler = DDIMScheduler.from_config(self.scheduler.config)
        
        self.scheduler.set_timesteps(num_inference_steps)
        
        # 初始化噪声
        latents = torch.randn(
            (batch_size, 4, 32, 32),
            device=self.unet.device
        )
        
        # 去噪循环
        for t in tqdm(self.scheduler.timesteps, desc="生成中"):
            noise_pred = self.unet(latents, t).sample
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        # 解码
        images = self.vae_wrapper.decode(latents)
        images = (images + 1) / 2
        images = images.clamp(0, 1)
        
        # 转为PIL
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        images = [Image.fromarray((img * 255).astype(np.uint8)) for img in images]
        
        return images


# ============ 8. 主函数 ============
if __name__ == "__main__":
    # 训练
    print("="*60)
    print("Latent Diffusion训练（使用预编码数据）")
    print("="*60)
    
    unet, vae_wrapper = train_latent_diffusion(
        latent_dir="/kaggle/input/data-latent2",  # 预编码的latent
        output_dir="/kaggle/working/latent_diffusion",
        num_epochs=30,
        batch_size=32,  # latent空间可以更大batch
        learning_rate=1e-4,
        vae_path="/kaggle/input/kl-vae-best-pt/kl_vae_best.pt",  # 可选，用于生成
    )
    
    # 测试生成（如果有VAE）
    if vae_wrapper is not None:
        print("\n测试生成...")
        scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
        )
        
        pipeline = LatentDiffusionPipeline(unet, vae_wrapper, scheduler)
        images = pipeline(batch_size=8, num_inference_steps=20)  # DDIM 20步
        
        # 显示结果
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        for i, (ax, img) in enumerate(zip(axes.flat, images)):
            ax.imshow(img)
            ax.axis('off')
        plt.suptitle('Latent Diffusion最终结果（20步DDIM）')
        plt.savefig('/kaggle/working/final_results.png')
    else:
        print("\n跳过测试生成（无VAE解码器）")
    
    print("完成！")
