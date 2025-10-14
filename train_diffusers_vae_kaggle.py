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
    加载已训练好的KL-VAE模型
    与prepare_microdoppler_data.py保持一致
    """
    try:
        # 加载KL-VAE
        import sys
        import os
        
        # 添加VAE模块路径
        vae_path_dir = '/kaggle/working/domain_adaptive_diffusion/vae'
        if vae_path_dir not in sys.path:
            sys.path.insert(0, vae_path_dir)
        
        # 检查kl_vae.py是否存在
        kl_vae_file = os.path.join(vae_path_dir, 'kl_vae.py')
        if not os.path.exists(kl_vae_file):
            raise ImportError(f"找不到kl_vae.py文件: {kl_vae_file}")
        
        from kl_vae import KL_VAE
        
        # 创建KL-VAE模型（与prepare_microdoppler_data.py一致）
        vae = KL_VAE()
        
        checkpoint = torch.load(vae_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            vae.load_state_dict(checkpoint['model_state_dict'])
        else:
            vae.load_state_dict(checkpoint)
        
        return vae, "kl_vae"
        
    except Exception as e:
        print(f"="*60)
        print(f"错误：加载KL-VAE失败！")
        print(f"原因：{e}")
        print(f"VAE路径：{vae_path}")
        print(f"="*60)
        print("请检查：")
        print("1. VAE checkpoint路径是否正确：/kaggle/input/kl-vae-best-pt/kl_vae_best.pt")
        print("2. kl_vae.py是否在：/kaggle/working/domain_adaptive_diffusion/vae/")
        print("3. VAE checkpoint格式是否包含'model_state_dict'键")
        print(f"="*60)
        import sys
        sys.exit(1)  # 直接退出程序


# ============ 2. VAE包装器（统一接口）============
class VAEWrapper:
    """统一不同VAE的接口"""
    
    def __init__(self, vae, vae_type="kl_vae"):
        self.vae = vae
        self.vae_type = vae_type
        # KL-VAE已经内置了scale_factor (0.18215)
        
    def encode(self, x):
        """编码到latent（用于实时编码，这里不需要因为用预编码数据）"""
        if self.vae_type == "kl_vae":
            # KL-VAE的encode_images方法已经应用了scale_factor
            with torch.no_grad():
                # x应该在[0,1]范围
                return self.vae.encode_images(x)
        else:
            # Diffusers VAE
            with torch.no_grad():
                latent = self.vae.encode(x).latent_dist.sample()
                return latent * 0.18215
    
    def decode(self, z):
        """解码到图像"""
        if self.vae_type == "kl_vae":
            # KL-VAE的decode_latents方法处理scale_factor
            with torch.no_grad():
                # decode_latents会自动处理scale_factor并clamp到[0,1]
                images = self.vae.decode_latents(z)
                # 转换到[-1,1]范围（用于可视化）
                return images * 2.0 - 1.0
        else:
            # Diffusers VAE
            with torch.no_grad():
                z = z / 0.18215
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
            
            # 检查数据是否为空
            if self.latents is None or len(self.latents) == 0:
                print(f"="*60)
                print("错误：latent数据为空！")
                print(f"数据文件：{data_path}")
                print(f"="*60)
                import sys
                sys.exit(1)
            
            print(f"加载 {len(self.latents)} 个latent样本, 形状: {self.latents[0].shape}")
            if self.labels is not None:
                num_classes = len(torch.unique(self.labels))
                print(f"条件类别数: {num_classes}个用户")
            
            # 加载数据统计信息（元数据）
            stats_path = os.path.join(latent_dir, "data_stats.pt")
            if os.path.exists(stats_path):
                self.stats = torch.load(stats_path, map_location='cpu')
                # latent已经在prepare_microdoppler_data.py中通过encode_images应用了scale_factor
            else:
                self.stats = None
        else:
            print(f"="*60)
            print(f"错误：找不到latent数据文件！")
            print(f"文件路径：{data_path}")
            print(f"="*60)
            print("请检查：")
            print("1. 数据集是否正确上传到Kaggle")
            print("2. 路径是否正确：/kaggle/input/data-latent2/")
            print("3. 文件名是否正确：source_train.pt")
            print(f"="*60)
            import sys
            sys.exit(1)  # 直接退出程序
    
    def __len__(self):
        return len(self.latents)
    
    def __getitem__(self, idx):
        latent = self.latents[idx]
        
        # 确保是3维张量 [C, H, W]，其中C=4 (KL-VAE的latent维度)
        if latent.dim() == 4 and latent.shape[0] == 1:
            latent = latent.squeeze(0)  # [1, 4, 32, 32] -> [4, 32, 32]
        elif latent.dim() == 2:
            # 不应该出现这种情况，但以防万一
            raise ValueError(f"意外的latent维度: {latent.shape}")
        
        # 获取标签（用户ID）用于条件扩散
        label = self.labels[idx] if self.labels is not None else 0
        
        # latent已经在prepare_microdoppler_data.py中应用了scale_factor (0.18215)
        return {
            "latents": latent,
            "labels": torch.tensor(label, dtype=torch.long)
        }


# ============ 5. Latent Diffusion训练 ============
def train_latent_diffusion(
    vae_path="/kaggle/input/kl-vae-best-pt/kl_vae_best.pt",  # 必需！用于解码
    latent_dir="/kaggle/input/data-latent2",  # 预编码的latent目录
    output_dir="/kaggle/working/latent_diffusion",
    num_epochs=50,
    batch_size=64,  # 增大batch size（还有10G显存）
    learning_rate=2e-4,  # 稍微增大学习率配合大batch
    weight_decay=0.01,  # 添加权重衰减正则化
):
    # 加速器
    accelerator = Accelerator(mixed_precision="fp16")
    
    # 加载VAE（必需，用于生成时解码）
    print("加载VAE...")
    vae_model, vae_type = load_pretrained_vae(vae_path)
    vae_model.cuda()
    vae_model.eval()  # VAE保持eval模式
    vae_wrapper = VAEWrapper(vae_model, vae_type)
    
    # 获取latent模型配置
    config = get_latent_model_config()
    
    # 创建条件UNet（支持用户ID作为条件）
    # Normal_line有31个用户（ID_01到ID_31）
    num_classes = 31
    
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
        # 添加条件编码
        class_embed_type="timestep",  # 将类别嵌入添加到时间步嵌入
        num_class_embeds=num_classes,  # 用户数
    )
    
    total_params = sum(p.numel() for p in unet.parameters()) / 1e6
    print(f"UNet参数: {total_params:.1f}M")
    
    # 数据集（使用预编码的latent）
    dataset = LatentDataset(
        latent_dir=latent_dir,
        split="train"  # 使用训练集
    )
    
    # 验证数据集
    if len(dataset) == 0:
        print("错误：数据集为空！")
        import sys
        sys.exit(1)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # latent已缓存，不需要多进程
        drop_last=True  # 保证batch一致性
    )
    print(f"训练集: {len(dataset)}个样本")
    
    # 加载验证集
    val_dataset = LatentDataset(latent_dir=latent_dir, split="val")
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    print(f"验证集: {len(val_dataset)}个样本")
    
    # 噪声调度器
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,  # SD标准值
        beta_end=0.012,
        beta_schedule="scaled_linear",
        prediction_type="epsilon",
    )
    
    # 优化器（使用传入的weight_decay参数）
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=weight_decay,  # 使用传入的参数
    )
    
    # 学习率调度
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * num_epochs,
    )
    
    # 准备训练（包括验证集）
    unet, optimizer, dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, val_dataloader, lr_scheduler
    )
    
    # 开始训练
    print(f"训练: {num_epochs}轮, 共{num_epochs * len(dataloader)}步")
    
    # 训练循环
    global_step = 0
    for epoch in range(num_epochs):
        # 使用leave=True保持每个epoch的进度条不消失
        progress_bar = tqdm(
            total=len(dataloader), 
            desc=f"Epoch {epoch+1}/{num_epochs}",
            leave=True,  # 保持进度条
            ncols=100,  # 固定宽度
            position=0  # 固定位置
        )
        
        epoch_losses = []
        for batch in dataloader:
            # 获取latent和标签
            latents = batch["latents"]
            labels = batch["labels"]  # 用户ID
            
            # 确保labels在正确的设备上
            labels = labels.to(latents.device)
            
            # 标准diffusion训练
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=latents.device
            ).long()
            
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # 条件预测噪声（传入用户ID作为类别条件）
            noise_pred = unet(
                noisy_latents, 
                timesteps,
                class_labels=labels  # 条件信息
            ).sample
            loss = F.mse_loss(noise_pred, noise)
            
            # 更新
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            # 记录损失
            epoch_losses.append(loss.item())
            
            # 更新进度条（简洁模式）
            progress_bar.update(1)
            progress_bar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'lr': f'{lr_scheduler.get_last_lr()[0]:.2e}'
            })
            
            global_step += 1
            
            # 定期生成样本
            if global_step % 500 == 0:
                generate_latent_samples(
                    unet, vae_wrapper, noise_scheduler, 
                    epoch, global_step
                )
        
        # 关闭进度条并输出epoch统计
        progress_bar.close()
        avg_train_loss = sum(epoch_losses) / len(epoch_losses)
        
        # 验证集评估
        val_losses = []
        unet.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                latents = batch["latents"]
                labels = batch["labels"]
                
                # 确保labels在正确的设备上
                labels = labels.to(latents.device)
                
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=latents.device
                ).long()
                
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                noise_pred = unet(noisy_latents, timesteps, class_labels=labels).sample
                val_loss = F.mse_loss(noise_pred, noise)
                val_losses.append(val_loss.item())
        
        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0
        unet.train()
        
        print(f"Epoch {epoch+1}/{num_epochs}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")
    
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
        "vae_path": vae_path,
        "latent_dir": latent_dir,
    }
    torch.save(pipeline_config, os.path.join(output_dir, "pipeline.pt"))
    
    print(f"模型已保存到 {output_dir}")
    
    return unet, vae_wrapper


# ============ 6. 生成样本 ============
def generate_latent_samples(unet, vae_wrapper, scheduler, epoch, step):
    """条件生成样本（选择4个不同用户）"""
    unet.eval()
    
    with torch.no_grad():
        # 初始化随机噪声（真实采样，非模拟数据）
        batch_size = 4
        latents = torch.randn((batch_size, 4, 32, 32), device=unet.device)
        
        # 选择不同的用户ID进行条件生成
        # 选择用户 0, 7, 14, 21（分散选择）
        class_labels = torch.tensor([0, 7, 14, 21], device=unet.device, dtype=torch.long)
        
        # DDIM采样（50步）
        scheduler.set_timesteps(50)  
        
        for t in scheduler.timesteps:
            # 条件生成
            noise_pred = unet(latents, t, class_labels=class_labels).sample
            latents = scheduler.step(noise_pred, t, latents).prev_sample
        
        # 解码到图像空间
        images = vae_wrapper.decode(latents)
        images = (images + 1) / 2  # [-1,1] -> [0,1]
        images = images.clamp(0, 1).cpu()
    
    unet.train()
    
    # 保存结果
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for i, ax in enumerate(axes):
        ax.imshow(images[i].permute(1, 2, 0).numpy())
        ax.axis('off')
    plt.suptitle(f'Step {step} (Epoch {epoch+1})')
    plt.savefig(f'/kaggle/working/samples_step_{step}.png', dpi=100, bbox_inches='tight')
    plt.close()


# ============ 7. 推理pipeline ============
class LatentDiffusionPipeline:
    """简单的推理pipeline"""
    
    def __init__(self, unet, vae_wrapper, scheduler):
        self.unet = unet
        self.vae_wrapper = vae_wrapper
        self.scheduler = scheduler
    
    @torch.no_grad()
    def __call__(self, batch_size=4, num_inference_steps=50, ddim=True, class_labels=None):
        # 使用DDIM加速
        if ddim:
            self.scheduler = DDIMScheduler.from_config(self.scheduler.config)
        
        self.scheduler.set_timesteps(num_inference_steps)
        
        # 初始化噪声
        latents = torch.randn(
            (batch_size, 4, 32, 32),
            device=self.unet.device
        )
        
        # 设置条件标签（如果没有指定，随机选择用户）
        if class_labels is None:
            # 随机选择不同用户
            class_labels = torch.randint(0, 31, (batch_size,), device=self.unet.device)
        elif not isinstance(class_labels, torch.Tensor):
            class_labels = torch.tensor(class_labels, device=self.unet.device, dtype=torch.long)
        
        # 条件去噪循环
        for t in tqdm(self.scheduler.timesteps, desc="生成中"):
            noise_pred = self.unet(latents, t, class_labels=class_labels).sample
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
    print("开始Latent Diffusion训练...")
    
    unet, vae_wrapper = train_latent_diffusion(
        vae_path="/kaggle/input/kl-vae-best-pt/kl_vae_best.pt",  # 必需！用于解码
        latent_dir="/kaggle/input/data-latent2",  # 预编码的latent
        output_dir="/kaggle/working/latent_diffusion",
        num_epochs=50,  # 训练更多轮次
        batch_size=64,  # 增大batch size（有10G空余显存）
        learning_rate=2e-4,  # 稍微提高学习率配合大batch
        weight_decay=0.01,  # L2正则化
    )
    
    # 测试生成
    print("生成测试样本...")
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
    )
    
    pipeline = LatentDiffusionPipeline(unet, vae_wrapper, scheduler)
    
    # 生成8个样本，选择8个不同的用户
    selected_users = [0, 4, 8, 12, 16, 20, 24, 28]  # 均匀分布的用户ID
    images = pipeline(
        batch_size=8, 
        num_inference_steps=50,  # DDIM 50步
        class_labels=selected_users  # 指定用户
    )
    
    # 保存结果
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, (ax, img) in enumerate(zip(axes.flat, images)):
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f'User {selected_users[i]}')
    plt.suptitle('Conditional Generation (50-step DDIM)')
    plt.savefig('/kaggle/working/final_results.png')
    plt.close()
    
    print("完成！结果保存在 /kaggle/working/")
