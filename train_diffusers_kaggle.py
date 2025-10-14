"""
Diffusers训练脚本 - Kaggle优化版
只修改模型配置，其他保持标准
"""

import os
import torch
import torch.nn.functional as F
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


# ============ 1. 模型配置（只改这里！）============
def get_model_config():
    """
    根据数据集大小选择配置
    4.6k图像 → 小模型
    """
    
    # 配置选项（从小到大）
    configs = {
        "nano": {  # ~10M参数，超快
            "layers_per_block": 1,
            "block_out_channels": (64, 128, 256),
            "attention_resolutions": [],  # 无注意力，纯CNN
        },
        "tiny": {  # ~30M参数，推荐
            "layers_per_block": 2,
            "block_out_channels": (64, 128, 256, 256),
            "attention_resolutions": [8],  # 只在最低分辨率
        },
        "small": {  # ~100M参数，质量好
            "layers_per_block": 2,
            "block_out_channels": (128, 256, 512, 512),
            "attention_resolutions": [8, 16],
        },
        "base": {  # ~400M参数，标准SD规模
            "layers_per_block": 2,
            "block_out_channels": (224, 448, 672, 896),
            "attention_resolutions": [8, 16, 32],
        }
    }
    
    # 选择配置（对4.6k数据，推荐tiny或small）
    selected = "tiny"  # ← 修改这里选择大小
    config = configs[selected]
    
    print(f"使用 {selected} 配置:")
    print(f"  - 通道数: {config['block_out_channels']}")
    print(f"  - 注意力层: {config['attention_resolutions']}")
    print(f"  - 估计参数量: ~{[10, 30, 100, 400][list(configs.keys()).index(selected)]}M")
    
    return config


# ============ 2. 数据集（标准实现）============
class MicroDopplerDataset(Dataset):
    """简单数据集：直接读取256x256图像"""
    
    def __init__(self, root_dir="/kaggle/input/your-dataset", split="train"):
        self.images = []
        
        # Kaggle数据路径（根据实际调整）
        if not os.path.exists(root_dir):
            # 如果输入数据不存在，尝试工作目录
            root_dir = "dataset/organized_gait_dataset"
        
        # 收集所有图像
        for subdir in os.listdir(root_dir):
            subdir_path = os.path.join(root_dir, subdir)
            if os.path.isdir(subdir_path):
                for img_file in os.listdir(subdir_path):
                    if img_file.endswith(('.jpg', '.png')):
                        self.images.append(os.path.join(subdir_path, img_file))
        
        print(f"找到 {len(self.images)} 张图像")
        
        # 简单的训练/验证分割
        if split == "train":
            self.images = self.images[:int(0.9 * len(self.images))]
        else:
            self.images = self.images[int(0.9 * len(self.images)):]
        
        # 标准变换
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # [-1, 1]
        ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        return {"images": self.transform(image)}


# ============ 3. 训练函数（标准流程）============
def train_model(
    output_dir="./ddpm_microdoppler",
    num_epochs=50,  # Kaggle限制：调整epoch数
    batch_size=16,   # 根据GPU调整
    learning_rate=1e-4,
    save_every_n_epochs=10,
    generate_samples_every=500  # 每N步生成样本
):
    # 初始化加速器
    accelerator = Accelerator(
        mixed_precision="fp16",  # Kaggle GPU支持FP16
        gradient_accumulation_steps=1,
    )
    
    # 获取模型配置
    config = get_model_config()
    
    # 创建模型（核心修改在这里的参数）
    model = UNet2DModel(
        sample_size=256,
        in_channels=3,
        out_channels=3,
        layers_per_block=config["layers_per_block"],
        block_out_channels=config["block_out_channels"],
        down_block_types=tuple(
            ["DownBlock2D"] * len(config["block_out_channels"])
        ),
        up_block_types=tuple(
            ["UpBlock2D"] * len(config["block_out_channels"])
        ),
        # 注意力配置
        attention_head_dim=8 if config["attention_resolutions"] else None,
        norm_num_groups=32,
        norm_eps=1e-6,
        act_fn="silu",
    )
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"模型总参数: {total_params:.1f}M")
    
    # 数据集
    train_dataset = MicroDopplerDataset(split="train")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2  # Kaggle CPU workers
    )
    
    # 噪声调度器（标准DDPM）
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        prediction_type="epsilon",  # 或"v_prediction"
    )
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.95, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )
    
    # 学习率调度
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(train_dataloader) * num_epochs,
    )
    
    # 准备加速
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    # 训练循环
    global_step = 0
    for epoch in range(num_epochs):
        progress_bar = tqdm(
            total=len(train_dataloader),
            disable=not accelerator.is_local_main_process,
            desc=f"Epoch {epoch+1}/{num_epochs}"
        )
        
        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"]
            
            # 采样噪声
            noise = torch.randn_like(clean_images)
            bs = clean_images.shape[0]
            
            # 随机时间步
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (bs,), device=clean_images.device
            ).long()
            
            # 添加噪声
            noisy_images = noise_scheduler.add_noise(
                clean_images, noise, timesteps
            )
            
            # 预测噪声
            with accelerator.accumulate(model):
                model_output = model(noisy_images, timesteps).sample
                
                # MSE损失
                loss = F.mse_loss(model_output, noise)
                
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            progress_bar.update(1)
            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step
            }
            progress_bar.set_postfix(**logs)
            
            global_step += 1
            
            # 定期生成样本
            if global_step % generate_samples_every == 0:
                generate_samples(model, noise_scheduler, epoch, global_step)
        
        # 保存checkpoint
        if (epoch + 1) % save_every_n_epochs == 0:
            save_checkpoint(accelerator, model, optimizer, epoch, output_dir)
    
    print("训练完成！")
    
    # 创建pipeline用于推理
    pipeline = DDPMPipeline(
        unet=accelerator.unwrap_model(model),
        scheduler=noise_scheduler,
    )
    pipeline.save_pretrained(output_dir)
    
    return pipeline


# ============ 4. 生成样本（监控质量）============
def generate_samples(model, scheduler, epoch, step):
    """生成样本查看训练进度"""
    model.eval()
    
    with torch.no_grad():
        # 初始化噪声
        images = torch.randn((4, 3, 256, 256), device=model.device)
        
        # DDPM采样
        for t in scheduler.timesteps:
            model_output = model(images, t).sample
            images = scheduler.step(model_output, t, images).prev_sample
    
    model.train()
    
    # 保存图像
    images = (images + 1) / 2  # [-1, 1] -> [0, 1]
    images = images.clamp(0, 1).cpu()
    
    # 显示
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for i, ax in enumerate(axes):
        ax.imshow(images[i].permute(1, 2, 0).numpy())
        ax.axis('off')
    plt.suptitle(f'Epoch {epoch}, Step {step}')
    plt.savefig(f'samples_step_{step}.png')
    plt.close()
    
    print(f"样本已保存: samples_step_{step}.png")


# ============ 5. 保存模型============
def save_checkpoint(accelerator, model, optimizer, epoch, output_dir):
    """保存训练checkpoint"""
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "model": accelerator.unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    
    save_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save(checkpoint, save_path)
    print(f"Checkpoint保存到: {save_path}")


# ============ 6. 主函数============
if __name__ == "__main__":
    # Kaggle特定设置
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # 配置（根据需要调整）
    config = {
        "output_dir": "/kaggle/working/ddpm_output",  # Kaggle输出
        "num_epochs": 30,      # Kaggle有时间限制
        "batch_size": 16,      # 根据GPU内存调整
        "learning_rate": 1e-4,
        "save_every_n_epochs": 10,
        "generate_samples_every": 500,
    }
    
    # 开始训练
    print("="*50)
    print("Diffusers 微多普勒训练")
    print("="*50)
    
    pipeline = train_model(**config)
    
    # 测试生成
    print("\n测试生成...")
    test_images = pipeline(
        batch_size=4,
        num_inference_steps=50,  # DDIM可以改为20
    ).images
    
    # 保存测试结果
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for i, (ax, img) in enumerate(zip(axes, test_images)):
        ax.imshow(img)
        ax.axis('off')
    plt.suptitle('最终生成结果')
    plt.savefig('/kaggle/working/final_results.png')
    print("完成！结果保存在 /kaggle/working/")
