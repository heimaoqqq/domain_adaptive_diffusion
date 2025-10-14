"""
Diffusers + VAE训练脚本 - Latent Diffusion版本
使用您已训练好的VAE，在latent空间训练diffusion
"""

import os
import sys
import torch
import torch.nn.functional as F
from diffusers import (
    UNet2DModel,
    UNet2DConditionModel,  # SD官方架构
    DDPMScheduler,
    DDIMScheduler
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


# ============ 2.5. SD官方风格的类别条件编码器 ============
class ClassConditioner(torch.nn.Module):
    """
    将类别ID转为适用于Cross-Attention的序列嵌入
    设计理念：类似CLIP text encoder，但针对类别ID优化
    """
    def __init__(self, num_classes=32, embed_dim=512, seq_length=8, dropout=0.1):
        """
        num_classes: 类别数（31用户+1无条件）
        embed_dim: 嵌入维度（与UNet的cross_attention_dim匹配）
        seq_length: 序列长度（更长的序列=更丰富的条件信息）
        dropout: Dropout比率（正则化）
        """
        super().__init__()
        self.num_classes = num_classes
        self.seq_length = seq_length
        self.embed_dim = embed_dim
        
        # 类别嵌入表
        self.class_embedding = torch.nn.Embedding(num_classes, embed_dim)
        
        # 位置编码（让序列的不同位置学习不同特征）
        self.position_embedding = torch.nn.Parameter(
            torch.randn(1, seq_length, embed_dim) * 0.02
        )
        
        # Transformer层：增强特征表达能力（SD官方做法）
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,  # 8个注意力头
            dim_feedforward=embed_dim * 4,  # FFN维度
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN（更稳定）
        )
        self.transformer = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers=2  # 2层transformer（平衡表达能力和过拟合）
        )
        
        # Dropout（正则化）
        self.dropout = torch.nn.Dropout(dropout)
        
        # Layer Norm（稳定训练）
        self.final_ln = torch.nn.LayerNorm(embed_dim)
        
    def forward(self, class_labels):
        """
        class_labels: [B] 类别ID张量
        返回: [B, seq_length, embed_dim] 用于cross-attention
        """
        batch_size = class_labels.shape[0]
        
        # 1. 类别嵌入 [B, 1, embed_dim]
        class_emb = self.class_embedding(class_labels).unsqueeze(1)
        
        # 2. 复制成序列 [B, seq_length, embed_dim]
        seq_emb = class_emb.expand(-1, self.seq_length, -1)
        
        # 3. 添加位置编码
        seq_emb = seq_emb + self.position_embedding
        
        # 4. Dropout
        seq_emb = self.dropout(seq_emb)
        
        # 5. Transformer增强特征
        seq_emb = self.transformer(seq_emb)
        
        # 6. Final LayerNorm
        seq_emb = self.final_ln(seq_emb)
        
        return seq_emb


# ============ 3. Latent Diffusion模型配置 ============
def get_latent_model_config():
    """
    SD官方架构的UNet配置（针对小数据集优化）
    使用UNet2DConditionModel + Cross-Attention
    """
    
    configs = {
        "micro": {  # ~25M参数（推荐：小数据集）
            "layers_per_block": 2,
            "block_out_channels": (128, 256, 384, 384),
            "attention_head_dim": 8,  # 8头注意力（SD标准）
            "cross_attention_dim": 512,  # 条件嵌入维度
            "transformer_layers_per_block": 1,  # 每个block的transformer层数
            "dropout": 0.1,  # UNet内部dropout（正则化）
        },
        "small": {  # ~45M参数
            "layers_per_block": 2,
            "block_out_channels": (160, 320, 480, 640),
            "attention_head_dim": 8,
            "cross_attention_dim": 640,
            "transformer_layers_per_block": 1,
            "dropout": 0.1,
        },
        "base": {  # ~860M参数（标准SD 1.5配置）
            "layers_per_block": 2,
            "block_out_channels": (320, 640, 1280, 1280),
            "attention_head_dim": 8,
            "cross_attention_dim": 768,
            "transformer_layers_per_block": 1,
            "dropout": 0.0,  # SD官方不用dropout（数据集大）
        }
    }
    
    # 选择micro配置（适合4K样本的小数据集）
    selected = "micro"
    config = configs[selected]
    
    print(f"使用 {selected} SD架构配置（Cross-Attention）:")
    print(f"  - Latent尺寸: 32x32x4")
    print(f"  - 通道数: {config['block_out_channels']}")
    print(f"  - Cross-Attention维度: {config['cross_attention_dim']}")
    print(f"  - Dropout: {config['dropout']}")
    
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
        if self.labels is not None:
            label = self.labels[idx]
            # 如果label已经是张量，获取其值；否则直接使用
            if isinstance(label, torch.Tensor):
                label = label.item()  # 转为Python标量
        else:
            label = 0
        
        # latent已经在prepare_microdoppler_data.py中应用了scale_factor (0.18215)
        return {
            "latents": latent,
            "labels": torch.tensor(label, dtype=torch.long)  # 现在label确保是Python标量
        }


# ============ 5. Latent Diffusion训练 ============
def train_latent_diffusion(
    vae_path="/kaggle/input/kl-vae-best-pt/kl_vae_best.pt",  # 必需！用于解码
    latent_dir="/kaggle/input/data-latent2",  # 预编码的latent目录
    output_dir="/kaggle/working/latent_diffusion",
    num_epochs=200,
    batch_size=64,  # 增大batch size（还有10G显存）
    learning_rate=8e-5,  # micro配置：比nano稍高，比tiny稍低
    weight_decay=0.015,  # 适中的L2正则化（micro配置）
    sample_every_n_epochs=5,  # 每N个epoch生成可视化样本
    early_stopping_patience=12,  # 早停：12个epoch（micro配置需要稍长训练）
):
    # 加速器
    accelerator = Accelerator(mixed_precision="fp16")
    
    # 加载VAE（必需，用于生成时解码）
    print("加载VAE...")
    vae_model, vae_type = load_pretrained_vae(vae_path)
    vae_model.cuda()
    vae_model.eval()  # VAE保持eval模式
    vae_wrapper = VAEWrapper(vae_model, vae_type)
    
    # 获取SD官方配置
    config = get_latent_model_config()
    num_classes = 32  # 31用户 + 1无条件
    
    # 创建条件编码器（类似SD的CLIP encoder）
    class_conditioner = ClassConditioner(
        num_classes=num_classes,
        embed_dim=config["cross_attention_dim"],
        seq_length=8,  # 序列长度（平衡表达能力和计算量）
        dropout=0.1  # 条件编码器内部dropout（正则化）
    )
    
    # 创建UNet2DConditionModel（SD官方架构）
    unet = UNet2DConditionModel(
        sample_size=32,  # latent尺寸
        in_channels=4,   # VAE latent通道
        out_channels=4,
        layers_per_block=config["layers_per_block"],
        block_out_channels=config["block_out_channels"],
        # SD官方：所有block都使用CrossAttention
        down_block_types=tuple(
            "CrossAttnDownBlock2D" for _ in config["block_out_channels"]
        ),
        up_block_types=tuple(
            "CrossAttnUpBlock2D" for _ in config["block_out_channels"]
        ),
        # Cross-Attention配置
        attention_head_dim=config["attention_head_dim"],
        cross_attention_dim=config["cross_attention_dim"],
        transformer_layers_per_block=config["transformer_layers_per_block"],
        # 正则化
        dropout=config["dropout"],  # UNet内部dropout
        # SD官方标准
        norm_num_groups=32,
        act_fn="silu",
    )
    
    # 打印参数量
    unet_params = sum(p.numel() for p in unet.parameters()) / 1e6
    cond_params = sum(p.numel() for p in class_conditioner.parameters()) / 1e6
    print(f"UNet参数: {unet_params:.1f}M + 条件编码器: {cond_params:.1f}M = 总计: {unet_params + cond_params:.1f}M")
    
    # 数据集（使用预编码的latent）
    dataset = LatentDataset(
        latent_dir=latent_dir,
        split="train"  # 使用训练集
    )
    
    # 验证数据集
    if len(dataset) == 0:
        print("错误：数据集为空！")
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
    
    # 优化器（同时优化UNet和条件编码器）
    # 增强weight_decay以防止过拟合
    optimizer = torch.optim.AdamW(
        list(unet.parameters()) + list(class_conditioner.parameters()),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.02,  # 增强L2正则化（从0.015→0.02）
    )
    
    # 学习率调度（余弦退火 + warmup）
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * num_epochs,
    )
    
    # 准备训练（包括两个模型）
    unet, class_conditioner, optimizer, dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        unet, class_conditioner, optimizer, dataloader, val_dataloader, lr_scheduler
    )
    
    # 开始训练
    print(f"训练: {num_epochs}轮, 共{num_epochs * len(dataloader)}步")
    print(f"早停机制: 验证损失{early_stopping_patience}个epoch无改善则停止")
    
    # 早停和最佳模型跟踪
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = None  # 跟踪最佳模型路径，用于删除旧模型
    
    # 训练循环
    global_step = 0
    num_uncond_label = 31  # 使用31作为无条件标签（超出用户ID范围）
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
            
            # 10%概率使用无条件训练（Classifier-Free Guidance）
            if torch.rand(1).item() < 0.1:
                uncond_labels = torch.full_like(labels, num_uncond_label)
            else:
                uncond_labels = labels
            
            # 生成条件嵌入（使用Cross-Attention）
            encoder_hidden_states = class_conditioner(uncond_labels)
            
            # 预测噪声（SD官方方式）
            noise_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states  # Cross-Attention条件
            ).sample
            loss = F.mse_loss(noise_pred, noise)
            
            # 反向传播和优化
            accelerator.backward(loss)
            # 梯度裁剪（防止梯度爆炸）
            accelerator.clip_grad_norm_(
                list(unet.parameters()) + list(class_conditioner.parameters()), 
                1.0
            )
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
        
        # 关闭进度条并输出epoch统计
        progress_bar.close()
        avg_train_loss = sum(epoch_losses) / len(epoch_losses)
        
        # 验证集评估
        val_losses = []
        unet.eval()
        class_conditioner.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                latents = batch["latents"]
                labels = batch["labels"]
                labels = labels.to(latents.device)
                
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=latents.device
                ).long()
                
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # 使用Cross-Attention
                encoder_hidden_states = class_conditioner(labels)
                noise_pred = unet(
                    noisy_latents, 
                    timesteps, 
                    encoder_hidden_states=encoder_hidden_states
                ).sample
                val_loss = F.mse_loss(noise_pred, noise)
                val_losses.append(val_loss.item())
        
        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0
        unet.train()
        class_conditioner.train()
        
        # 早停和最佳模型保存逻辑
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # 保存最佳模型（删除旧的最佳模型）
            os.makedirs(output_dir, exist_ok=True)
            
            # 删除旧的最佳模型
            if best_model_path is not None and os.path.exists(best_model_path):
                os.remove(best_model_path)
                best_cond_path = best_model_path.replace("unet", "conditioner")
                if os.path.exists(best_cond_path):
                    os.remove(best_cond_path)
                print(f"删除旧模型: {os.path.basename(best_model_path)}")
            
            # 保存新的最佳模型（UNet + 条件编码器）
            best_model_path = os.path.join(output_dir, f"best_unet_epoch{epoch+1}_valloss{avg_val_loss:.4f}.pt")
            best_cond_path = os.path.join(output_dir, f"best_conditioner_epoch{epoch+1}_valloss{avg_val_loss:.4f}.pt")
            
            torch.save(
                accelerator.unwrap_model(unet).state_dict(),
                best_model_path
            )
            torch.save(
                accelerator.unwrap_model(class_conditioner).state_dict(),
                best_cond_path
            )
            print(f"Epoch {epoch+1}/{num_epochs}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f} ★NEW BEST★")
        else:
            patience_counter += 1
            print(f"Epoch {epoch+1}/{num_epochs}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f} (无改善 {patience_counter}/{early_stopping_patience})")
            
            # 检查是否触发早停
            if patience_counter >= early_stopping_patience:
                print(f"\n早停触发！验证损失已{early_stopping_patience}个epoch无改善")
                print(f"最佳验证损失: {best_val_loss:.4f}")
                print(f"最佳模型: {os.path.basename(best_model_path)}")
                break
        
        # 每N个epoch生成可视化样本
        if (epoch + 1) % sample_every_n_epochs == 0:
            generate_latent_samples(
                unet, class_conditioner, vae_wrapper, noise_scheduler,
                epoch, global_step
            )
    
    # 训练结束后的处理
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存最后一个epoch的模型（作为备份）
    last_unet_path = os.path.join(output_dir, "last_unet.pt")
    last_cond_path = os.path.join(output_dir, "last_conditioner.pt")
    torch.save(accelerator.unwrap_model(unet).state_dict(), last_unet_path)
    torch.save(accelerator.unwrap_model(class_conditioner).state_dict(), last_cond_path)
    print(f"\n最后epoch模型已保存: {os.path.basename(last_unet_path)}, {os.path.basename(last_cond_path)}")
    
    # 创建最佳模型的符号链接（便于加载）
    if best_model_path is not None:
        import shutil
        best_unet_link = os.path.join(output_dir, "best_unet.pt")
        best_cond_link = os.path.join(output_dir, "best_conditioner.pt")
        shutil.copy(best_model_path, best_unet_link)
        
        best_cond_path = best_model_path.replace("unet", "conditioner")
        if os.path.exists(best_cond_path):
            shutil.copy(best_cond_path, best_cond_link)
        
        print(f"最佳模型已复制: best_unet.pt, best_conditioner.pt")
        print(f"  └─ 来源: {os.path.basename(best_model_path)}")
        print(f"  └─ 最佳验证损失: {best_val_loss:.4f}")
    
    # 保存完整配置（用于推理，使用最佳模型）
    pipeline_config = {
        "scheduler": noise_scheduler,
        "vae_path": vae_path,
        "latent_dir": latent_dir,
        "best_val_loss": best_val_loss,
        "best_model_path": best_model_path,
    }
    torch.save(pipeline_config, os.path.join(output_dir, "training_config.pt"))
    
    print(f"\n所有模型和配置已保存到: {output_dir}")
    
    return unet, vae_wrapper, class_conditioner


# ============ 6. 生成样本 ============
def generate_latent_samples(unet, class_conditioner, vae_wrapper, scheduler, epoch, step, guidance_scale=3.5):
    """条件生成样本（选择4个不同用户）"""
    unet.eval()
    class_conditioner.eval()
    
    with torch.no_grad():
        # 初始化随机噪声
        batch_size = 4
        latents = torch.randn((batch_size, 4, 32, 32), device=unet.device)
        
        # 选择不同的用户ID进行条件生成（分散选择）
        class_labels = torch.tensor([0, 7, 14, 21], device=unet.device, dtype=torch.long)
        
        # DDIM采样（100步，提高质量）
        scheduler.set_timesteps(100)
        
        for t in scheduler.timesteps:
            # Classifier-Free Guidance（SD官方做法）
            if guidance_scale > 1.0:
                # 合并条件和无条件输入
                latent_input = torch.cat([latents] * 2)
                # 正确扩展时间步以匹配batch大小
                t_input = t.unsqueeze(0).expand(latent_input.shape[0])
                
                # 条件嵌入
                cond_emb = class_conditioner(class_labels)
                uncond_labels = torch.full_like(class_labels, 31)
                uncond_emb = class_conditioner(uncond_labels)
                encoder_hidden_states = torch.cat([cond_emb, uncond_emb])
                
                # 预测噪声
                noise_pred = unet(latent_input, t_input, encoder_hidden_states=encoder_hidden_states).sample
                
                # 分离条件和无条件预测
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                # 仅条件生成
                encoder_hidden_states = class_conditioner(class_labels)
                noise_pred = unet(latents, t, encoder_hidden_states=encoder_hidden_states).sample
            
            latents = scheduler.step(noise_pred, t, latents).prev_sample
        
        # 解码到图像空间
        images = vae_wrapper.decode(latents)
        images = (images + 1) / 2  # [-1,1] -> [0,1]
        images = images.clamp(0, 1).cpu()
    
    unet.train()
    class_conditioner.train()
    
    # 保存结果
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for i, ax in enumerate(axes):
        ax.imshow(images[i].permute(1, 2, 0).numpy())
        ax.axis('off')
        ax.set_title(f'User {class_labels[i].item()}')
    plt.suptitle(f'Epoch {epoch+1} (100-step DDIM, guidance=3.5)')
    plt.savefig(f'/kaggle/working/samples_epoch_{epoch+1}.png', dpi=100, bbox_inches='tight')
    plt.close()


# ============ 7. 推理pipeline ============
class LatentDiffusionPipeline:
    """SD架构的推理pipeline（支持Cross-Attention）"""
    
    def __init__(self, unet, class_conditioner, vae_wrapper, scheduler):
        self.unet = unet
        self.class_conditioner = class_conditioner
        self.vae_wrapper = vae_wrapper
        self.scheduler = scheduler
    
    @torch.no_grad()
    def __call__(self, batch_size=4, num_inference_steps=50, ddim=True, class_labels=None, guidance_scale=2.0):
        # 设置为eval模式
        self.unet.eval()
        self.class_conditioner.eval()
        
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
            class_labels = torch.randint(0, 31, (batch_size,), device=self.unet.device)
        elif not isinstance(class_labels, torch.Tensor):
            class_labels = torch.tensor(class_labels, device=self.unet.device, dtype=torch.long)
        
        # 去噪循环（SD官方CFG方式）
        for t in tqdm(self.scheduler.timesteps, desc="生成中"):
            if guidance_scale > 1.0:
                # Classifier-Free Guidance（合并batch）
                latent_input = torch.cat([latents] * 2)
                # 正确扩展时间步以匹配batch大小
                t_input = t.unsqueeze(0).expand(latent_input.shape[0])
                
                # 条件嵌入
                cond_emb = self.class_conditioner(class_labels)
                uncond_labels = torch.full_like(class_labels, 31)
                uncond_emb = self.class_conditioner(uncond_labels)
                encoder_hidden_states = torch.cat([cond_emb, uncond_emb])
                
                # 预测噪声
                noise_pred = self.unet(latent_input, t_input, encoder_hidden_states=encoder_hidden_states).sample
                
                # 分离并组合
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                # 仅条件生成
                encoder_hidden_states = self.class_conditioner(class_labels)
                noise_pred = self.unet(latents, t, encoder_hidden_states=encoder_hidden_states).sample
            
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
    # 训练（SD官方架构 + Cross-Attention）
    print("开始Latent Diffusion训练（SD架构 + 强正则化）...")
    
    unet, vae_wrapper, class_conditioner = train_latent_diffusion(
        vae_path="/kaggle/input/kl-vae-best-pt/kl_vae_best.pt",  # 必需！用于解码
        latent_dir="/kaggle/input/data-latent2",  # 预编码的latent
        output_dir="/kaggle/working/latent_diffusion",
        num_epochs=200,  # 最多200轮（早停会提前结束）
        batch_size=64,  # 增大batch size（有10G空余显存）
        learning_rate=8e-5,  # SD架构：适中学习率
        sample_every_n_epochs=5,  # 每5个epoch生成可视化样本
        early_stopping_patience=12,  # 早停：12个epoch验证损失无改善则停止
    )
    
    # 测试生成
    print("生成测试样本...")
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        prediction_type="epsilon",
    )
    
    pipeline = LatentDiffusionPipeline(unet, class_conditioner, vae_wrapper, scheduler)
    
    # 生成8个样本，选择8个不同的用户
    selected_users = [0, 4, 8, 12, 16, 20, 24, 28]  # 均匀分布的用户ID
    images = pipeline(
        batch_size=8, 
        num_inference_steps=100,  # DDIM 100步（提高细节和锐利度）
        class_labels=selected_users,  # 指定用户
        guidance_scale=3.5  # 提高guidance强度（用户差异小时需要更强的条件引导）
    )
    
    # 保存结果
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, (ax, img) in enumerate(zip(axes.flat, images)):
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f'User {selected_users[i]}')
    plt.suptitle('Conditional Generation (100-step DDIM, guidance=3.5)')
    plt.savefig('/kaggle/working/final_results.png')
    plt.close()
    
    print("完成！结果保存在 /kaggle/working/")
