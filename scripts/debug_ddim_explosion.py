"""
专门调试DDIM采样数值爆炸问题
"""
import torch
from diffusers import DDIMScheduler, DDPMScheduler
from diffusers import UNet2DConditionModel
import matplotlib.pyplot as plt
import numpy as np

def debug_ddim_explosion():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=== DDIM数值爆炸调试 ===\n")
    
    # 1. 创建简单的UNet
    unet = UNet2DConditionModel(
        sample_size=32,
        in_channels=4,
        out_channels=4,
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        block_out_channels=(128, 256, 512, 512),
        layers_per_block=2,
        num_class_embeds=32,  # 31个真实类别 + 1个null class
        class_embed_type="timestep",
    ).to(device)
    
    # 2. 测试不同的调度器配置
    configs = [
        {
            "name": "标准DDIM",
            "params": {
                "num_train_timesteps": 1000,
                "beta_start": 0.0001,
                "beta_end": 0.02,
                "beta_schedule": "linear",
                "clip_sample": False,
                "set_alpha_to_one": False,
                "steps_offset": 1,
                "prediction_type": "epsilon",
            }
        },
        {
            "name": "稳定DDIM (clip_sample=True)",
            "params": {
                "num_train_timesteps": 1000,
                "beta_start": 0.0001,
                "beta_end": 0.02,
                "beta_schedule": "linear",
                "clip_sample": True,  # 裁剪输出
                "set_alpha_to_one": False,
                "steps_offset": 1,
                "prediction_type": "epsilon",
            }
        },
        {
            "name": "DDIM (set_alpha_to_one=True)",
            "params": {
                "num_train_timesteps": 1000,
                "beta_start": 0.0001,
                "beta_end": 0.02,
                "beta_schedule": "linear",
                "clip_sample": False,
                "set_alpha_to_one": True,  # 第一步alpha=1
                "steps_offset": 1,
                "prediction_type": "epsilon",
            }
        },
    ]
    
    # 3. 测试每种配置
    for config in configs:
        print(f"\n测试: {config['name']}")
        print("-" * 40)
        
        scheduler = DDIMScheduler(**config['params'])
        scheduler.set_timesteps(50)
        
        # 初始噪声
        x_t = torch.randn(1, 4, 32, 32).to(device)
        label = torch.tensor([0]).to(device)
        # 创建dummy的encoder_hidden_states（使用1280维）
        dummy_encoder_hidden_states = torch.zeros(1, 1, 1280).to(device)
        
        stds = [x_t.std().item()]
        
        # 模拟去噪过程
        with torch.no_grad():
            for i, t in enumerate(scheduler.timesteps[:10]):  # 只看前10步
                # 扩展时间步
                timestep = t.expand(x_t.shape[0]).to(device)
                
                # 预测噪声（使用真实UNet）
                noise_pred = unet(x_t, timestep, class_labels=label, 
                                encoder_hidden_states=dummy_encoder_hidden_states,
                                return_dict=False)[0]
                
                # DDIM步骤
                x_t = scheduler.step(noise_pred, t, x_t, return_dict=False)[0]
                
                stds.append(x_t.std().item())
                
                if i < 5:  # 打印前5步
                    print(f"  Step {i}, t={t}: std={x_t.std():.3f}, "
                          f"noise_pred_std={noise_pred.std():.3f}, "
                          f"range=[{x_t.min():.2f}, {x_t.max():.2f}]")
        
        # 判断是否爆炸
        if stds[-1] > stds[0] * 2:
            print(f"  ⚠️ 数值爆炸! 初始std={stds[0]:.3f}, 最终std={stds[-1]:.3f}")
        else:
            print(f"  ✓ 数值稳定! 初始std={stds[0]:.3f}, 最终std={stds[-1]:.3f}")
    
    # 4. 测试不同的噪声预测缩放
    print("\n\n=== 测试噪声预测缩放 ===")
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        clip_sample=False,
        prediction_type="epsilon",
    )
    scheduler.set_timesteps(50)
    
    for scale in [0.1, 0.5, 1.0, 2.0]:
        print(f"\n噪声缩放因子: {scale}")
        x_t = torch.randn(1, 4, 32, 32).to(device)
        
        with torch.no_grad():
            for i, t in enumerate(scheduler.timesteps[:5]):
                timestep = t.expand(x_t.shape[0]).to(device)
                noise_pred = unet(x_t, timestep, class_labels=label, 
                                encoder_hidden_states=dummy_encoder_hidden_states,
                                return_dict=False)[0]
                noise_pred = noise_pred * scale  # 缩放噪声预测
                x_t = scheduler.step(noise_pred, t, x_t, return_dict=False)[0]
                
            print(f"  最终std: {x_t.std():.3f}")

if __name__ == "__main__":
    debug_ddim_explosion()
