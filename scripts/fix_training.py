"""
修复训练中的scale问题
"""
import torch
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from domain_adaptive_diffusion.utils.data_loader import create_dataloaders
from domain_adaptive_diffusion.vae.kl_vae import KL_VAE
from torchvision.utils import save_image


def diagnose_and_fix():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. 加载一个batch看看
    train_loader, _ = create_dataloaders(
        data_path="/kaggle/working/data/processed",
        phase='pretrain',
        batch_size=4,
        num_workers=0,
        augmentation=False,
        device=device
    )
    
    batch = next(iter(train_loader))
    latents = batch['latent'].to(device)
    
    print("原始数据集中的latents:")
    print(f"  范围: [{latents.min():.4f}, {latents.max():.4f}]")
    print(f"  标准差: {latents.std():.4f}")
    
    # 2. 加载VAE测试解码
    vae = KL_VAE(ddconfig=None, embed_dim=4, scale_factor=0.18215)
    checkpoint = torch.load("/kaggle/input/kl-vae-best-pt/kl_vae_best.pt", map_location='cpu')
    if 'model_state_dict' in checkpoint:
        vae.load_state_dict(checkpoint['model_state_dict'])
    else:
        vae.load_state_dict(checkpoint)
    vae.eval().to(device)
    
    with torch.no_grad():
        # 测试1: 直接解码（使用decode_latents，会除以scale_factor）
        decoded1 = vae.decode_latents(latents)
        
        # 测试2: 先放大再解码（补偿scale_factor）
        latents_scaled = latents / 0.18215  # 反向补偿
        decoded2 = vae.decode_latents(latents_scaled)
        
        # 测试3: 绕过scale处理，直接用decoder
        decoded3 = vae.decode(latents)
    
    print("\n解码测试结果:")
    print(f"1. 直接decode_latents: 范围[{decoded1.min():.2f}, {decoded1.max():.2f}], std={decoded1.std():.3f}")
    print(f"2. 补偿后decode_latents: 范围[{decoded2.min():.2f}, {decoded2.max():.2f}], std={decoded2.std():.3f}")  
    print(f"3. 直接decode: 范围[{decoded3.min():.2f}, {decoded3.max():.2f}], std={decoded3.std():.3f}")
    
    # 保存对比
    images = []
    for decoded, name in [(decoded1, "direct"), (decoded2, "compensated"), (decoded3, "bypass")]:
        if decoded.min() >= -1.1 and decoded.max() <= 1.1:
            normalized = (decoded + 1) / 2
        else:
            normalized = (decoded - decoded.min()) / (decoded.max() - decoded.min() + 1e-8)
        images.append(normalized)
    
    comparison = torch.cat(images[:3], dim=0)
    save_image(comparison, 'scale_fix_comparison.png', nrow=4)
    print("\n✅ 对比图保存到: scale_fix_comparison.png")
    
    # 3. 给出解决方案
    print("\n" + "="*50)
    print("解决方案：")
    print("="*50)
    
    # 判断哪种方法最好
    stds = [decoded1.std(), decoded2.std(), decoded3.std()]
    best_idx = stds.index(max(stds))
    
    if best_idx == 0:
        print("✅ 当前的scale处理看起来是正确的")
    elif best_idx == 1:
        print("⚠️ 数据集中的latents已经包含了scale_factor!")
        print("\n修复方法 - 在VAEWrapper中修改decode方法：")
        print("```python")
        print("def decode(self, latents):")
        print("    with torch.no_grad():")
        print("        # 数据集已包含scale，不需要再除")
        print("        images = self.vae.decode(latents)")
        print("        return images")
        print("```")
    else:
        print("⚠️ 应该绕过scale_factor处理!")
        print("\n修复方法 - 修改VAEWrapper：")
        print("```python") 
        print("def decode(self, latents):")
        print("    with torch.no_grad():")
        print("        # 直接解码，不处理scale")
        print("        return self.vae.decode(latents)")
        print("```")


if __name__ == "__main__":
    diagnose_and_fix()
