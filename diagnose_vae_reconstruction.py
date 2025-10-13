"""
诊断VAE重建误差问题 - 专注于归一化和缩放
"""

import torch
import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from vae_wrapper import VAEInterface

def diagnose_vae():
    """诊断VAE重建问题"""
    print("=" * 60)
    print("VAE归一化诊断")
    print("=" * 60)
    
    # 创建VAE接口
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vae_path = "/kaggle/input/kl-vae-best-pt/kl_vae_best.pt"
    if not os.path.exists(vae_path):
        vae_path = "domain_adaptive_diffusion/vae/kl_vae_best.pt"
    
    vae = VAEInterface(vae_path=vae_path, device=device)
    
    print(f"\nVAE配置:")
    print(f"  scale_factor: {vae.scale_factor}")
    print(f"  设备: {device}")
    
    # 创建输出目录
    output_dir = Path("vae_reconstruction_analysis")
    output_dir.mkdir(exist_ok=True)
    
    # 测试真实的微多普勒图像
    print("\n1. 测试真实微多普勒图像")
    test_real_microdoppler_images(vae, output_dir)
    
    # 测试更多真实图像的统计
    print("\n2. 分析真实数据集的重建质量")
    analyze_dataset_reconstruction(vae, output_dir)
    
    print(f"\n✅ 分析完成！对比图保存在 {output_dir} 目录")
        
def test_reconstruction(vae, image, name):
    """测试单个图像的重建"""
    print(f"  输入范围: [{image.min():.3f}, {image.max():.3f}]")
    
    # 编码
    with torch.no_grad():
        latent = vae.encode_batch(image)
    print(f"  编码latent: shape={latent.shape}, mean={latent.mean():.4f}, std={latent.std():.4f}")
    print(f"  编码latent范围: [{latent.min():.3f}, {latent.max():.3f}]")
    
    # 解码
    with torch.no_grad():
        recon = vae.decode_batch(latent)
    print(f"  重建范围: [{recon.min():.3f}, {recon.max():.3f}]")
    
    # 将两者都clamp到[0,1]来计算误差
    image_clamped = torch.clamp(image, 0, 1)
    recon_clamped = torch.clamp(recon, 0, 1)
    
    # 计算误差
    mse = ((image_clamped - recon_clamped) ** 2).mean().item()
    mae = (image_clamped - recon_clamped).abs().mean().item()
    
    print(f"  重建误差: MSE={mse:.4f}, MAE={mae:.4f}")
    
    # 分析误差分布
    error = (image_clamped - recon_clamped).abs()
    print(f"  误差分布: mean={error.mean():.4f}, max={error.max():.4f}")
    
def analyze_vae_internals(vae, test_img):
    """分析VAE内部处理细节"""
    print("\n分析VAE内部处理:")
    
    # 直接调用VAE的encode方法查看中间结果
    with torch.no_grad():
        # 编码
        posterior = vae.vae.encode(test_img)
        z = posterior.sample()
        
        print(f"  原始latent (未缩放): mean={z.mean():.4f}, std={z.std():.4f}")
        print(f"  原始latent范围: [{z.min():.3f}, {z.max():.3f}]")
        
        # 应用scale factor
        z_scaled = z * vae.scale_factor
        print(f"  缩放后latent: mean={z_scaled.mean():.4f}, std={z_scaled.std():.4f}")
        print(f"  缩放后范围: [{z_scaled.min():.3f}, {z_scaled.max():.3f}]")
        
        # 解码
        z_unscaled = z_scaled / vae.scale_factor
        decoded = vae.vae.decode(z_unscaled)
        
        print(f"  解码输出范围: [{decoded.min():.3f}, {decoded.max():.3f}]")
        
def test_reconstruction_with_save(vae, image, name, save_path):
    """测试重建并保存对比图"""
    print(f"  输入范围: [{image.min():.3f}, {image.max():.3f}]")
    
    # 编码
    with torch.no_grad():
        latent = vae.encode_batch(image)
    print(f"  编码latent: shape={latent.shape}, mean={latent.mean():.4f}, std={latent.std():.4f}")
    
    # 解码
    with torch.no_grad():
        recon = vae.decode_batch(latent)
    
    # 计算误差
    image_clamped = torch.clamp(image, 0, 1)
    recon_clamped = torch.clamp(recon, 0, 1)
    mse = ((image_clamped - recon_clamped) ** 2).mean().item()
    mae = (image_clamped - recon_clamped).abs().mean().item()
    print(f"  重建误差: MSE={mse:.4f}, MAE={mae:.4f}")
    
    # 保存对比图
    save_comparison_figure(image_clamped, recon_clamped, save_path, name, mse, mae)
    
    return recon_clamped

def save_comparison_figure(original, reconstructed, save_path, title, mse, mae):
    """保存对比图"""
    # 转换为numpy并移到CPU
    orig_np = original[0].cpu().numpy().transpose(1, 2, 0)
    recon_np = reconstructed[0].cpu().numpy().transpose(1, 2, 0)
    diff_np = np.abs(orig_np - recon_np)
    
    # 创建图形
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # 原始图像
    axes[0].imshow(orig_np)
    axes[0].set_title('原始图像')
    axes[0].axis('off')
    
    # 重建图像
    axes[1].imshow(recon_np)
    axes[1].set_title('重建图像')
    axes[1].axis('off')
    
    # 差异图
    im_diff = axes[2].imshow(diff_np, cmap='hot')
    axes[2].set_title('绝对差异')
    axes[2].axis('off')
    plt.colorbar(im_diff, ax=axes[2], fraction=0.046)
    
    # 差异直方图
    axes[3].hist(diff_np.flatten(), bins=50, edgecolor='black')
    axes[3].set_title(f'差异分布\nMSE={mse:.4f}, MAE={mae:.4f}')
    axes[3].set_xlabel('绝对差异')
    axes[3].set_ylabel('像素数')
    
    plt.suptitle(f'{title} - VAE重建质量分析', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  对比图已保存: {save_path}")

def test_real_microdoppler_images(vae, output_dir):
    """测试真实微多普勒图像"""
    
    # 查找数据集目录
    dataset_dirs = [
        Path("dataset/organized_gait_dataset"),
        Path("/kaggle/input/organized-gait-dataset"),
        Path("G:/VA-VAE/dataset/organized_gait_dataset")
    ]
    
    dataset_dir = None
    for d in dataset_dirs:
        if d.exists():
            dataset_dir = d
            break
    
    if dataset_dir is None:
        print("  未找到数据集目录，跳过真实图像测试")
        return
    
    # 获取一些示例图像
    subdirs = ["Normal_free", "Normal_line", "Bag_free", "Backpack_free"]
    test_images = []
    
    for subdir in subdirs:
        subdir_path = dataset_dir / subdir
        if subdir_path.exists():
            # 获取第一个用户的第一张图像
            user_dirs = sorted([d for d in subdir_path.iterdir() if d.is_dir()])
            if user_dirs:
                images = list(user_dirs[0].glob("*.jpg"))[:1]
                if images:
                    test_images.append((images[0], subdir))
    
    if not test_images:
        print("  未找到测试图像")
        return
    
    # 测试每张图像
    for img_path, gait_type in test_images:
        print(f"\n  测试 {gait_type} - {img_path.name}")
        
        # 加载图像
        pil_img = Image.open(img_path).convert("RGB")
        
        # 调整到256x256
        pil_img = pil_img.resize((256, 256), Image.LANCZOS)
        
        # 转换为tensor并归一化到[0, 1]
        img_array = np.array(pil_img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(vae.device)
        
        # 测试重建
        save_path = output_dir / f"real_{gait_type}.png"
        test_reconstruction_with_save(vae, img_tensor, f"真实图像-{gait_type}", save_path)

def analyze_dataset_reconstruction(vae, output_dir):
    """分析整个数据集的重建质量"""
    
    # 查找数据集目录
    dataset_dirs = [
        Path("dataset/organized_gait_dataset"),
        Path("/kaggle/input/organized-gait-dataset"),
        Path("G:/VA-VAE/dataset/organized_gait_dataset")
    ]
    
    dataset_dir = None
    for d in dataset_dirs:
        if d.exists():
            dataset_dir = d
            break
    
    if dataset_dir is None:
        print("  未找到数据集目录")
        return
    
    # 统计所有步态类型
    all_mse = []
    gait_stats = {}
    
    subdirs = ["Normal_free", "Normal_line", "Bag_free", "Bag_line", 
               "Backpack_free", "Backpack_line", "Bag_Phone_free", "Bag_Phone_line"]
    
    for subdir in subdirs:
        subdir_path = dataset_dir / subdir
        if not subdir_path.exists():
            continue
        
        print(f"\n  分析 {subdir}...")
        subdir_mse = []
        
        # 采样测试（每个用户取2张图）
        user_dirs = sorted([d for d in subdir_path.iterdir() if d.is_dir()])[:5]  # 测试前5个用户
        
        for user_dir in user_dirs:
            images = list(user_dir.glob("*.jpg"))[:2]  # 每个用户取2张
            
            for img_path in images:
                # 加载和预处理
                pil_img = Image.open(img_path).convert("RGB")
                pil_img = pil_img.resize((256, 256), Image.LANCZOS)
                img_array = np.array(pil_img).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(vae.device)
                
                # 计算重建误差
                with torch.no_grad():
                    latent = vae.encode_batch(img_tensor)
                    recon = vae.decode_batch(latent)
                    mse = ((img_tensor - recon) ** 2).mean().item()
                    subdir_mse.append(mse)
                    all_mse.append(mse)
        
        if subdir_mse:
            avg_mse = np.mean(subdir_mse)
            std_mse = np.std(subdir_mse)
            gait_stats[subdir] = {"avg": avg_mse, "std": std_mse, "samples": len(subdir_mse)}
            print(f"    MSE: {avg_mse:.5f} ± {std_mse:.5f} (n={len(subdir_mse)})")
    
    # 总体统计
    if all_mse:
        print("\n  📊 总体统计:")
        print(f"    平均MSE: {np.mean(all_mse):.5f}")
        print(f"    标准差: {np.std(all_mse):.5f}")
        print(f"    最小MSE: {np.min(all_mse):.5f}")
        print(f"    最大MSE: {np.max(all_mse):.5f}")
        print(f"    中位数: {np.median(all_mse):.5f}")
        print(f"    样本数: {len(all_mse)}")
        
        # 保存统计结果
        stats_path = output_dir / "reconstruction_stats.txt"
        with open(stats_path, 'w') as f:
            f.write("VAE重建质量统计\n")
            f.write("="*50 + "\n\n")
            f.write("各步态类型统计:\n")
            for gait, stats in gait_stats.items():
                f.write(f"{gait}: MSE={stats['avg']:.5f} ± {stats['std']:.5f} (n={stats['samples']})\n")
            f.write(f"\n总体统计:\n")
            f.write(f"平均MSE: {np.mean(all_mse):.5f}\n")
            f.write(f"标准差: {np.std(all_mse):.5f}\n")
            f.write(f"最小MSE: {np.min(all_mse):.5f}\n")
            f.write(f"最大MSE: {np.max(all_mse):.5f}\n")
            f.write(f"中位数: {np.median(all_mse):.5f}\n")
            f.write(f"样本数: {len(all_mse)}\n")
        
        print(f"\n  ✅ 统计结果已保存到 {stats_path}")
        
        # 结论
        avg_mse = np.mean(all_mse)
        if avg_mse < 0.01:
            print("\n  🎯 结论: VAE重建质量优秀！MSE < 0.01，完全适合扩散模型训练")
        elif avg_mse < 0.05:
            print("\n  ✅ 结论: VAE重建质量良好，MSE < 0.05，适合扩散模型训练")
        else:
            print("\n  ⚠️ 结论: VAE重建质量一般，可能需要调整")

if __name__ == "__main__":
    diagnose_vae()
