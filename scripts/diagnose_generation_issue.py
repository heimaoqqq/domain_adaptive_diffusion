"""
综合诊断脚本：系统性检查VAE和DDPM问题
"""
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from domain_adaptive_diffusion.vae.kl_vae import KL_VAE
from domain_adaptive_diffusion.scripts.train_adm_official import SimplifiedADMUNet, GaussianDiffusion
from domain_adaptive_diffusion.utils.helpers import load_config
from domain_adaptive_diffusion.utils.data_loader import create_dataloaders
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F


def diagnose_all(config_path, vae_checkpoint, data_path, checkpoint_dir, device='cuda'):
    """综合诊断VAE和DDPM的问题"""
    
    print("="*60)
    print("开始综合诊断")
    print("="*60)
    
    # 加载配置
    config = load_config(config_path)
    
    # 1. 检查原始数据
    print("\n[1] 检查训练数据")
    print("-"*40)
    
    train_loader, _ = create_dataloaders(
        data_path=data_path,
        phase='pretrain',
        batch_size=8,
        num_workers=0,
        augmentation=False,
        device=device
    )
    
    batch = next(iter(train_loader))
    latents = batch['latent'].to(device)
    labels = batch['label'].to(device)
    
    print(f"✓ Latent shape: {latents.shape}")
    print(f"✓ Latent统计: mean={latents.mean():.4f}, std={latents.std():.4f}")
    print(f"✓ Latent范围: [{latents.min():.4f}, {latents.max():.4f}]")
    print(f"✓ 标签: {labels.tolist()}")
    
    # 检查latent的方差
    latent_var = latents.var(dim=(0,2,3)).mean()
    print(f"✓ Latent平均方差: {latent_var:.4f}")
    if latent_var < 0.01:
        print("⚠️ 警告: Latent方差很小，可能缺乏特征信息！")
    
    # 2. 检查VAE解码能力
    print("\n[2] 检查VAE解码")
    print("-"*40)
    
    vae = KL_VAE(ddconfig=None, embed_dim=4, scale_factor=0.18215)
    checkpoint = torch.load(vae_checkpoint, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        vae.load_state_dict(checkpoint['model_state_dict'])
    else:
        vae.load_state_dict(checkpoint)
    vae.eval()
    vae = vae.to(device)
    
    with torch.no_grad():
        # 解码原始latents
        decoded = vae.decode_latents(latents)
        
        print(f"✓ 解码图像统计: mean={decoded.mean():.4f}, std={decoded.std():.4f}")
        print(f"✓ 解码图像范围: [{decoded.min():.4f}, {decoded.max():.4f}]")
        
        # 计算图像的边缘强度（检查是否有特征）
        gray = decoded.mean(dim=1, keepdim=True)  # 转为灰度
        # 使用简单的梯度检测
        dx = gray[:, :, :, 1:] - gray[:, :, :, :-1]
        dy = gray[:, :, 1:, :] - gray[:, :, :-1, :]
        edge_strength = (dx.abs().mean() + dy.abs().mean()) / 2
        print(f"✓ 图像边缘强度: {edge_strength:.4f}")
        
        if edge_strength < 0.01:
            print("⚠️ 警告: 解码的图像几乎没有边缘特征！")
        
        # 保存解码的图像
        if decoded.min() >= -0.1 and decoded.max() <= 1.1:
            decoded_norm = torch.clamp(decoded, 0, 1)
        elif decoded.min() >= -1.1 and decoded.max() <= 1.1:
            decoded_norm = (decoded + 1) / 2
        else:
            decoded_norm = (decoded - decoded.min()) / (decoded.max() - decoded.min() + 1e-8)
        
        save_image(decoded_norm, 'diagnose_vae_decoded.png', nrow=4)
        print("✓ VAE解码图像已保存到: diagnose_vae_decoded.png")
    
    # 3. 测试VAE的重建能力
    print("\n[3] 测试VAE重建能力")
    print("-"*40)
    
    # 创建有明显特征的测试图像
    test_images = torch.zeros(4, 3, 256, 256, device=device)
    # 图像1: 垂直条纹
    test_images[0, :, :, ::32] = 1.0
    # 图像2: 水平条纹
    test_images[1, :, ::32, :] = 1.0
    # 图像3: 棋盘格
    for i in range(0, 256, 32):
        for j in range(0, 256, 32):
            if (i//32 + j//32) % 2 == 0:
                test_images[2, :, i:i+32, j:j+32] = 1.0
    # 图像4: 中心圆形
    y, x = torch.meshgrid(torch.arange(256), torch.arange(256))
    mask = ((x - 128)**2 + (y - 128)**2) < 50**2
    test_images[3, :, mask] = 1.0
    
    # 转换到VAE输入范围 [-1, 1]
    test_images = test_images * 2 - 1
    
    with torch.no_grad():
        # 编码再解码
        test_latents = vae.encode_images(test_images)
        reconstructed = vae.decode_latents(test_latents)
        
        print(f"✓ 测试latents统计: mean={test_latents.mean():.4f}, std={test_latents.std():.4f}")
        print(f"✓ 重建图像统计: mean={reconstructed.mean():.4f}, std={reconstructed.std():.4f}")
        
        # 计算重建误差
        if reconstructed.min() >= -1.1 and reconstructed.max() <= 1.1:
            reconstructed_normalized = reconstructed
        else:
            reconstructed_normalized = reconstructed * 2 - 1
        
        mse = F.mse_loss(reconstructed_normalized, test_images)
        print(f"✓ 重建MSE: {mse:.4f}")
        
        if mse > 0.1:
            print("⚠️ 警告: VAE重建误差较大！")
        
        # 保存对比图
        comparison = torch.cat([test_images, reconstructed_normalized], dim=0)
        comparison = (comparison + 1) / 2
        save_image(comparison, 'diagnose_vae_reconstruction.png', nrow=4)
        print("✓ VAE重建对比已保存到: diagnose_vae_reconstruction.png")
    
    # 4. 检查DDPM模型
    print("\n[4] 检查DDPM模型")
    print("-"*40)
    
    # 初始化模型
    model = SimplifiedADMUNet(
        in_channels=4,
        model_channels=int(config['model']['model_channels']),
        out_channels=4,
        num_res_blocks=int(config['model']['num_res_blocks']),
        channel_mult=config['model']['channel_mult'],
        dropout=float(config['model']['dropout']),
        num_classes=int(config['data']['num_classes']),
        use_scale_shift_norm=config['model']['use_scale_shift_norm'],
        use_fp16=config.get('use_fp16', False),
        resblock_updown=config['model'].get('resblock_updown', False)
    ).to(device)
    
    # 查找最新的checkpoint
    checkpoint_path = Path(checkpoint_dir)
    checkpoints = list(checkpoint_path.glob("checkpoint_epoch_*.pt"))
    
    if checkpoints:
        latest_checkpoint = sorted(checkpoints)[-1]
        print(f"✓ 加载checkpoint: {latest_checkpoint}")
        ckpt = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        epoch = ckpt['epoch']
        print(f"✓ 已训练epochs: {epoch}")
    else:
        print("⚠️ 未找到训练的checkpoint，使用随机初始化模型")
        epoch = 0
    
    model.eval()
    
    # 初始化扩散
    diffusion = GaussianDiffusion(
        timesteps=int(config['diffusion']['timesteps']),
        beta_schedule=config['diffusion']['beta_schedule'],
        beta_start=float(config['diffusion']['beta_start']),
        beta_end=float(config['diffusion']['beta_end'])
    )
    
    # 5. 检查条件响应
    print("\n[5] 检查DDPM条件响应")
    print("-"*40)
    
    with torch.no_grad():
        # 在中间时间步测试
        t = 500
        t_batch = torch.full((latents.shape[0],), t, device=device, dtype=torch.long)
        
        # 添加噪声
        noise = torch.randn_like(latents)
        x_t = diffusion.q_sample(latents, t_batch, noise)
        
        # 使用正确的标签预测
        pred_noise_correct = model(x_t, t_batch, y=labels)
        
        # 使用随机标签预测
        random_labels = torch.randint(0, int(config['data']['num_classes']), 
                                     (latents.shape[0],), device=device)
        pred_noise_random = model(x_t, t_batch, y=random_labels)
        
        # 计算差异
        cond_diff = (pred_noise_correct - pred_noise_random).abs().mean()
        print(f"✓ 正确vs随机标签的预测差异: {cond_diff:.4f}")
        
        if cond_diff < 0.01:
            print("⚠️ 严重问题: 模型对不同条件几乎没有响应！")
            print("   可能原因:")
            print("   - 类别嵌入没有正确初始化")
            print("   - Scale-shift norm没有生效")
            print("   - 训练时标签可能有问题")
        elif cond_diff < 0.05:
            print("⚠️ 警告: 条件响应较弱")
        else:
            print("✓ 条件响应正常")
    
    # 6. 完整生成测试
    print("\n[6] 完整生成测试")
    print("-"*40)
    
    with torch.no_grad():
        # 为前4个用户生成样本
        test_labels = torch.arange(4, device=device)
        shape = (4, 4, 32, 32)
        
        # 记录生成过程
        x = torch.randn(shape, device=device)
        x_history = [x.clone()]
        
        print("✓ 开始生成过程...")
        for t in reversed(range(0, 1000, 100)):  # 每100步记录一次
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            x = diffusion.p_sample(model, x, t_batch, model_kwargs={'y': test_labels})
            x_history.append(x.clone())
            
            if t % 200 == 0:
                print(f"  t={t}: mean={x.mean():.4f}, std={x.std():.4f}")
        
        # 最终采样
        for t in reversed(range(100)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            x = diffusion.p_sample(model, x, t_batch, model_kwargs={'y': test_labels})
        
        print(f"\n✓ 生成完成:")
        print(f"  最终latent: mean={x.mean():.4f}, std={x.std():.4f}")
        print(f"  最终范围: [{x.min():.4f}, {x.max():.4f}]")
        
        # 计算生成过程中的变化
        total_change = (x - x_history[0]).abs().mean()
        print(f"  总变化量: {total_change:.4f}")
        
        if total_change < 0.5:
            print("⚠️ 警告: 生成过程几乎没有变化！")
        
        # 解码生成的样本
        generated_images = vae.decode_latents(x)
        
        # 归一化并保存
        if generated_images.min() >= -0.1 and generated_images.max() <= 1.1:
            generated_norm = torch.clamp(generated_images, 0, 1)
        elif generated_images.min() >= -1.1 and generated_images.max() <= 1.1:
            generated_norm = (generated_images + 1) / 2
        else:
            generated_norm = (generated_images - generated_images.min()) / \
                           (generated_images.max() - generated_images.min() + 1e-8)
        
        save_image(generated_norm, 'diagnose_generated_samples.png', nrow=2)
        print("✓ 生成样本已保存到: diagnose_generated_samples.png")
    
    # 7. 总结诊断结果
    print("\n" + "="*60)
    print("诊断总结")
    print("="*60)
    
    print("\n请检查生成的图像文件:")
    print("1. diagnose_vae_decoded.png - VAE解码的训练数据")
    print("2. diagnose_vae_reconstruction.png - VAE重建测试")
    print("3. diagnose_generated_samples.png - DDPM生成的样本")
    
    print("\n根据诊断结果，可能的问题:")
    if edge_strength < 0.01:
        print("• VAE解码的图像缺乏特征 → 检查VAE训练或数据准备")
    if cond_diff < 0.01:
        print("• DDPM条件响应极弱 → 检查标签嵌入和scale-shift norm")
    if total_change < 0.5:
        print("• 生成过程变化很小 → 检查扩散参数和模型架构")


def main():
    # 配置路径
    config_path = "domain_adaptive_diffusion/configs/adm_official.yaml"
    vae_checkpoint = "/kaggle/input/kl-vae-best-pt/kl_vae_best.pt"
    data_path = "/kaggle/working/data/processed"
    checkpoint_dir = "/kaggle/working/ddpm_checkpoints"
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 运行完整诊断
    diagnose_all(config_path, vae_checkpoint, data_path, checkpoint_dir, device)


if __name__ == "__main__":
    main()
