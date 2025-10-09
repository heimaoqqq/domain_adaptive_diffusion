"""
加载VA-VAE模型 - 使用与run_universal_guidance.py相同的方法
"""

import sys
import os
import tempfile
import yaml
from pathlib import Path
import torch
import torch.nn as nn


def load_vae_model(checkpoint_path: str, device: str = 'cuda') -> nn.Module:
    """
    加载VAE模型 - 使用与run_universal_guidance.py相同的方法
    
    Args:
        checkpoint_path: VAE checkpoint路径
        device: 计算设备
    
    Returns:
        加载的VAE模型
    """
    print(f"\n📦 加载VAE模型...")
    
    # 检查checkpoint路径
    if not checkpoint_path:
        raise ValueError("必须提供VAE checkpoint路径")
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"VAE checkpoint不存在: {checkpoint_path}")
    
    # 检查是否是我们训练的KL-VAE
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model_type = checkpoint.get('model_type', None)
        
        if model_type == 'kl_vae_ddpm':
            # 使用我们的KL-VAE
            print("   检测到KL-VAE checkpoint")
            vae_path = Path(__file__).parent.parent / 'vae'
            sys.path.insert(0, str(vae_path))
            from kl_vae import KL_VAE
            
            # 重建配置
            embed_dim = checkpoint.get('embed_dim', 4)
            scale_factor = checkpoint.get('scale_factor', 0.18215)
            
            # 创建模型
            vae = KL_VAE(embed_dim=embed_dim, scale_factor=scale_factor)
            vae.load_state_dict(checkpoint['model_state_dict'])
            vae = vae.to(device)
            vae.eval()
            
            print("   ✅ KL-VAE加载成功")
            print(f"   Latent channels: {embed_dim}")
            print(f"   Downsample factor: 4x (256->64)")
            print(f"   Scale factor: {scale_factor}")
            return vae
    except:
        pass  # 不是KL-VAE checkpoint，继续尝试其他方法
    
    # 检测运行环境
    is_kaggle = os.path.exists('/kaggle/working')
    
    # 如果在Kaggle环境，检查多个位置的simplified_vavae.py
    if is_kaggle:
        vae_paths = [
            '/kaggle/working/domain_adaptive_diffusion/utils/simplified_vavae.py',
            '/kaggle/working/simplified_vavae.py'
        ]
        
        for vae_path in vae_paths:
            if os.path.exists(vae_path):
                print(f"   使用simplified_vavae.py: {vae_path}")
                sys.path.insert(0, os.path.dirname(vae_path))
                from simplified_vavae import SimplifiedVAVAE
                
                vae = SimplifiedVAVAE(checkpoint_path=str(checkpoint_path))
                vae = vae.to(device)
                vae.eval()
                
                print("   ✅ VA-VAE加载成功")
                print(f"   Latent channels: 32")
                print(f"   Downsample factor: 16")
                scale_factor = getattr(vae, 'scale_factor', 1.0)
                print(f"   Scale factor: {scale_factor}")
                return vae
        else:
            raise FileNotFoundError("找不到simplified_vavae.py，请检查文件位置")
    
    # 添加LightningDiT路径（如果不在Kaggle或需要fallback）
    vavae_root = Path(__file__).parent.parent.parent  # 回到VA-VAE根目录
    lightningdit_path = vavae_root / "LightningDiT"
    if lightningdit_path.exists():
        sys.path.insert(0, str(lightningdit_path))
    
    try:
        from tokenizer.vavae import VA_VAE
        
        # 创建VAE配置（与run_universal_guidance.py完全一致）
        # 这个配置经过验证，用于您训练的VA-VAE checkpoint
        vae_config = {
            'ckpt_path': str(checkpoint_path),
            'model': {
                'base_learning_rate': 2.0e-05,
                'target': 'ldm.models.autoencoder.AutoencoderKL',
                'params': {
                    'monitor': 'val/rec_loss',
                    'embed_dim': 32,
                    'use_vf': 'dinov2',
                    'reverse_proj': True,
                    'ddconfig': {
                        'double_z': True, 'z_channels': 32, 'resolution': 256,
                        'in_channels': 3, 'out_ch': 3, 'ch': 128,
                        'ch_mult': [1, 1, 2, 2, 4], 'num_res_blocks': 2,
                        'attn_resolutions': [16], 'dropout': 0.0
                    },
                    'lossconfig': {
                        'target': 'ldm.modules.losses.contperceptual.LPIPSWithDiscriminator',
                        'params': {
                            'disc_start': 1, 'kl_weight': 1e-6,
                            'pixelloss_weight': 1.0, 'perceptual_weight': 1.0,
                        }
                    }
                }
            }
        }
        
        # 写入临时配置文件
        temp_config_fd, temp_config_path = tempfile.mkstemp(suffix='.yaml')
        with open(temp_config_path, 'w') as f:
            yaml.dump(vae_config, f)
        os.close(temp_config_fd)
        
        try:
            vae = VA_VAE(temp_config_path)
            if hasattr(vae, 'to'):
                vae = vae.to(device)
            if hasattr(vae, 'eval'):
                vae.eval()
            print("   ✅ VA-VAE加载成功")
            print(f"   Latent channels: 32")
            print(f"   Downsample factor: 16")
            # VA_VAE可能没有scale_factor属性，使用默认值1.0
            scale_factor = getattr(vae, 'scale_factor', 1.0)
            print(f"   Scale factor: {scale_factor}")
            # 如果没有scale_factor属性，添加它
            if not hasattr(vae, 'scale_factor'):
                vae.scale_factor = 1.0
            return vae
        finally:
            # 清理临时文件
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
    
    except ImportError as e:
        raise ImportError(f"无法导入VA_VAE: {e}\n请确保LightningDiT在正确的路径")
