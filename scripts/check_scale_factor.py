"""
检查和计算VAE的正确scale_factor

该脚本帮助确定VAE的正确scale_factor值
基于DDPM/LDM的标准实践：scale_factor = 1 / latent.std()
"""

import torch
import numpy as np
from pathlib import Path

def analyze_scale_factor():
    """
    分析scale_factor的正确值
    
    基于LDM/DDPM的实践：
    1. scale_factor用于将latent归一化到单位标准差
    2. 计算方式：scale_factor = 1 / latent.std()
    3. 常见值：
       - Stable Diffusion VAE: 0.18215 (对应std ≈ 5.49)
       - 一般VAE: 根据实际latent分布计算
    """
    
    print("""
    ============================================================
    VAE Scale Factor 分析
    ============================================================
    
    对于DDPM/LDM，scale_factor的作用：
    - 编码时：z_scaled = z * scale_factor
    - 解码时：z_original = z_scaled / scale_factor
    
    目的：确保latent空间有适合扩散过程的数值范围
    
    计算方法：
    1. 如果VAE训练时使用了scale_by_std：
       scale_factor = 1 / latent.std()
    
    2. 如果未使用scale_by_std：
       需要从checkpoint中读取或根据实际latent分布计算
    
    ============================================================
    
    根据您刚才运行的prepare_microdoppler_data.py输出：
    
    源域训练集 Latent statistics:
      Mean: 0.0603
      Std: 1.5414
      Min: -9.2399
      Max: 9.1938
    
    源域验证集 Latent statistics:
      Mean: 0.0617
      Std: 1.5402
      Min: -8.2493
      Max: 8.2746
    
    目标域训练集 Latent statistics:
      Mean: 0.0432
      Std: 1.5687
      Min: -8.6123
      Max: 8.0289
    
    ============================================================
    分析结果：
    """)
    
    # 基于实际数据计算推荐的scale_factor
    source_std = 1.5414
    target_std = 1.5687
    avg_std = (source_std + target_std) / 2
    
    print(f"    源域latent标准差: {source_std:.4f}")
    print(f"    目标域latent标准差: {target_std:.4f}")
    print(f"    平均标准差: {avg_std:.4f}")
    print()
    
    # 不同策略的scale_factor
    scale_factor_by_std = 1.0 / avg_std
    scale_factor_unity = 1.0
    
    print(f"    策略1 - 基于标准差归一化:")
    print(f"      scale_factor = 1 / std = {scale_factor_by_std:.4f}")
    print(f"      缩放后std ≈ 1.0 (适合标准高斯扩散)")
    print()
    
    print(f"    策略2 - 保持原始值 (scale_factor = 1.0):")
    print(f"      当前您的设置")
    print(f"      缩放后std ≈ {avg_std:.4f}")
    print()
    
    print("""
    ============================================================
    建议：
    ============================================================
    
    1. 如果您的VAE checkpoint中已保存scale_factor：
       - SimplifiedVAVAE会自动读取（第99-107行）
       - 应该信任checkpoint中的值
    
    2. 如果checkpoint中没有scale_factor（显示1.0）：
       有两种选择：
       
       a) 保持scale_factor = 1.0
          - 优点：不改变原始latent分布
          - 缺点：latent std较大(~1.54)，可能影响扩散训练稳定性
       
       b) 使用scale_factor = 0.6400 (1/1.5550)
          - 优点：归一化到单位方差，更稳定的扩散训练
          - 缺点：需要手动设置
    
    3. 实验建议：
       - 先用scale_factor = 1.0训练看效果
       - 如果训练不稳定或生成质量差，尝试0.6400
    
    ============================================================
    如何手动设置scale_factor（如果需要）：
    ============================================================
    
    在simplified_vavae.py的__init__中，checkpoint加载后添加：
    
    # 如果checkpoint中没有scale_factor，手动设置
    if self.scale_factor == 1.0:
        self.scale_factor = 0.6400  # 基于实际latent分布计算
        print(f"手动设置scale_factor为 {self.scale_factor}")
    
    ============================================================
    """)

if __name__ == "__main__":
    analyze_scale_factor()
