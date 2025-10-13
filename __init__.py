"""
VAE版本的Guided Diffusion
基于OpenAI的Diffusion Models Beat GANS on Image Synthesis
修改以支持VAE latent空间训练
"""

# 导出主要模块
from . import dist_util, logger
from .gaussian_diffusion import GaussianDiffusion
from .unet import UNetModel
