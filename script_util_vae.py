"""
修改后的script_util - 支持VAE集成
主要修改：输入输出通道从3改为4
"""

import argparse
import inspect

try:
    from . import gaussian_diffusion as gd
    from .respace import SpacedDiffusion, space_timesteps
    from .unet import UNetModel
except ImportError:
    import gaussian_diffusion as gd
    from respace import SpacedDiffusion, space_timesteps
    from unet import UNetModel

# 微多普勒数据集的类别数
NUM_CLASSES = 31  # 31个用户


def model_and_diffusion_defaults_vae():
    """
    VAE版本的默认模型和扩散参数
    主要改动：通道数从3改为4
    """
    return dict(
        # 模型架构
        image_size=64,
        num_channels=128,  # 基础通道数（小模型）
        num_res_blocks=2,   # 残差块数量
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=32,
        attention_resolutions="32,16,8",  # 注意力分辨率
        channel_mult="",  # 会根据image_size自动设置
        dropout=0.1,
        class_cond=True,  # 使用类别条件
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_fp16=False,
        use_new_attention_order=True,
        
        # 扩散参数
        diffusion_steps=1000,
        noise_schedule="cosine",  # 对小数据集更友好
        learn_sigma=False,  # 简化：不学习方差
        sigma_small=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        
        # VAE特定参数
        use_vae=True,
        vae_channels=4,  # VAE latent通道数
    )


def create_model_and_diffusion_vae(
    image_size,
    class_cond,
    learn_sigma,
    num_channels,
    num_res_blocks,
    channel_mult,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    rescale_timesteps,
    rescale_learned_sigmas,
    sigma_small,
    use_checkpoint,
    use_scale_shift_norm,
    resblock_updown,
    use_fp16,
    use_new_attention_order,
    use_vae=True,
    vae_channels=4,
):
    """创建VAE版本的模型和扩散过程"""
    
    # 创建模型
    model = create_model_vae(
        image_size,
        num_channels,
        num_res_blocks,
        channel_mult=channel_mult,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        use_new_attention_order=use_new_attention_order,
        use_vae=use_vae,
        vae_channels=vae_channels,
    )
    
    # 创建扩散过程
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        sigma_small=sigma_small,
        noise_schedule=noise_schedule,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
    )
    
    return model, diffusion


def create_model_vae(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult="",
    learn_sigma=False,
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
    use_vae=True,
    vae_channels=4,
):
    """
    创建VAE版本的UNet模型
    主要改动：输入输出通道数
    """
    
    # 自动设置channel_mult
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        elif image_size == 32:
            channel_mult = (1, 2, 2, 2)
        elif image_size == 16:  # VAE latent空间 (64->16)
            channel_mult = (1, 2, 2)
        else:
            raise ValueError(f"不支持的图像尺寸: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))
    
    # 处理注意力分辨率
    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))
    
    # 确定输入输出通道
    if use_vae:
        in_channels = vae_channels  # VAE latent: 4通道
        # 输出通道：如果学习方差，通道数翻倍
        out_channels = vae_channels if not learn_sigma else vae_channels * 2
    else:
        in_channels = 3  # RGB
        out_channels = 3 if not learn_sigma else 6
    
    print(f"创建模型:")
    print(f"  - 输入通道: {in_channels}")
    print(f"  - 输出通道: {out_channels}")
    print(f"  - 图像尺寸: {image_size}")
    print(f"  - 通道倍数: {channel_mult}")
    print(f"  - 注意力分辨率: {attention_ds}")
    
    return UNetModel(
        image_size=image_size,
        in_channels=in_channels,
        model_channels=num_channels,
        out_channels=out_channels,
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
    )


def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    """创建高斯扩散过程"""
    
    # 创建beta schedule
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    
    # 确定模型预测类型
    if learn_sigma:
        model_var_type = (
            gd.ModelVarType.LEARNED_RANGE
            if not sigma_small
            else gd.ModelVarType.LEARNED
        )
    else:
        model_var_type = (
            gd.ModelVarType.FIXED_LARGE
            if not sigma_small
            else gd.ModelVarType.FIXED_SMALL
        )
    
    # 确定损失类型
    if rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    
    # 处理时间步重采样
    if timestep_respacing:
        diffusion = SpacedDiffusion(
            use_timesteps=space_timesteps(steps, timestep_respacing),
            betas=betas,
            model_mean_type=gd.ModelMeanType.EPSILON,  # 预测噪声
            model_var_type=model_var_type,
            loss_type=loss_type,
            rescale_timesteps=rescale_timesteps,
        )
    else:
        diffusion = gd.GaussianDiffusion(
            betas=betas,
            model_mean_type=gd.ModelMeanType.EPSILON,
            model_var_type=model_var_type,
            loss_type=loss_type,
            rescale_timesteps=rescale_timesteps,
        )
    
    return diffusion


def add_dict_to_argparser(parser, default_dict):
    """添加字典参数到argparser"""
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    """将args转换为字典"""
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """字符串转布尔值"""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def test_model_creation():
    """测试模型创建"""
    print("=" * 60)
    print("测试VAE模型创建")
    print("=" * 60)
    
    # 使用默认参数
    defaults = model_and_diffusion_defaults_vae()
    
    # 创建模型和扩散
    model, diffusion = create_model_and_diffusion_vae(**defaults)
    
    print(f"\n✓ 模型创建成功")
    print(f"  - 参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"  - 扩散步数: {diffusion.num_timesteps}")
    
    # 测试前向传播（VAE latent输入）
    batch_size = 2
    latent_size = 16  # 64x64图像 -> 16x16 latent
    x = torch.randn(batch_size, 4, latent_size, latent_size)  # 4通道latent
    t = torch.randint(0, 1000, (batch_size,))
    y = torch.randint(0, NUM_CLASSES, (batch_size,))
    
    print(f"\n测试前向传播:")
    print(f"  - 输入: {x.shape}")
    
    with torch.no_grad():
        output = model(x, t, y=y)
    
    print(f"  - 输出: {output.shape}")
    print(f"✓ 前向传播成功")
    
    print("\n" + "=" * 60)
    print("VAE模型测试完成")
    print("=" * 60)


if __name__ == "__main__":
    import torch
    test_model_creation()

