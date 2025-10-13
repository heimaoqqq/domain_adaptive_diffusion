"""
基于官方image_train.py的VAE版本训练脚本
主要修改：
1. 使用VAE数据加载器
2. 使用4通道模型
3. 添加VAE相关参数
"""

import argparse
import os
import sys

# 添加父目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# 尝试不同的导入方式
try:
    from guided_diffusion_vae import dist_util, logger
    from guided_diffusion_vae.image_datasets_vae import load_data_vae
    from guided_diffusion_vae.resample import create_named_schedule_sampler
    from guided_diffusion_vae.script_util_vae import (
        model_and_diffusion_defaults_vae,
        create_model_and_diffusion_vae,
        args_to_dict,
        add_dict_to_argparser,
    )
    from guided_diffusion_vae.train_util import TrainLoop
    from guided_diffusion_vae.vae_wrapper import VAEInterface
except ImportError:
    # 如果上面失败，尝试直接导入（对于Kaggle环境）
    try:
        sys.path.insert(0, parent_dir)
        import dist_util, logger
        from image_datasets_vae import load_data_vae
        from resample import create_named_schedule_sampler
        from script_util_vae import (
            model_and_diffusion_defaults_vae,
            create_model_and_diffusion_vae,
            args_to_dict,
            add_dict_to_argparser,
        )
        from train_util import TrainLoop
        from vae_wrapper import VAEInterface
    except ImportError:
        # 最后尝试从当前目录的父目录导入
        sys.path.insert(0, os.path.dirname(current_dir))
        import dist_util, logger
        from image_datasets_vae import load_data_vae
        from resample import create_named_schedule_sampler
        from script_util_vae import (
            model_and_diffusion_defaults_vae,
            create_model_and_diffusion_vae,
            args_to_dict,
            add_dict_to_argparser,
        )
        from train_util import TrainLoop
        from vae_wrapper import VAEInterface


def main():
    args = create_argparser().parse_args()

    # 设置分布式训练（单GPU时也能工作）
    dist_util.setup_dist()
    
    # 配置日志
    log_dir = args.log_dir or f"logs/vae_diffusion_{args.image_size}"
    os.makedirs(log_dir, exist_ok=True)
    logger.configure(dir=log_dir)

    logger.log("创建VAE接口...")
    # 创建VAE接口
    device = dist_util.dev()
    vae_interface = VAEInterface(
        vae_path=args.vae_model_path,
        device=device
    )
    
    # 测试VAE
    if args.test_vae:
        logger.log("测试VAE编码解码...")
        success = vae_interface.test_roundtrip(args.image_size)
        if not success:
            logger.log("警告: VAE往返测试误差较大")

    logger.log("创建模型和扩散过程...")
    # 创建模型和扩散过程
    model, diffusion = create_model_and_diffusion_vae(
        **args_to_dict(args, model_and_diffusion_defaults_vae().keys())
    )
    model.to(dist_util.dev())
    
    # 创建采样调度器
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("创建数据加载器...")
    # 计算latent尺寸
    latent_size = vae_interface.get_latent_size(args.image_size)
    logger.log(f"图像尺寸: {args.image_size}x{args.image_size}")
    logger.log(f"Latent尺寸: {latent_size}x{latent_size}")
    
    # 创建数据加载器
    data = load_data_vae(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        vae_interface=vae_interface,
        use_cached_latents=args.use_cached_latents,
        cached_latents_dir=args.cached_latents_dir,
        class_cond=args.class_cond,
        random_flip=args.random_flip,
    )

    logger.log("开始训练...")
    logger.log(f"参数:")
    logger.log(f"  - 批大小: {args.batch_size}")
    logger.log(f"  - 微批大小: {args.microbatch}")
    logger.log(f"  - 学习率: {args.lr}")
    logger.log(f"  - EMA率: {args.ema_rate}")
    logger.log(f"  - 使用FP16: {args.use_fp16}")
    logger.log(f"  - 权重衰减: {args.weight_decay}")
    
    # 创建训练循环并运行
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        # 数据参数
        data_dir="",
        use_cached_latents=False,  # 默认不使用预编码（实时编码更灵活）
        cached_latents_dir="",     # 预编码latents目录
        random_flip=True,          # 数据增强
        
        # VAE参数
        vae_model_path="domain_adaptive_diffusion/vae/vae_model.pt",
        test_vae=True,  # 是否测试VAE
        
        # 训练参数
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=16,
        microbatch=-1,  # -1表示不使用微批
        ema_rate="0.9999",  # 逗号分隔的EMA值列表
        log_interval=10,
        save_interval=1000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        
        # 日志
        log_dir="",
    )
    
    # 添加模型和扩散默认参数
    defaults.update(model_and_diffusion_defaults_vae())
    
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
