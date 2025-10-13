"""
基于官方image_sample.py的VAE版本采样脚本
主要修改：
1. 生成latent而不是图像
2. 使用VAE解码到图像
3. 保存解码后的图像
"""

import argparse
import os
import sys

import numpy as np
import torch as th
import torch.distributed as dist
from PIL import Image

# 添加父目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# 尝试不同的导入方式
try:
    from guided_diffusion_vae import dist_util, logger
    from guided_diffusion_vae.script_util_vae import (
        NUM_CLASSES,
        model_and_diffusion_defaults_vae,
        create_model_and_diffusion_vae,
        add_dict_to_argparser,
        args_to_dict,
    )
    from guided_diffusion_vae.vae_wrapper import VAEInterface
except ImportError:
    # 如果上面失败，尝试直接导入（对于Kaggle环境）
    try:
        sys.path.insert(0, parent_dir)
        import dist_util, logger
        from script_util_vae import (
            NUM_CLASSES,
            model_and_diffusion_defaults_vae,
            create_model_and_diffusion_vae,
            add_dict_to_argparser,
            args_to_dict,
        )
        from vae_wrapper import VAEInterface
    except ImportError:
        # 最后尝试从当前目录的父目录导入
        sys.path.insert(0, os.path.dirname(current_dir))
        import dist_util, logger
        from script_util_vae import (
            NUM_CLASSES,
            model_and_diffusion_defaults_vae,
            create_model_and_diffusion_vae,
            add_dict_to_argparser,
            args_to_dict,
        )
        from vae_wrapper import VAEInterface


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    
    # 配置日志
    sample_dir = args.sample_dir or f"samples/vae_diffusion_{args.image_size}"
    os.makedirs(sample_dir, exist_ok=True)
    logger.configure(dir=sample_dir)

    logger.log("创建VAE接口...")
    # 创建VAE接口
    device = dist_util.dev()
    vae_interface = VAEInterface(
        vae_path=args.vae_model_path,
        device=device
    )
    
    # 计算latent尺寸
    latent_size = vae_interface.get_latent_size(args.image_size)
    logger.log(f"图像尺寸: {args.image_size}x{args.image_size}")
    logger.log(f"Latent尺寸: {latent_size}x{latent_size}")

    logger.log("创建模型和扩散过程...")
    # 重要：使用latent尺寸而不是图像尺寸
    model_args = args_to_dict(args, model_and_diffusion_defaults_vae().keys())
    model_args['image_size'] = latent_size  # 覆盖为latent尺寸
    
    model, diffusion = create_model_and_diffusion_vae(**model_args)
    
    # 加载模型权重
    logger.log(f"加载模型权重: {args.model_path}")
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("开始采样...")
    all_images = []
    all_labels = []
    
    # 采样循环
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        
        # 添加类别条件
        if args.class_cond:
            if args.fixed_class >= 0:
                # 使用固定类别
                classes = th.ones(args.batch_size, dtype=th.long, device=dist_util.dev()) * args.fixed_class
            else:
                # 随机类别
                classes = th.randint(
                    low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
                )
            model_kwargs["y"] = classes
        
        # 选择采样方法
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        
        # 生成latent样本
        # 注意：这里的shape是latent空间的
        logger.log(f"生成 {args.batch_size} 个样本...")
        sample = sample_fn(
            model,
            (args.batch_size, 4, latent_size, latent_size),  # 4通道latent
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            progress=True,
        )
        
        # 从标准化空间恢复到VAE尺度
        # 训练时我们除以了0.18215，现在要乘回来
        sample = sample * 0.18215
        
        logger.log("解码latent到图像...")
        # 使用VAE解码
        with th.no_grad():
            images = vae_interface.decode_batch(sample)
        
        # 转换到uint8
        images = (images * 255).clamp(0, 255).to(th.uint8)
        images = images.permute(0, 2, 3, 1).contiguous()
        
        # 收集样本（分布式训练时）
        gathered_samples = [th.zeros_like(images) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, images)
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        
        # 收集标签
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        
        logger.log(f"已创建 {len(all_images) * args.batch_size} 个样本")

    # 截取到指定数量
    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]

    # 保存样本
    if dist.get_rank() == 0:
        # 保存为单独的图像文件
        if args.save_images_separately:
            logger.log("保存单独的图像文件...")
            img_dir = os.path.join(sample_dir, "images")
            os.makedirs(img_dir, exist_ok=True)
            
            for i in range(len(arr)):
                img = Image.fromarray(arr[i])
                if args.class_cond:
                    filename = f"sample_{i:06d}_class_{label_arr[i]}.png"
                else:
                    filename = f"sample_{i:06d}.png"
                img.save(os.path.join(img_dir, filename))
        
        # 保存为npz文件
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(sample_dir, f"samples_{shape_str}.npz")
        logger.log(f"保存到 {out_path}")
        
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)
        
        # 保存一个样本网格用于快速预览
        if args.save_grid:
            logger.log("创建样本网格...")
            create_sample_grid(arr[:min(64, len(arr))], os.path.join(sample_dir, "sample_grid.png"))

    dist.barrier()
    logger.log("采样完成！")


def create_sample_grid(images, save_path, grid_size=8):
    """创建图像网格用于可视化"""
    from torchvision.utils import make_grid
    
    # 转换为tensor
    if isinstance(images, np.ndarray):
        images = th.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
    
    # 创建网格
    grid = make_grid(images, nrow=grid_size, padding=2, pad_value=1)
    
    # 转换为PIL图像并保存
    grid = grid.permute(1, 2, 0).cpu().numpy()
    grid = (grid * 255).astype(np.uint8)
    Image.fromarray(grid).save(save_path)


def create_argparser():
    defaults = dict(
        # 采样参数
        clip_denoised=True,
        num_samples=100,
        batch_size=16,
        use_ddim=True,  # 默认使用DDIM快速采样
        ddim_steps=50,
        
        # 模型参数
        model_path="",
        
        # VAE参数
        vae_model_path="domain_adaptive_diffusion/vae/vae_model.pt",
        
        # 类别条件
        fixed_class=-1,  # -1表示随机，>=0表示固定类别
        
        # 保存选项
        sample_dir="",
        save_images_separately=True,  # 是否保存单独的图像文件
        save_grid=True,  # 是否保存样本网格
    )
    
    defaults.update(model_and_diffusion_defaults_vae())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
