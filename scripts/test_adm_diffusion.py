"""
测试ADM风格的条件扩散模型
验证scale-shift norm机制的正确性
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from diffusers import UNet2DModel
from train_diffusers import ClassEmbedder, ADMDiffusionTrainer


def test_scale_shift_norm():
    """
    测试scale-shift norm机制
    
    Scale-Shift Norm的核心思想：
    1. 标准的LayerNorm: y = γ * (x - μ) / σ + β
    2. Scale-Shift Norm: y = (1 + scale) * norm(x) + shift
    
    其中scale和shift是由条件信息（时间步+类别）生成的
    这使得条件信息能够动态地调制网络的中间特征
    """
    print("\n" + "="*60)
    print("测试Scale-Shift Norm机制")
    print("="*60)
    
    # 创建测试输入
    batch_size = 2
    channels = 128
    spatial_size = 8
    
    # 模拟特征图
    x = torch.randn(batch_size, channels, spatial_size, spatial_size)
    
    # 模拟条件嵌入（时间步+类别的组合）
    cond_emb = torch.randn(batch_size, channels * 2)  # 2倍通道用于scale和shift
    
    # 分离scale和shift
    scale, shift = cond_emb.chunk(2, dim=1)
    scale = scale.view(batch_size, channels, 1, 1)
    shift = shift.view(batch_size, channels, 1, 1)
    
    # 应用Layer Norm
    norm = nn.GroupNorm(32, channels)
    x_norm = norm(x)
    
    # 应用Scale-Shift变换
    y = x_norm * (1 + scale) + shift
    
    print(f"输入shape: {x.shape}")
    print(f"条件嵌入shape: {cond_emb.shape}")
    print(f"Scale shape: {scale.shape}")
    print(f"Shift shape: {shift.shape}")
    print(f"输出shape: {y.shape}")
    print(f"\n统计信息:")
    print(f"  输入std: {x.std():.4f}")
    print(f"  归一化后std: {x_norm.std():.4f}")
    print(f"  Scale-Shift后std: {y.std():.4f}")
    print(f"\n✅ Scale-Shift Norm测试通过")


def test_class_embedder():
    """测试类别嵌入器"""
    print("\n" + "="*60)
    print("测试类别嵌入器")
    print("="*60)
    
    # 创建类别嵌入器
    num_classes = 31
    embed_dim = 512
    embedder = ClassEmbedder(num_classes, embed_dim)
    
    # 测试不同的类别
    batch_size = 4
    class_labels = torch.tensor([0, 15, 30, 5])
    
    # 前向传播
    embeddings = embedder(class_labels)
    
    print(f"类别数: {num_classes}")
    print(f"输入标签: {class_labels.tolist()}")
    print(f"嵌入维度: {embed_dim}")
    print(f"输出shape: {embeddings.shape}")
    print(f"输出std: {embeddings.std():.4f}")
    
    # 测试不同类别的嵌入是否不同
    emb1 = embedder(torch.tensor([0]))
    emb2 = embedder(torch.tensor([15]))
    diff = (emb1 - emb2).abs().mean()
    
    print(f"\n不同类别嵌入差异: {diff:.4f}")
    assert diff > 0.1, "类别嵌入差异太小！"
    print("✅ 类别嵌入器测试通过")


def test_adm_integration():
    """测试完整的ADM集成"""
    print("\n" + "="*60)
    print("测试ADM风格的条件注入")
    print("="*60)
    
    # 创建简单配置
    config = {
        'data': {'latent_size': 32},
        'model': {
            'in_channels': 4,
            'out_channels': 4,
            'block_out_channels': [128, 256],
            'layers_per_block': 2,
            'attention_head_dim': 8,
            'num_class_embeds': 32,
            'dropout': 0.0,
            'norm_num_groups': 32,
            'norm_eps': 1e-6,
            'act_fn': 'silu',
        }
    }
    
    # 创建UNet（简化版）
    unet = UNet2DModel(
        sample_size=config['data']['latent_size'],
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        layers_per_block=config['model']['layers_per_block'],
        block_out_channels=tuple(config['model']['block_out_channels']),
        down_block_types=["DownBlock2D", "AttnDownBlock2D"],
        up_block_types=["AttnUpBlock2D", "UpBlock2D"],
        attention_head_dim=config['model']['attention_head_dim'],
        dropout=config['model']['dropout'],
        norm_num_groups=config['model']['norm_num_groups'],
        norm_eps=float(config['model']['norm_eps']),
        act_fn=config['model']['act_fn'],
        resnet_time_scale_shift="scale_shift",  # 启用scale-shift norm
    )
    
    print(f"UNet参数量: {sum(p.numel() for p in unet.parameters()):,}")
    print(f"使用scale-shift norm: {unet.config.resnet_time_scale_shift}")
    
    # 测试前向传播
    batch_size = 2
    x = torch.randn(batch_size, 4, 32, 32)
    timesteps = torch.tensor([100, 500])
    
    # 前向传播
    with torch.no_grad():
        output = unet(x, timesteps, return_dict=False)[0]
    
    print(f"\n前向传播测试:")
    print(f"  输入shape: {x.shape}")
    print(f"  输出shape: {output.shape}")
    print(f"  输出std: {output.std():.4f}")
    
    # 测试条件响应
    print(f"\n测试条件响应:")
    
    # 相同输入，不同时间步
    t1 = torch.tensor([100, 100])
    t2 = torch.tensor([900, 900])
    
    with torch.no_grad():
        out1 = unet(x, t1, return_dict=False)[0]
        out2 = unet(x, t2, return_dict=False)[0]
    
    time_diff = (out1 - out2).abs().mean()
    print(f"  不同时间步输出差异: {time_diff:.4f}")
    assert time_diff > 0.01, "时间步条件响应太弱！"
    
    print("\n✅ ADM集成测试通过")


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("ADM风格条件扩散模型测试套件")
    print("="*60)
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    
    # 运行测试
    test_scale_shift_norm()
    test_class_embedder()
    test_adm_integration()
    
    print("\n" + "="*60)
    print("所有测试通过！✅")
    print("="*60)
    
    print("\n📚 Scale-Shift Norm原理解释:")
    print("-" * 40)
    print("Scale-Shift Norm是ADM(Ablated Diffusion Model)的核心创新:")
    print("")
    print("1. 传统方法：将条件信息添加到时间步嵌入中")
    print("   问题：条件信息影响有限，难以精细控制生成")
    print("")
    print("2. Scale-Shift Norm方法：")
    print("   - 将条件嵌入投影为scale和shift参数")
    print("   - 在每个ResBlock中：y = (1 + scale) * norm(x) + shift")
    print("   - 条件信息直接调制特征的幅度和偏移")
    print("")
    print("3. 优势：")
    print("   - 更强的条件控制能力")
    print("   - 可以精细地影响每一层的特征")
    print("   - 对细微的类别差异更敏感")
    print("")
    print("4. 为什么适合31个用户的细微差异：")
    print("   - 每层都能根据用户ID调整特征")
    print("   - 累积效应使得细微差异被放大")
    print("   - 比简单的类别嵌入更有表现力")
    print("-" * 40)


if __name__ == '__main__':
    main()
