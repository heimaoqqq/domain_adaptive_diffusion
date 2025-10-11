"""
深入诊断类别嵌入问题
"""
import torch
from diffusers import UNet2DConditionModel
import numpy as np

def diagnose_class_embedding():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}\n")
    
    # 创建UNet
    print("=" * 60)
    print("1. 检查UNet的类别嵌入层")
    print("=" * 60)
    
    unet = UNet2DConditionModel(
        sample_size=32,
        in_channels=4,
        out_channels=4,
        down_block_types=("DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D"),
        block_out_channels=(128, 256),
        layers_per_block=2,
        num_class_embeds=32,
        class_embed_type="timestep",
    ).to(device)
    
    # 检查是否有class_embedding层
    print("\n检查UNet的模块:")
    has_class_embedding = False
    for name, module in unet.named_modules():
        if 'class_embedding' in name.lower() or 'class_emb' in name.lower():
            print(f"  找到类别嵌入层: {name}")
            if hasattr(module, 'weight'):
                weight = module.weight
                print(f"    形状: {weight.shape}")
                print(f"    均值: {weight.mean().item():.6f}")
                print(f"    标准差: {weight.std().item():.6f}")
                print(f"    范围: [{weight.min().item():.6f}, {weight.max().item():.6f}]")
            has_class_embedding = True
    
    if not has_class_embedding:
        print("  ⚠️ 未找到显式的class_embedding层!")
    
    # 测试类别嵌入的实际影响
    print("\n" + "=" * 60)
    print("2. 测试类别嵌入的影响")
    print("=" * 60)
    
    # 准备输入
    batch_size = 8
    x = torch.randn(batch_size, 4, 32, 32).to(device)
    t = torch.tensor([500] * batch_size).to(device)
    dummy_encoder = torch.zeros(batch_size, 1, 1280).to(device)
    
    # 测试1：所有相同类别
    print("\n测试1: 所有样本使用相同类别")
    labels_same = torch.tensor([5] * batch_size).to(device)
    with torch.no_grad():
        out_same = unet(x, t, class_labels=labels_same,
                       encoder_hidden_states=dummy_encoder,
                       return_dict=False)[0]
    
    # 计算样本间差异
    diffs_same = []
    for i in range(batch_size - 1):
        diff = (out_same[i] - out_same[i+1]).abs().mean().item()
        diffs_same.append(diff)
    mean_diff_same = np.mean(diffs_same)
    print(f"  相同类别的样本间平均差异: {mean_diff_same:.4f}")
    
    # 测试2：所有不同类别
    print("\n测试2: 每个样本使用不同类别")
    labels_diff = torch.arange(batch_size).to(device)
    with torch.no_grad():
        out_diff = unet(x, t, class_labels=labels_diff,
                       encoder_hidden_states=dummy_encoder,
                       return_dict=False)[0]
    
    # 计算样本间差异
    diffs_diff = []
    for i in range(batch_size - 1):
        diff = (out_diff[i] - out_diff[i+1]).abs().mean().item()
        diffs_diff.append(diff)
    mean_diff_diff = np.mean(diffs_diff)
    print(f"  不同类别的样本间平均差异: {mean_diff_diff:.4f}")
    
    # 测试3：使用None vs 使用类别
    print("\n测试3: 无条件 vs 有条件")
    with torch.no_grad():
        # 有条件
        out_with_class = unet(x[:1], t[:1], 
                             class_labels=torch.tensor([0]).to(device),
                             encoder_hidden_states=dummy_encoder[:1],
                             return_dict=False)[0]
        # 无条件
        out_no_class = unet(x[:1], t[:1], 
                           class_labels=None,
                           encoder_hidden_states=dummy_encoder[:1],
                           return_dict=False)[0]
    
    diff_cond = (out_with_class - out_no_class).abs().mean().item()
    print(f"  有条件 vs 无条件差异: {diff_cond:.4f}")
    
    # 测试4：极端类别值
    print("\n测试4: 类别0 vs 类别31")
    with torch.no_grad():
        out_class0 = unet(x[:1], t[:1], 
                         class_labels=torch.tensor([0]).to(device),
                         encoder_hidden_states=dummy_encoder[:1],
                         return_dict=False)[0]
        out_class31 = unet(x[:1], t[:1], 
                          class_labels=torch.tensor([31]).to(device),
                          encoder_hidden_states=dummy_encoder[:1],
                          return_dict=False)[0]
    
    diff_extreme = (out_class0 - out_class31).abs().mean().item()
    print(f"  类别0 vs 类别31差异: {diff_extreme:.4f}")
    
    # 诊断结果
    print("\n" + "=" * 60)
    print("3. 诊断结果")
    print("=" * 60)
    
    if mean_diff_diff > mean_diff_same * 1.5:
        print("✅ 类别条件有一定影响")
    else:
        print("❌ 类别条件几乎没有影响")
    
    if diff_cond > 0.01:
        print("✅ 有/无条件存在差异")
    else:
        print("❌ 有/无条件几乎相同")
    
    if diff_extreme > 0.01:
        print("✅ 不同类别ID产生不同输出")
    else:
        print("❌ 不同类别ID输出几乎相同")
    
    # 测试时间步嵌入是否正常工作
    print("\n" + "=" * 60)
    print("4. 对比：时间步的影响")
    print("=" * 60)
    
    t1 = torch.tensor([100]).to(device)
    t2 = torch.tensor([900]).to(device)
    
    with torch.no_grad():
        out_t1 = unet(x[:1], t1, class_labels=None,
                     encoder_hidden_states=dummy_encoder[:1],
                     return_dict=False)[0]
        out_t2 = unet(x[:1], t2, class_labels=None,
                     encoder_hidden_states=dummy_encoder[:1],
                     return_dict=False)[0]
    
    diff_time = (out_t1 - out_t2).abs().mean().item()
    print(f"  不同时间步(100 vs 900)的差异: {diff_time:.4f}")
    
    if diff_time > diff_extreme * 10:
        print("\n⚠️ 时间步影响远大于类别条件！")
        print("   这说明类别嵌入可能没有被正确整合到时间步嵌入中。")

if __name__ == "__main__":
    diagnose_class_embedding()
