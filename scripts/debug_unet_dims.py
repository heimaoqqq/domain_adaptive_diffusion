"""
快速测试UNet的维度需求
"""
import torch
from diffusers import UNet2DConditionModel

def test_unet_dims():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建不同配置的UNet
    configs = [
        {
            "name": "无CrossAttention",
            "params": {
                "sample_size": 32,
                "in_channels": 4,
                "out_channels": 4,
                "down_block_types": ("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
                "up_block_types": ("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
                "block_out_channels": (128, 256, 512, 512),
                "layers_per_block": 2,
                "num_class_embeds": 31,
                "class_embed_type": "timestep",
                "add_attention": False,  # 尝试完全禁用attention
            }
        },
        {
            "name": "带CrossAttention(1280维)",
            "params": {
                "sample_size": 32,
                "in_channels": 4,
                "out_channels": 4,
                "down_block_types": ("DownBlock2D", "DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
                "up_block_types": ("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
                "block_out_channels": (128, 256, 512, 512),
                "layers_per_block": 2,
                "num_class_embeds": 31,
                "class_embed_type": "timestep",
                "cross_attention_dim": 1280,  # 标准SD维度
            }
        }
    ]
    
    for config in configs:
        print(f"\n测试配置: {config['name']}")
        print("-" * 50)
        
        try:
            unet = UNet2DConditionModel(**config['params']).to(device)
            
            # 测试输入
            x = torch.randn(1, 4, 32, 32).to(device)
            t = torch.tensor([500]).to(device)
            label = torch.tensor([0]).to(device)
            
            # 检查是否需要encoder_hidden_states
            if 'cross_attention_dim' in config['params']:
                cross_dim = config['params']['cross_attention_dim']
                encoder_hidden_states = torch.zeros(1, 1, cross_dim).to(device)
                print(f"  需要encoder_hidden_states，维度: {cross_dim}")
                
                # 测试前向传播
                out = unet(x, t, class_labels=label, 
                          encoder_hidden_states=encoder_hidden_states,
                          return_dict=False)[0]
            else:
                print(f"  不需要encoder_hidden_states")
                # 测试是否真的不需要
                try:
                    out = unet(x, t, class_labels=label, return_dict=False)[0]
                    print(f"  ✓ 成功！不需要encoder_hidden_states")
                except Exception as e:
                    print(f"  ✗ 失败：{type(e).__name__}: {str(e)[:100]}")
                    
                    # 尝试提供dummy encoder_hidden_states
                    for dim in [128, 768, 1280]:
                        try:
                            dummy = torch.zeros(1, 1, dim).to(device)
                            out = unet(x, t, class_labels=label, 
                                      encoder_hidden_states=dummy,
                                      return_dict=False)[0]
                            print(f"  ✓ 使用{dim}维dummy encoder_hidden_states成功")
                            break
                        except:
                            print(f"  ✗ {dim}维失败")
            
            print(f"  输出shape: {out.shape}")
            
        except Exception as e:
            print(f"  创建UNet失败: {type(e).__name__}: {str(e)}")

if __name__ == "__main__":
    test_unet_dims()
