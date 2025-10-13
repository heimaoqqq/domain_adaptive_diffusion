# VAE集成到Guided-Diffusion的详细计划

## 📋 集成步骤概览

### ✅ 已完成
1. **复制官方代码** - 所有核心文件已复制到 `guided_diffusion_vae/`
2. **创建配置文件** - `configs/microdoppler_small.yaml`
3. **分析差异** - 明确了官方代码不使用VAE，直接在像素空间训练

### 🔧 待实施步骤

## Step 1: 创建VAE包装器
创建 `vae_wrapper.py`，提供统一的VAE接口：

```python
class VAEInterface:
    def __init__(self, vae_path):
        # 加载VAE模型
        self.vae = load_vae(vae_path)
        self.scale_factor = 0.18215
        
    def encode_batch(self, images):
        """编码图像批次到latent"""
        # images: [B, 3, 64, 64] in [0, 1]
        with torch.no_grad():
            latents = self.vae.encode(images)
            return latents * self.scale_factor
            
    def decode_batch(self, latents):
        """解码latent批次到图像"""
        # latents: [B, 4, 16, 16] (对于64x64图像)
        with torch.no_grad():
            latents = latents / self.scale_factor
            images = self.vae.decode(latents)
            return torch.clamp(images, 0, 1)
```

## Step 2: 修改数据加载器
修改 `image_datasets.py`:

### 原始代码（像素空间）:
```python
def __getitem__(self, idx):
    # 加载图像
    pil_image = Image.open(path)
    arr = preprocess(pil_image)
    # 归一化到[-1, 1]
    arr = arr.astype(np.float32) / 127.5 - 1
    return arr, label
```

### 修改后（VAE latent空间）:
```python
def __getitem__(self, idx):
    # 方案A: 实时编码（灵活但慢）
    pil_image = Image.open(path)
    arr = preprocess(pil_image)
    # 归一化到[0, 1] for VAE
    arr = arr.astype(np.float32) / 255.0
    
    # 编码到latent
    with torch.no_grad():
        tensor = torch.from_numpy(arr).unsqueeze(0)
        latent = self.vae.encode_batch(tensor)
        # 标准化到std≈1.0用于扩散
        latent = latent / 0.18215
    
    return latent.squeeze(0).numpy(), label
    
    # 方案B: 预编码（快但需要存储）
    # 直接加载预编码的.npy文件
    latent = np.load(latent_path)
    return latent / 0.18215, label  # 标准化
```

## Step 3: 修改模型配置
修改 `script_util.py` 中的 `create_model`:

### 原始:
```python
return UNetModel(
    in_channels=3,      # RGB
    out_channels=3,     # RGB
    ...
)
```

### 修改后:
```python
return UNetModel(
    in_channels=4,      # VAE latent
    out_channels=4,     # VAE latent  
    ...
)
```

## Step 4: 修改训练脚本
修改 `scripts/image_train_vae.py`:

```python
def main():
    # 初始化VAE
    vae = VAEInterface(args.vae_path)
    
    # 创建数据加载器（带VAE）
    data = load_data_with_vae(
        vae=vae,
        data_dir=args.data_dir,
        ...
    )
    
    # 创建模型（4通道）
    model, diffusion = create_model_and_diffusion_vae(
        ...
    )
    
    # 训练循环保持不变
    TrainLoop(...).run_loop()
```

## Step 5: 修改采样脚本  
修改 `scripts/image_sample_vae.py`:

```python
def sample():
    # 生成latent
    sample = diffusion.p_sample_loop(
        model,
        (batch_size, 4, 16, 16),  # latent shape
        ...
    )
    
    # 标准化回VAE空间
    sample = sample * 0.18215
    
    # 解码到图像
    images = vae.decode_batch(sample)
    
    # 保存图像
    save_images(images)
```

## Step 6: 关键修改点总结

### 1. **尺度匹配**
- 训练时：latent / 0.18215 → std≈1.0
- 采样时：sample * 0.18215 → VAE scale

### 2. **输出层初始化**
保持官方的 `zero_module` 初始化：
```python
self.out = nn.Sequential(
    normalization(ch),
    nn.SiLU(),
    zero_module(conv_nd(dims, 4, 4, 3, padding=1)),  # 4通道
)
```

### 3. **图像尺寸映射**
- 64x64 图像 → 16x16 latent (下采样4x)
- 128x128 图像 → 32x32 latent

## 📊 对比表

| 方面 | 官方Guided-Diffusion | VAE集成版本 |
|-----|-------------------|-----------|
| 输入通道 | 3 (RGB) | 4 (latent) |
| 输出通道 | 3 (RGB) | 4 (latent) |
| 数据范围 | [-1, 1] | std≈1.0 |
| 空间尺寸 | 64x64 | 16x16 |
| 输出初始化 | zero_module | zero_module |
| 内存占用 | 高 | 低（16x压缩）|
| 训练速度 | 慢 | 快（小尺寸）|

## ⚠️ 注意事项

1. **保持官方的核心逻辑不变**
   - 扩散过程
   - 损失计算
   - 采样算法

2. **只在接口处修改**
   - 数据输入
   - 模型通道
   - 结果输出

3. **调试建议**
   - 先用小batch测试
   - 检查每步的tensor形状
   - 验证缩放是否正确

## 🚀 实施顺序

1. **Phase 1**: 创建VAE接口（vae_wrapper.py）
2. **Phase 2**: 测试VAE编码/解码 
3. **Phase 3**: 修改数据加载器
4. **Phase 4**: 修改模型创建
5. **Phase 5**: 创建训练脚本
6. **Phase 6**: 测试训练流程
7. **Phase 7**: 创建采样脚本
8. **Phase 8**: 完整测试

## 📝 验证检查点

- [ ] VAE能正确加载
- [ ] 编码/解码往返测试通过
- [ ] 数据加载器返回正确形状
- [ ] 模型前向传播正常
- [ ] 损失计算正确
- [ ] 采样生成有意义的图像

