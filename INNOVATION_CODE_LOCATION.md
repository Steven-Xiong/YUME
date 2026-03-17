# 🎯 YUME 创新点代码定位指南

本文档详细标注了 YUME 项目中 5 个核心创新点的具体代码实现位置。

---

## 1️⃣ TSCM (Temporal-Spatial-Channel Modeling) - 历史帧自适应压缩

### 📍 核心代码位置
**文件**: `wan23/modules/model.py`  
**方法**: `WanModel.forward()`  
**行号**: 486-720

### 🔧 实现原理
TSCM 根据历史帧数量动态调整时空压缩率，避免历史上下文过长导致计算爆炸。

### 📝 关键代码片段

#### **不同压缩层级的 patch_embedding**
```python
# 行号: 486-495
self.patch_embedding_2x  = upsample_conv3d_weights_auto(self.patch_embedding, (1,4,4))
self.patch_embedding_4x  = upsample_conv3d_weights_auto(self.patch_embedding, (1,8,8))
self.patch_embedding_8x  = upsample_conv3d_weights_auto(self.patch_embedding, (1,16,16))
self.patch_embedding_16x = upsample_conv3d_weights_auto(self.patch_embedding, (1,32,32))
self.patch_embedding_2x_f = nn.Conv3d(
    self.patch_embedding.in_channels,
    self.patch_embedding.in_channels,
    kernel_size=(1,4,4), stride=(1,4,4),
)
```

#### **层级 1: ≤ 2+4 帧** (最少压缩)
```python
# 行号: 599-615
if f_num - latent_frame_zero <= 2 + 4:
    f_zero = u1.shape[2]
    u_1 = self.patch_embedding(u1[:,:,0].unsqueeze(2))  # 第1帧: 无压缩
    
    if f_zero - 2 <= 0:
        u_2 = self.patch_embedding_2x(convpadd(u1[:,:,-1].unsqueeze(2),4))  # 2x压缩
    else:
        u_2 = self.patch_embedding_2x(convpadd(u1[:,:,1:-1],4))  # 中间帧: 2x
    
    u_3 = self.patch_embedding(u1[:,:,-1].unsqueeze(2))  # 最后1帧: 无压缩
    
    # 生成对应的 RoPE 编码
    freqs_i = torch.cat([
        up_fre(f_1,f_2,f_3,u_1,0), 
        up_fre(f_1,f_2,f_3,u_2,f1,True), 
        up_fre(f_1,f_2,f_3,u_3,f1+f2)
    ], dim=0)
```

#### **层级 2: ≤ 2+4+16 帧**
```python
# 行号: 618-637
elif f_num - latent_frame_zero <= 2 + 4 + 16:
    u_1 = self.patch_embedding(u1[:,:,0].unsqueeze(2))       # 第1帧: 无压缩
    u_2 = self.patch_embedding_4x(convpadd(u1[:,:,1:-5],8))  # 老帧: 4x压缩
    u_3 = self.patch_embedding_2x(convpadd(u1[:,:,-5:-3],4)) # 近帧: 2x压缩
    u_4 = self.patch_embedding(u1[:,:,-3:])                  # 最近3帧: 无压缩
```

#### **层级 3: ≤ 2+4+16+64 帧**
```python
# 行号: 640-661
elif f_num - latent_frame_zero <= 2 + 4 + 16 + 64:
    u_1 = self.patch_embedding(u1[:,:,0].unsqueeze(2))        # 1x
    u_2 = self.patch_embedding_8x(convpadd(u1[:,:,1:-21],16)) # 8x
    u_3 = self.patch_embedding_4x(convpadd(u1[:,:,-21:-5],8)) # 4x
    u_4 = self.patch_embedding_2x(convpadd(u1[:,:,-5:-3],4))  # 2x
    u_5 = self.patch_embedding(u1[:,:,-3:])                   # 1x
```

#### **层级 4: ≤ 2+4+16+64+256 帧**
```python
# 行号: 664-688
elif f_num - latent_frame_zero <= 2 + 4 + 16 + 64 + 256:
    u_1 = self.patch_embedding_2x(...)      # 首帧: 2x
    u_2 = self.patch_embedding_16x(...)     # 远古帧: 16x
    u_3 = self.patch_embedding_8x(...)      # 较老帧: 8x
    u_4 = self.patch_embedding_4x(...)      # 老帧: 4x
    u_5 = self.patch_embedding_2x(...)      # 近帧: 2x
    u_6 = self.patch_embedding(...)         # 最新帧: 1x
```

#### **层级 5: ≤ 2+4+16+64+256+1024 帧** (最大压缩)
```python
# 行号: 690-718
elif f_num - latent_frame_zero <= 2 + 4 + 16 + 64 + 256 + 1024:
    u_1 = self.patch_embedding_2x(...)                    # 2x
    u_2 = self.patch_embedding_16x(                       # 超远古帧: 16x + 2x
            convpadd(self.patch_embedding_2x_f(...), 32))
    u_3 = self.patch_embedding_16x(...)                   # 16x
    u_4 = self.patch_embedding_8x(...)                    # 8x
    u_5 = self.patch_embedding_4x(...)                    # 4x
    u_6 = self.patch_embedding_2x(...)                    # 2x
    u_7 = self.patch_embedding(...)                       # 1x (最新帧)
```

### 📊 TSCM 压缩策略总结

| 历史帧数量 | 压缩层级 | 压缩比率分布 | 对应论文 |
|------------|---------|-------------|---------|
| ≤ 6 帧 | 3层 | 1x, 2x, 1x | 初始阶段 |
| ≤ 22 帧 | 4层 | 1x, 4x, 2x, 1x | - |
| ≤ 86 帧 | 5层 | 1x, 8x, 4x, 2x, 1x | - |
| ≤ 342 帧 | 6层 | 2x, 16x, 8x, 4x, 2x, 1x | - |
| > 342 帧 | 7层 | 2x, 32x, 16x, 8x, 4x, 2x, 1x | Eq.3 最大压缩 |

---

## 2️⃣ Self-Forcing - 使用模型自己的预测作为历史帧

### 📍 核心代码位置
**文件**: `webapp_single_gpu.py`  
**方法**: `long_generate()`  
**行号**: 836-863

### 🔧 实现原理
在长视频生成的每一段（chunk）完成后，将 **clean latent**（去噪后的干净 latent）作为下一段的历史条件，而不是使用原始视频的 latent。

### 📝 关键代码片段

#### **更新 latent 历史帧** (Self-Forcing 核心)
```python
# 行号: 836-840
if is_i2v_mode or seg > 0:
    # 保留旧的历史帧，拼接新生成的 clean latent
    model_input_latent = torch.cat([
        model_input_latent[:,:-latent_frame_zero,:,:],  # 旧历史
        latent[:,-latent_frame_zero:,:,:]                # 新生成的 clean latent ✅
    ], dim=1)
else:
    model_input_latent = latent[:,-latent_frame_zero:,:,:]
```

#### **更新 pixel 历史帧** (解码后的视频帧)
```python
# 行号: 857-861
if is_i2v_mode or seg > 0:
    # 使用 VAE 解码后的新视频帧更新历史
    model_input_de = torch.cat([
        model_input_de[:,:-frame_zero,:,:],          # 旧历史
        video_tail_px[:,-frame_zero:,:,:]            # 新解码的视频帧 ✅
    ], dim=1)
else:
    model_input_de = video_tail_px[:,-frame_zero:,:,:]
```

#### **保存到全局状态** (用于继续生成)
```python
# 行号: 872-876
LAST["last_model_input_latent"] = model_input_latent.detach()  # 保存 clean latent
LAST["last_model_input_de"] = model_input_de.detach()          # 保存解码视频
LAST["frame_total"] = frame_total
LAST["last_video_path"] = out_path
LAST["last_prompt"] = final_prompt
```

### 🔄 Self-Forcing 流程图

```
第1段生成:
  输入图片 → VAE编码 → latent_0
  扩散采样 → clean_latent_1 → VAE解码 → video_1

第2段生成: (Self-Forcing)
  [latent_0, clean_latent_1] ✅ → 作为历史条件
  扩散采样 → clean_latent_2 → VAE解码 → video_2

第3段生成:
  [latent_0, clean_latent_1, clean_latent_2] ✅
  扩散采样 → clean_latent_3 → ...
```

### 📊 与论文对应
- **论文 Section 4.2.2**: "During training, we use the model's own predictions as the autoregressive context..."
- **代码实现**: 每次循环都用 `latent[:,-latent_frame_zero:,:,:]` (模型预测的 clean latent) 更新历史

### 🎯 训练代码位置
**文件**: `fastvideo/distill_model.py`  
**行号**: 390

```python
# 行号: 390
# 使用模型自己预测的 clean latent 作为历史
latent = torch.cat([
    noise[:,:-9,:,:]*sampling_sigmas[max(i-1,0)] + 
    (1-sampling_sigmas[max(i-1,0)])*model_input[:,:-9,:,:],  # 历史部分
    temp_x0  # 新预测的 clean latent ✅
], dim=1)
```

---

## 3️⃣ Keyboard/Mouse Control - 相机控制转换为自然语言

### 📍 核心代码位置
**文件**: `fastvideo/dataset/t2v_datasets.py`  
**方法**: `StableVideoAnimationDataset.get_sample()`  
**行号**: 218-232 (解析), 393-418 (转换)

### 🔧 实现原理
从 `.txt` 文件中解析键盘（W/A/S/D）和鼠标（↑/↓/←/→）控制指令，通过预定义词汇表转换为自然语言描述，拼接到 caption 中。

### 📝 关键代码片段

#### **解析 TXT 文件**
```python
# 行号: 218-232
def parse_txt_file(txt_path):
    """Parse TXT file to extract Keys and Mouse information"""
    keys = None
    mouse = None
    try:
        with open(txt_path, 'r') as f:
            for line in f:
                if line.startswith('Keys:'):
                    keys = line.split(':', 1)[1].strip()  # 提取 "W", "W+A", etc.
                elif line.startswith('Mouse:'):
                    mouse = line.split(':', 1)[1].strip()  # 提取 "→", "↑", etc.
                if keys is not None and mouse is not None:
                    break
    except Exception as e:
        print(f"Error parsing {txt_path}: {e}")
    return keys, mouse
```

#### **键盘控制词汇表** (vocab_k)
```python
# 行号: 393-405
vocab = {
    "W": "Person moves forward (W).",          # 前进
    "A": "Person moves left (A).",             # 左移
    "S": "Person moves backward (S).",         # 后退
    "D": "Person moves right (D).",            # 右移
    "W+A": "Person moves forward and left (W+A).",   # 左前
    "W+D": "Person moves forward and right (W+D).",  # 右前
    "S+D": "Person moves backward and right (S+D).", # 右后
    "S+A": "Person moves backward and left (S+A).",  # 左后
    "None": "Person stands still (·).",        # 静止
    "·": "Person stands still (·)."
}
caption = caption + vocab[keys[0]]  # 拼接到 caption
```

#### **鼠标控制词汇表** (vocab_c)
```python
# 行号: 407-418
vocab = {
    "→": "Camera turns right (→).",                    # 右转
    "←": "Camera turns left (←).",                     # 左转
    "↑": "Camera tilts up (↑).",                       # 上仰
    "↓": "Camera tilts down (↓).",                     # 下俯
    "↑→": "Camera tilts up and turns right (↑→).",    # 右上
    "↑←": "Camera tilts up and turns left (↑←).",     # 左上
    "↓→": "Camera tilts down and turns right (↓→).",  # 右下
    "↓←": "Camera tilts down and turns left (↓←).",   # 左下
    "·": "Camera remains still (·)."                   # 静止
}
caption = caption + vocab[mouse[0]]  # 拼接到 caption
```

#### **附加相机参数** (可选)
```python
# 行号: 429-432
caption = caption + \
    "Actual distance moved:" + str(avg_speed*100) + " at 100 meters per second." + \
    "Angular change rate (turn speed):" + str(avg_traj_angle) + "." + \
    "View rotation speed:" + str(avg_rot_angle) + "."
```

### 📂 数据格式示例

#### **TXT 文件内容** (`1hCmaStd-AY_0028950_0030750_frames_00901-00938.txt`)
```
Start Frame: 901
End Frame: 938
Duration: 38 frames
Keys: W+A
Mouse: →
```

#### **生成的 Caption**
```
This video depicts a city walk scene with a first-person view (FPV).
Person moves forward and left (W+A).
Camera turns right (→).
Actual distance moved:4 at 100 meters per second.
Angular change rate (turn speed):0.
View rotation speed:0.
```

### 🗂️ 目录结构
```
mp4_frame/
├── Keys_W_Mouse_·/
│   ├── video_id.mp4
│   └── video_id.txt  # 包含 Keys 和 Mouse 标注
├── Keys_W_A_Mouse_Right/
└── Keys_S_Mouse_Up/
```

### 📊 与论文对应
- **论文 Eq.1 & Eq.2**: 定义了 `vocab_human` 和 `vocab_camera`
- **代码实现**: `vocab` 字典直接对应论文中的词汇表

---

## 4️⃣ GAN-based Distillation - 加速推理（4步生成）

### 📍 核心代码位置
**文件**: `fastvideo/distill_model.py`  
**判别器定义**: `ADD/models/discriminator.py`  
**行号**: 320-355 (训练), 532-579 (初始化)

### 🔧 实现原理
使用 **ProjectedDiscriminator**（基于 DINO ViT 特征）对生成的 latent 进行对抗训练，强制模型在少步采样下也能生成高质量视频。

### 📝 关键代码片段

#### **判别器初始化**
```python
# fastvideo/distill_model.py 行号: 532-535
if args.Distil:
    from ADD.models.discriminator import ProjectedDiscriminator
    discriminator = ProjectedDiscriminator(c_dim=384).train()
    # ... FSDP 包装 ...
```

#### **GAN Loss 计算** (训练循环)
```python
# 行号: 320-355
if Distil:
    # 1) 计算 clean latent (去噪结果)
    model_denoing = xt - t * model_output                    # x_0 = x_t - t*ε
    model_denoing = model_denoing[:,-9:]                     # 只取最后9帧
    model_input_gan = model_input[:,-9:]                     # Ground truth
    
    # 2) 判别器前向传播 (Real vs Fake)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_real, pred_real_f, _ = discriminator(
            model_input_gan.permute(1,0,2,3).detach().reshape(b1*f1,c1,h1,w1), None
        )
        pred_fake, pred_fake_f, _ = discriminator(
            model_denoing.permute(1,0,2,3).detach().reshape(b1*f1,c1,h1,w1), None
        )
    
    # 3) 计算判别器损失 (Hinge Loss)
    pred_fake = torch.cat(pred_fake, dim=1)
    pred_real = torch.cat(pred_real, dim=1)
    loss_real = torch.mean(torch.relu(1.0 - pred_real))
    loss_fake = torch.mean(torch.relu(1.0 + pred_fake))
    loss_d = (loss_real + loss_fake) / 2.0
    
    # 4) 更新判别器
    loss_d.backward()
    d_grad_norm = discriminator.clip_grad_norm_(max_grad_norm).item()
    discriminator_optimizer.step()
    discriminator_optimizer.zero_grad()
    
    # 5) 计算生成器 GAN Loss
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_fake, pred_fake_f, _ = discriminator(
            model_denoing.permute(1,0,2,3).reshape(b1*f1,c1,h1,w1), None
        )
    pred_fake = torch.cat(pred_fake, dim=1)
    pred_fake_f = torch.cat(pred_fake_f, dim=1)
    gan_loss = -torch.mean(pred_fake) - torch.mean(pred_fake_f)  # Non-saturating loss
    
    # 6) 总损失 = 扩散损失 + GAN损失
    loss = loss + 0.01 * gan_loss  # 权重 0.01
```

#### **ProjectedDiscriminator 架构**
```python
# ADD/models/discriminator.py 行号: 176-225
class ProjectedDiscriminator(nn.Module):
    def __init__(self, c_dim: int, diffaug: bool = True, p_crop: float = 0.5):
        super().__init__()
        self.dino = DINO()  # ViT-Small DINO 特征提取器
        self.up = SubPixelConvLayer(in_channels=16, out_channels=3, upscale_factor=4)
        
        # 多尺度判别头
        heads = []
        for i in range(self.dino.n_hooks):
            heads += [str(i), DiscHead(self.dino.embed_dim, c_dim)],
        self.heads = nn.ModuleDict(heads)
    
    def forward(self, x: torch.Tensor, c: torch.Tensor):
        x = self.up(x)  # latent → RGB (upsampling)
        
        # 提取 DINO 多层特征
        features = self.dino(x)
        
        # 对每层特征应用判别头
        logits = []
        for k, head in self.heads.items():
            features[k].requires_grad_(True)
            logits.append(head(features[k], c).view(x.size(0), -1))
        
        return logits, logits_f, features
```

### 🎛️ 训练参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `--Distil` | flag | 启用 GAN 蒸馏 |
| `--discriminator_learning_rate` | 1e-5 | 判别器学习率 |
| `gan_loss` 权重 | 0.01 | GAN loss 权重 |
| 判别帧数 | 9 | 只对最后 9 帧判别 |

### 🚀 推理加速效果
- **训练**: 50 steps (标准扩散)
- **推理**: 4 steps (蒸馏加速) ✅
- **论文对比**: Table 1 显示 4 steps 下性能接近 50 steps

### 📊 与论文对应
- **论文 Section 4.2.3**: "Distribution Matching Distillation (DMD)"
- **代码实现**: GAN-based 对抗蒸馏（实际上不是 DMD，而是改用 GAN）
- **判别器**: 基于 StyleGAN-T 的 ProjectedDiscriminator

---

## 5️⃣ InternVL3-2B 集成 - 增强 Prompt 理解

### 📍 核心代码位置
**文件**: 
- `fastvideo/sample/sample_5b.py` (推理脚本)
- `webapp_single_gpu.py` (Web UI)

**行号**: 
- 1288-1295 (加载模型)
- 798-804, 848-854 (Prompt 精炼)
- 258-282 (Web UI 加载)
- 286-339 (Web UI Prompt 精炼)

### 🔧 实现原理
使用 **InternVL3-2B-Instruct** 多模态大模型，根据输入图片和用户 prompt 生成更详细、更符合图片内容的视频生成 prompt。

### 📝 关键代码片段

#### **模型加载** (推理脚本)
```python
# fastvideo/sample/sample_5b.py 行号: 1288-1295
path = 'InternVL3-2B-Instruct'
camption_model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True
).eval().to(device)

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
```

#### **Prompt 精炼** (I2V 模式)
```python
# 行号: 798-804
if prompt1 == None:
    question = '<image>\nWe want to generate a video using this image. ' \
               'Please generate a prompt word for the video of this image. ' \
               'Don\'t split it into points; just write a paragraph directly'
else:
    question = f'<image>\nWe want to generate a video using this prompt: "{prompt1}". ' \
               f'Please modify and refine this prompt for the video of this image (<image>). ' \
               f'Note that "{prompt1}" must appear and revolve around the extension. ' \
               'Don\'t split it into points; just write a paragraph directly'

response = camption_model.chat(tokenizer, pixel_values, question, generation_config)
```

#### **Web UI 加载** (CPU 常驻，按需上 GPU)
```python
# webapp_single_gpu.py 行号: 258-282
@torch.inference_mode()
def load_caption_model() -> str:
    global CAP_READY
    LOGGER.info("[load_caption_model] start (BF16)")
    
    if CAP_READY:
        return "✅ InternVL 已加载（BF16）"
    
    from transformers import AutoModel, AutoTokenizer
    caption_model = AutoModel.from_pretrained(
        INTERNVL_PATH,  # "./InternVL3-2B-Instruct"
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True
    ).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        INTERNVL_PATH, trust_remote_code=True, use_fast=False
    )
    
    MODELS.caption_model = caption_model.cpu()  # 先放 CPU 节省显存
    MODELS.tokenizer = tokenizer
    CAP_READY = True
    
    return f"✅ InternVL 已加载（BF16）"
```

#### **Web UI Prompt 精炼** (临时上 GPU，用完回 CPU)
```python
# 行号: 286-339
@torch.inference_mode()
def refine_prompt_from_image(image_path: str, user_prompt: str) -> str:
    if not CAP_READY or MODELS.caption_model is None:
        return user_prompt
    
    try:
        # 1) 图片预处理（dynamic tiles）
        img = Image.open(image_path).convert("RGB")
        tiles = dynamic_preprocess(img, image_size=448, use_thumbnail=True, max_num=12)
        px = torch.stack([transform(im) for im in tiles])
        
        # 2) 临时将模型移到 GPU
        caption_model = MODELS.caption_model.to(MODELS.device, dtype=DTYPE)
        px = px.to(MODELS.device, dtype=DTYPE)
        
        # 3) 构建 prompt 精炼问题
        question = (
            f'<image>\nWe want to generate a video using this prompt: "{user_prompt}". '
            'Please refine it for this image (<image>). Keep it one paragraph.'
        )
        
        # 4) InternVL 推理
        gen_cfg = dict(max_new_tokens=512, do_sample=True)
        out = caption_model.chat(MODELS.tokenizer, px, question, gen_cfg)
        
        # 5) 用完后移回 CPU
        MODELS.caption_model.cpu()
        
        return out or user_prompt
    
    except Exception as e:
        LOGGER.exception("[caption] refine failed: %s", e)
        try:
            MODELS.caption_model.cpu()
        except Exception:
            pass
        return user_prompt
```

### 🖼️ 图片预处理 (Dynamic Preprocess)
```python
# 行号: 290-321
def dynamic_preprocess(image: Image.Image, min_num=1, max_num=12, 
                      image_size=448, use_thumbnail=True):
    """
    将图片切分成多个 448x448 的 tiles，用于 InternVL 处理
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    
    # 计算最佳切分方案
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    
    # 选择最接近的切分方案
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    
    # 切分图片
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = split_to_patches(image.resize((target_width, target_height)), image_size)
    
    # 可选添加缩略图
    if use_thumbnail and len(blocks) != 1:
        blocks.append(image.resize((image_size, image_size)))
    
    return blocks
```

### 🎨 System Prompt (系统提示词)
```python
# wan23/utils/system_prompt.py
SYSTEM_PROMPT_I2V_CN = """
你需要根据这张图片(<image>)和我提供的事件描述，生成一段适合用于视频生成的提示词。

要求：
1. 必须围绕我提供的事件描述展开
2. 充分结合图片内容，描述画面中的元素如何随着事件发展
3. 使用动态、生动的语言描述运动和变化
4. 保持一段话，不要分点列举
"""

SYSTEM_PROMPT_I2V_EN = """
Based on this image (<image>) and the event description I provide, 
generate a prompt suitable for video generation.

Requirements:
1. Must revolve around the provided event description
2. Fully integrate image content, describing how elements evolve with the event
3. Use dynamic, vivid language to describe motion and changes
4. Keep it as one paragraph, don't split into bullet points
"""
```

### 📊 模型信息

| 项目 | 信息 |
|------|------|
| 模型名称 | InternVL3-2B-Instruct |
| 参数量 | 2B |
| 基座模型 | OpenGVLab/InternVL3-2B-Pretrained |
| 语言模型 | Qwen2.5-2B-Instruct |
| 视觉编码器 | InternViT-300M-448px |
| 输入分辨率 | 448×448 (动态切分) |
| 精度 | BF16 |

### 🚀 使用场景

#### **场景 1: I2V 首次生成**
```
输入图片: 城市街道.jpg
用户 Prompt: "A sunny day"
InternVL 输出: "On a bright sunny day, the bustling city street comes alive 
                with pedestrians walking along the sidewalks, cars driving 
                past modern buildings, and sunlight casting warm shadows..."
```

#### **场景 2: 事件驱动生成**
```
输入图片: 猫咪.jpg
用户 Prompt: "The cat suddenly jumps"
InternVL 输出: "In this cozy room, a curious tabby cat suddenly springs into 
                action, leaping from the floor towards the window sill with 
                graceful agility, its whiskers twitching with excitement..."
```

### 📊 与论文对应
- **论文 Figure 2**: "Using InternVL3-78B to re-annotate for I2V"
- **代码实现**: 使用 InternVL3-**2B**（论文用 78B，代码降级为 2B 节省资源）
- **功能**: Prompt refinement for event-driven generation

---

## 🗺️ 代码关系图

```
YUME 架构
│
├─ 1️⃣ TSCM (历史帧压缩)
│   └─ wan23/modules/model.py::WanModel.forward()
│       ├─ patch_embedding_2x/4x/8x/16x
│       └─ 5 级自适应压缩策略
│
├─ 2️⃣ Self-Forcing (模型预测作为历史)
│   ├─ webapp_single_gpu.py::long_generate()
│   │   └─ 更新 model_input_latent (line 836-840)
│   └─ fastvideo/distill_model.py::distill_one_step()
│       └─ 训练时使用模型预测 (line 390)
│
├─ 3️⃣ Keyboard/Mouse Control (相机控制)
│   └─ fastvideo/dataset/t2v_datasets.py
│       ├─ parse_txt_file() - 解析 TXT
│       ├─ vocab (Keys) - W/A/S/D 转自然语言
│       └─ vocab (Mouse) - ↑/↓/←/→ 转自然语言
│
├─ 4️⃣ GAN-based Distillation (加速推理)
│   ├─ fastvideo/distill_model.py
│   │   ├─ 初始化 ProjectedDiscriminator (line 532-535)
│   │   └─ GAN Loss 计算 (line 320-355)
│   └─ ADD/models/discriminator.py
│       └─ ProjectedDiscriminator (DINO ViT + 多头判别)
│
└─ 5️⃣ InternVL3-2B (Prompt 增强)
    ├─ fastvideo/sample/sample_5b.py
    │   └─ 加载 InternVL + prompt 精炼 (line 1288-1295)
    └─ webapp_single_gpu.py
        ├─ load_caption_model() - CPU 常驻
        └─ refine_prompt_from_image() - 临时上 GPU
```

---

## 🎯 快速索引

### 按功能查找
- **长视频生成**: `webapp_single_gpu.py::long_generate()`
- **训练脚本**: `fastvideo/distill_model.py`
- **数据加载**: `fastvideo/dataset/t2v_datasets.py`
- **模型定义**: `wan23/modules/model.py::WanModel`
- **VAE**: `wan23/modules/vae2_2.py::Wan2_2_VAE`
- **判别器**: `ADD/models/discriminator.py::ProjectedDiscriminator`

### 按论文章节查找
- **Section 3 (Dataset)**: `fastvideo/dataset/t2v_datasets.py`
- **Section 4.1 (TSCM)**: `wan23/modules/model.py` line 486-720
- **Section 4.2.2 (Self-Forcing)**: `webapp_single_gpu.py` line 836-863
- **Section 4.2.3 (Distillation)**: `fastvideo/distill_model.py` line 320-355
- **Figure 2 (InternVL)**: `fastvideo/sample/sample_5b.py` line 1288-1295

---

## 🚀 运行示例

### 使用所有创新点的完整流程

```bash
# 1. 加载带 TSCM 的模型
python webapp_single_gpu.py

# 2. 在 Web UI 中:
#    - 勾选"加载 InternVL (Caption)" ✅ (创新点 5)
#    - 上传图片
#    - 输入 Prompt: "Person walks forward, camera turns right"  ✅ (创新点 3)
#    - 勾选"从图片精炼 Prompt" ✅ (创新点 5)
#    - 点击"开始生成" → 首段视频生成 (使用 TSCM ✅ 创新点 1)
#    - 勾选"继续续帧" ✅ (创新点 2: Self-Forcing)
#    - 点击"继续生成" → 长视频生成

# 3. 训练带 GAN 蒸馏的模型 ✅ (创新点 4)
bash scripts/finetune/finetune.sh  # 已设置 --Distil 和 --MVDT
```

---

## 📚 扩展阅读

### 相关论文实现
- **OSV (One Step Video)**: `fastvideo/distill_model.py` (GAN 蒸馏基础)
- **Wan 2.2**: `wan23/textimage2video.py` (基座模型)
- **StyleGAN-T**: `ADD/models/discriminator.py` (判别器设计)
- **InternVL3**: `InternVL3-2B-Instruct/` (多模态理解)

### 配置文件
- **5B 模型配置**: `wan23/configs/wan_ti2v_5B.py`
- **训练参数**: `scripts/finetune/finetune.sh`
- **推理参数**: `scripts/inference/sample_5b.sh`

---

## 📞 问题排查

### Q1: TSCM 没有生效？
**检查**: `wan23/modules/model.py` line 486-495 的 patch_embedding 层是否正确初始化

### Q2: Self-Forcing 效果不好？
**检查**: `webapp_single_gpu.py` line 872-876，确保 `LAST["last_model_input_latent"]` 被正确保存

### Q3: Keyboard/Mouse 控制不生效？
**检查**: 数据集 TXT 文件格式是否正确（参见 `test_video/` 示例）

### Q4: GAN 蒸馏训练不稳定？
**调整**: `fastvideo/distill_model.py` line 354 的 `gan_loss` 权重（默认 0.01）

### Q5: InternVL 精炼后 prompt 质量差？
**检查**: `wan23/utils/system_prompt.py` 的 system prompt 是否适合你的场景

---

## ✅ 总结

本文档详细标注了 YUME 项目的 **5 个核心创新点** 在代码中的具体位置：

1. ✅ **TSCM** → `wan23/modules/model.py` (5级自适应压缩)
2. ✅ **Self-Forcing** → `webapp_single_gpu.py` + `fastvideo/distill_model.py`
3. ✅ **Keyboard/Mouse Control** → `fastvideo/dataset/t2v_datasets.py`
4. ✅ **GAN-based Distillation** → `fastvideo/distill_model.py` + `ADD/models/discriminator.py`
5. ✅ **InternVL3-2B** → `fastvideo/sample/sample_5b.py` + `webapp_single_gpu.py`

每个创新点都包含：
- 📍 准确的文件路径和行号
- 🔧 实现原理说明
- 📝 关键代码片段（带注释）
- 📊 与论文的对应关系
- 🚀 使用示例

建议按顺序阅读各个创新点的实现，理解它们如何协同工作以实现 YUME 的长视频生成能力。
