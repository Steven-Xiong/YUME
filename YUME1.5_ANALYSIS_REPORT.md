# YUME 1.5 技术分析报告

> 论文: *Yume1.5: A Text-Controlled Interactive World Generation Model*  
> arXiv: 2512.22096v1 (2025年12月)  
> 机构: 上海 AI 实验室 / 复旦大学  
> GitHub: https://github.com/stdstu12/YUME

---

## 目录

1. [项目概述](#1-项目概述)
2. [核心创新点详解](#2-核心创新点详解)
   - 2.1 TSCM — 联合时空-通道建模
   - 2.2 Self-Forcing with TSCM — 抑制误差累积
   - 2.3 键盘/鼠标控制转自然语言
   - 2.4 GAN-based Distillation — 4步推理加速
   - 2.5 InternVL3-2B — 多模态 Prompt 精炼
3. [训练阶段分析](#3-训练阶段分析)
4. [推理流程梳理](#4-推理流程梳理)
5. [性能对比](#5-性能对比)
6. [代码文件速查](#6-代码文件速查)

---

## 1. 项目概述

**YUME 1.5** 是基于 Wan2.2-5B 预训练模型的交互式世界生成系统，支持从单张图片或文本生成可**无限自回归探索**的动态世界，并通过键盘（W/A/S/D）和鼠标（↑↓←→）实现直觉性相机控制。

### 三种生成模式

| 模式 | 描述 |
|------|------|
| Text-to-World (T2V) | 从文本描述生成可探索世界 |
| Image-to-World (I2V) | 从单张图片生成视频世界 |
| Event Editing | 用文本触发场景内事件（如"突然下雨"） |

### 核心性能指标

| 模型 | 推理时间(s)↓ | 指令跟随↑ | 主题一致性↑ | 背景一致性↑ |
|------|------------|---------|----------|----------|
| Wan-2.1 | 611 | 0.057 | 0.859 | 0.899 |
| MatrixGame | 971 | 0.271 | 0.911 | 0.932 |
| Yume 1.0 | 572 | 0.657 | 0.932 | 0.941 |
| **Yume 1.5** | **8** | **0.836** | **0.932** | **0.945** |

**核心亮点**：推理速度提升 **70×** 以上，指令跟随能力提升 27%（0.657→0.836），单卡 A100 实现 **12 FPS（540P）**实时生成。

---

## 2. 核心创新点详解

---

### 2.1 TSCM — 联合时空-通道建模 (Section 4.2)

TSCM 是 Yume1.5 最核心的架构创新，解决了长视频自回归生成中**历史帧上下文随时间线性增长**导致计算爆炸的根本问题。分为两个互补的子机制。

---

#### 子机制 A：时空压缩 (Temporal-Spatial Compression)

**核心思想**：距离越远的历史帧，空间分辨率压缩越大；保留最近帧的高精度信息。

论文 Eq.3 定义的压缩方案：

| 历史帧位置 | Patchify 压缩率 (T,H,W) | 等效 Token 倍率 |
|-----------|------------------------|----------------|
| t-1 到 t-2（最近帧） | (1, 2, 2) | 4x tokens |
| t-3 到 t-6 | (1, 4, 4) | 16x tokens |
| t-7 到 t-23 | (1, 8, 8) | 64x tokens |
| 更远历史帧 | (1, 16, 16) | 256x tokens |
| 初始帧 | (1, 2, 2) | 特殊保留 |

**代码实现** — `wan23/modules/model.py` 第 486-494 行：

```python
# 通过插值预训练权重生成多级压缩的 patch_embedding，无需重新训练
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

自适应压缩层级（`wan23/modules/model.py` 第 599-718 行）：

| 历史帧总数 | 压缩层级数 | 压缩比分布（从旧→新） |
|-----------|-----------|---------------------|
| ≤ 6 帧 | 3层 | 1x, 2x, 1x |
| ≤ 22 帧 | 4层 | 1x, 4x, 2x, 1x |
| ≤ 86 帧 | 5层 | 1x, 8x, 4x, 2x, 1x |
| ≤ 342 帧 | 6层 | 2x, 16x, 8x, 4x, 2x, 1x |
| > 342 帧 | 7层 | 2x, 32x, 16x, 8x, 4x, 2x, 1x |

---

#### 子机制 B：通道压缩 + Linear Attention (Channel Compression)

**这是 Yume1.5 相比 Yume1.0 的最大架构创新**。

对历史帧再施加 `(8, 4, 4)` 超高压缩率，通道维降至 96，得到 `z_linear`，通过 **Linear Attention** 而非标准 Self-Attention 进行融合。

数学公式（论文 Eq.4 & Eq.5）：

$$o^l = \frac{\sum_{i=1}^N v_i^l \phi(k_i^l)^T \phi(q^l)}{\sum_{j=1}^N \phi(k_j^l)^T \phi(q^l)}, \quad \hat{o}^t = \text{Norm}(o^t) W^o$$

其中 $\phi$ 为 ReLU 激活，将标准 softmax attention 的 $O(N^2)$ 复杂度降为 $O(N)$。

**两者的互补设计逻辑**：
- 标准 Attention 对 **Token 数量**敏感（O(N²)） → 历史帧用**时空压缩**减少 Token 数
- Linear Attention 对 **通道维度**敏感 → 历史帧用**通道压缩**降维
- 两者合并，实现联合时空-通道双向压缩

**推理速度验证**（论文 Figure 7）：

| 方法 | 视频块增加时推理时间变化 |
|------|----------------------|
| TSCM（Yume1.5） | **恒定稳定**（超过8个视频块后每步时间不变） |
| 空间压缩（Yume 1.0） | 逐渐增长 |
| 全上下文输入 | 急剧增长，8块时截断 |

**Ablation 对比**（论文 Table 2）：

| 方法 | 指令跟随↑ | 主题一致性↑ |
|------|---------|----------|
| TSCM（Yume1.5） | **0.836** | 0.932 |
| 空间压缩（Yume 1.0） | 0.767 | 0.935 |

---

### 2.2 Self-Forcing with TSCM — 抑制误差累积 (Section 4.3)

**问题**：自回归生成中，训练时看真实帧，推理时看自己生成的帧（含误差），这个 train-inference discrepancy 导致误差随块数累积，后期视频质量下降。

**方案**：类似 Self-Forcing [Huang et al., 2025]，但用 TSCM 替换了 KV Cache，使历史上下文可以无限延伸（而不是截断）。

训练时主动将模型自己预测的 clean latent 作为下一块历史输入：

**训练代码** — `fastvideo/distill_model.py` 第 386-390 行：

```python
# Self-Forcing: 使用模型自己预测的 clean latent 作为历史条件
if i+1 == 50:
    temp_x0 = latent[:,-9:,:,:] + (0-sampling_sigmas[i])*noise_pred_cond[:,-9:,:,:]
else:
    temp_x0 = latent[:,-9:,:,:] + (sampling_sigmas[i+1]-sampling_sigmas[i])*noise_pred_cond[:,-9:,:,:]

latent = torch.cat([
    noise[:,:-9,:,:]*sampling_sigmas[max(i-1,0)] + (1-sampling_sigmas[max(i-1,0)])*model_input[:,:-9,:,:],
    temp_x0  # 新预测的 clean latent ✅
], dim=1)
```

**推理代码** — `webapp_single_gpu.py` 第 836-861 行：

```python
# 更新 latent 历史（Self-Forcing 核心）
if is_i2v_mode or seg > 0:
    model_input_latent = torch.cat([
        model_input_latent[:,:-latent_frame_zero,:,:],  # 旧历史
        latent[:,-latent_frame_zero:,:,:]                # 新生成的 clean latent ✅
    ], dim=1)

# 更新像素历史
if is_i2v_mode or seg > 0:
    model_input_de = torch.cat([
        model_input_de[:,:-frame_zero,:,:],
        video_tail_px[:,-frame_zero:,:,:]  # 新解码的视频帧 ✅
    ], dim=1)
```

**效果验证**（论文 Figure 5/6）：

| 指标 | 第6块（含 Self-Forcing+TSCM） | 第6块（无） |
|------|------------------------------|-----------|
| Aesthetic Score | **0.523** | 0.442 |
| Image Quality | **0.601** | 0.542 |

---

### 2.3 键盘/鼠标控制转自然语言 (Section 3.1, Eq.1 & 2)

**核心思路**：将离散的键盘（W/A/S/D）和鼠标（↑↓←→）控制信号，通过**预定义词汇表**映射为文本描述，拼接进 Caption，让语言模型来学习控制。

**Caption 分解为两部分**（Yume1.5 架构创新）：

- **Event Description**（事件描述）：场景内容，只在初次生成时用 T5 编码一次
- **Action Description**（动作描述）：键盘+鼠标控制，**有限集合可预计算缓存**，大幅减少 T5 计算开销

**代码实现** — `fastvideo/dataset/t2v_datasets.py` 第 393-418 行：

```python
# 键盘控制词汇表 (vocab_human)
vocab_k = {
    "W":   "Person moves forward (W).",
    "A":   "Person moves left (A).",
    "S":   "Person moves backward (S).",
    "D":   "Person moves right (D).",
    "W+A": "Person moves forward and left (W+A).",
    "W+D": "Person moves forward and right (W+D).",
    "S+D": "Person moves backward and right (S+D).",
    "S+A": "Person moves backward and left (S+A).",
    "None":"Person stands still (·).",
}

# 鼠标控制词汇表 (vocab_camera)
vocab_c = {
    "→":  "Camera turns right (→).",
    "←":  "Camera turns left (←).",
    "↑":  "Camera tilts up (↑).",
    "↓":  "Camera tilts down (↓).",
    "↑→": "Camera tilts up and turns right (↑→).",
    "↑←": "Camera tilts up and turns left (↑←).",
    "↓→": "Camera tilts down and turns right (↓→).",
    "↓←": "Camera tilts down and turns left (↓←).",
    "·":  "Camera remains still (·).",
}
caption = caption + vocab_k[keys[0]] + vocab_c[mouse[0]]
```

**生成的 Caption 示例**：

```
This video depicts a city walk scene with a first-person view (FPV).
Person moves forward and left (W+A).
Camera turns right (→).
Actual distance moved: 4 at 100 meters per second.
Angular change rate (turn speed): 0.
View rotation speed: 0.
```

**数据集目录结构**：

```
mp4_frame/
├── Keys_W_Mouse_·/          # 前进 + 相机静止
├── Keys_W_A_Mouse_Right/    # 左前 + 相机右转
└── Keys_S_Mouse_Up/         # 后退 + 相机上仰
```

---

### 2.4 GAN-based Distillation — 4步推理加速 (Section 4.3)

**目标**：将需要 50 步采样的扩散模型压缩为 **4步**推理，同时保持生成质量。

论文采用 DMD（Distribution Matching Distillation），通过最小化生成分布与真实分布之间的 KL 散度实现（论文 Eq.6）。代码实现则采用更直接的 **GAN 对抗训练**：

**代码实现** — `fastvideo/distill_model.py` 第 320-354 行：

```python
if Distil:
    # 1) 计算 clean latent (去噪结果)
    model_denoing = xt - t * model_output        # x_0 = x_t - t*ε
    model_denoing = model_denoing[:,-9:]          # 只判别最后9帧

    # 2) 判别器前向 (Real vs Fake)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_real, pred_real_f, _ = discriminator(
            model_input_gan.permute(1,0,2,3).detach().reshape(b1*f1,c1,h1,w1), None)
        pred_fake, pred_fake_f, _ = discriminator(
            model_denoing.permute(1,0,2,3).detach().reshape(b1*f1,c1,h1,w1), None)

    # 3) Hinge Loss
    loss_real = torch.mean(torch.relu(1.0 - pred_real)) + ...
    loss_fake = torch.mean(torch.relu(1.0 + pred_fake)) + ...
    loss_d = (loss_real + loss_fake) / 2.0
    loss_d.backward()

    # 4) 生成器 GAN Loss
    gan_loss = -torch.mean(pred_fake) - torch.mean(pred_fake_f)
    loss = loss + 0.01 * gan_loss  # 权重 0.01
```

**判别器架构** — `ADD/models/discriminator.py`：

- **ProjectedDiscriminator**：基于 DINO ViT-Small 特征
- `SubPixelConvLayer`：将 latent (16ch) 上采样至 RGB 空间
- 多尺度判别头：对 DINO 各层特征分别判别
- Hinge Loss + Non-saturating Loss 混合

**关键超参**：

| 参数 | 值 |
|------|---|
| 判别器学习率 | 1e-5 |
| GAN Loss 权重 | 0.01 |
| 推理步数（训练） | 50 steps |
| 推理步数（蒸馏后） | **4 steps** |
| 判别帧数 | 9帧（仅最新 chunk） |

---

### 2.5 InternVL3-2B — 多模态 Prompt 精炼 (Section 3.1, Figure 2)

**问题**：用户输入的简短 prompt 难以精确描述图片内容中应发生的视频事件。

**方案**：用 InternVL3（论文用 78B，代码用 **2B** 节省资源）多模态大模型，根据输入图片 + 用户 prompt 自动生成更丰富的视频生成 prompt。

同时用 InternVL3-78B 对训练数据集重新标注，区分：
- **T2V 标注**：描述场景和上下文（保留原始字幕）
- **I2V 标注**：聚焦动态事件（VLM 重新生成，如图 Figure 2）

**代码实现** — `webapp_single_gpu.py` 第 258-547 行：

```python
# 内存优化策略：CPU 常驻 + 临时上 GPU
caption_model = AutoModel.from_pretrained(
    "InternVL3-2B-Instruct",
    torch_dtype=torch.bfloat16,
    use_flash_attn=True,
).eval()
MODELS.caption_model = caption_model.cpu()  # 先放 CPU 节省显存

# 使用时：图片 → 动态切分 (dynamic tiles) → 临时上 GPU
caption_model = MODELS.caption_model.to(MODELS.device, dtype=DTYPE)
question = (
    f'<image>\nWe want to generate a video using this prompt: "{user_prompt}". '
    'Please refine it for this image (<image>). Keep it one paragraph.'
)
out = caption_model.chat(MODELS.tokenizer, px, question, gen_cfg)
MODELS.caption_model.cpu()  # 用完立即回 CPU
```

**图片预处理**：`dynamic_preprocess()` 将图片动态切分为多个 448×448 的 tiles（最多12块）。

**模型信息**：

| 项目 | 信息 |
|------|------|
| 论文使用 | InternVL3-78B |
| 代码实现 | InternVL3-**2B**（降级节省资源） |
| 视觉编码器 | InternViT-300M-448px |
| 语言模型 | Qwen2.5-2B-Instruct |
| 精度 | BF16 |

---

## 3. 训练阶段分析

### 论文描述 vs 代码实现

论文（Section 4.3，5.1.1）描述了 2 个正式训练阶段，代码层面实际对应 **3 个可分离子阶段**：

---

### Stage 1：Foundation Model — `train-5b.py`

**脚本**：`scripts/finetune/finetune-5b.sh`

```bash
torchrun --nproc_per_node 8 train-5b.py \
    --learning_rate=1e-5 \
    --max_train_steps=600000
    # 无 --MVDT，无 --Distil
```

**功能**：
- 在混合数据集上训练（Real-world + Synthetic + Event Dataset）
- T2V 和 I2V **交替训练**（奇数步用 T2V 数据，偶数步用 I2V 数据）
- 标准 Rectified Flow 损失（无 GAN，无 mask）
- 论文训练 **10,000 iterations**

**实现状态**：✅ 完整实现
- 注：`train-5b.py` 内有 MVDT 相关代码，但已被**注释掉**（`# MVDT`），说明 MVDT 曾尝试在 Stage 1 引入但后来移除

---

### Stage 2：MVDT Self-Forcing — `distill_model.py --MVDT`

**核心**：Masked Video Diffusion Training（参考 MDT, sail-sg/MDT）

在每个训练 step 中，额外进行一次**随机 Token Masking** 的前向+反向传播：

```python
# fastvideo/distill_model.py
if MVDT:
    _, _, _, loss_dict_mask = denoiser.training_losses(
        transformer, model_input, arg_c,
        enable_mask=True   # 触发随机 mask + sideblock 重建
    )
    loss = loss_dict_mask["loss"].mean()
    loss.backward()  # 单独的 backward pass
```

**模型内部** — `wan23/modules/model.py` 第 764-848 行：
- 随机 mask 30%~50% 的 token（仅历史帧部分）
- 用 `sideblock`（独立 WanAttentionBlock）重建被 mask 的 token
- 在 DiT Block 中间层插入重建过程（第 `len_blocks//2` 个 block 后）

**意义**：让模型学会在历史上下文**不完整/损坏**时也能生成一致预测，与 Self-Forcing 的需求完美对应（推理时历史帧含生成误差，等价于"部分损坏"）。

**实现状态**：✅ 完整实现

---

### Stage 3：GAN Distillation — `distill_model.py --MVDT --Distil`

**脚本**：`scripts/finetune/finetune.sh`（**Stage 2 + Stage 3 合并运行**）

```bash
torchrun --nproc_per_node 8 fastvideo/distill_model.py \
    --MVDT \   # Stage 2: Masked Self-Forcing
    --Distil \ # Stage 3: GAN 蒸馏
    --learning_rate=1e-5 \
    --discriminator_learning_rate=1e-5
```

在 Stage 2 的 Self-Forcing 损失之上，额外加入 GAN 判别器对抗损失，强制 50 steps → **4 steps** 推理能力。论文训练 **600 iterations**。

**实现状态**：✅ 完整实现

---

### 训练阶段总结

```
Stage 1: Foundation Model (train-5b.py)
  ├─ T2V + I2V 交替训练          ✅ 实现
  ├─ Rectified Flow 损失          ✅ 实现
  ├─ 混合数据集 (Real+Syn+Event)   ✅ 实现
  └─ MVDT 在 Stage1 尝试          ❌ 已注释掉

Stage 2: Self-Forcing/MVDT (distill_model.py --MVDT)
  ├─ 随机 Mask 历史 Token (30~50%) ✅ 实现
  ├─ SideBlock Token 重建          ✅ 实现
  └─ Self-Forcing 循环采样         ✅ 实现

Stage 3: GAN Distillation (distill_model.py --MVDT --Distil)
  ├─ ProjectedDiscriminator        ✅ 实现
  ├─ Hinge Loss                    ✅ 实现
  └─ 50 steps → 4 steps 加速      ✅ 实现

注意: finetune.sh 将 Stage2 + Stage3 合并为同一次运行（600 iters）
```

---

## 4. 推理流程梳理

**入口函数**：`webapp_single_gpu.py::long_generate(g: LongGenArgs)`

---

### 输入/输出

```
输入:
  g.mode           = "I2V" | "T2V"
  g.jpg_path       = 首帧图片路径（I2V 必须提供）
  g.prompt         = 事件描述文本
  g.camera_movement1 = 键盘控制 "W" / "S+A" / "None" / ...
  g.camera_movement2 = 鼠标控制 "→" / "↑←" / "·" / ...
  g.sample_steps   = 扩散步数（默认 4 步，蒸馏后）
  g.sample_num     = 生成 chunk 数量
  g.frame_zero     = 每个 chunk 的像素帧数（例 33）
  g.resolution     = "544x960" | "704x1280"
  LAST[...]        = 上次生成的历史状态（续帧时使用）

输出:
  out_path         = 生成的 .mp4 文件路径
  final_prompt     = 最终使用的 prompt 文本（供展示）
```

---

### 完整推理流程

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1：初始化历史状态
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

续帧模式 (continue_from_last=True):
  model_input_de      ← LAST["last_model_input_de"]      像素历史
  model_input_latent  ← LAST["last_model_input_latent"]  latent 历史

I2V 首次生成:
  图片 → 复制16帧头部 → pixel_values_vid (C, 16+33, H, W)
  VAE Encode → model_input_latent (C, Fz_hist, Hz, Wz)

T2V 首次生成:
  model_input_de / latent 均为空，从纯噪声出发

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2：构建 Prompt（三层叠加）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  [1] 基础前缀: "First-person perspective."

  [2] 动作描述（vocab 映射）:
      camera_movement1="W" → "The camera pushes forward (W)."
      camera_movement2="→" → "The camera pans to the right (→)."

  [3] （可选）InternVL3-2B 精炼:
      图片 + 用户 prompt → InternVL → 更详细的一段话描述

  final_prompt = [前缀] + [动作描述] + [精炼描述/用户 prompt]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3：T5 文本编码（wan.generate）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  final_prompt → T5 Encoder（5B 参数，4096 维）
  ↓
  arg_c = {
    context: text embeddings,
    seq_len: token 序列长度,
    grid_sizes: latent 空间尺寸,
    freqs: RoPE 位置编码,
    ...
  }
  noise    = 高斯噪声 (C, Fz_hist+latent_frame_zero, Hz, Wz)
  mask2    = 时序 mask（0=历史帧/固定, 1=新生成帧）
  img_lat  = 历史 latent 条件

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 4：Chunk 循环（sample_num 次）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

for seg in range(sample_num):

  [A] 准备 latent 输入:
      seg==0 (I2V):  latent = [model_input_latent(历史) | noise(新帧tail)]
      seg>0:         latent = [model_input_latent(累积历史) | 新 noise(tail)]

  [B] 扩散采样循环（steps=4）:
      for i in range(4):

        ① 构建时间步向量 tvec:
           历史帧部分: mask2=0 → t ≈ 0（无噪声，不更新）
           新帧 tail 部分: t = sampling_sigmas[i] * 1000

        ② Transformer 前向（TSCM 在此发生）:
           transformer(latent, t=tvec, **arg_c)
           │
           └─ TSCM：根据历史帧数自动选压缩级别
              ≤6帧   → patch_embedding_2x（2x 压缩）
              ≤22帧  → 4x/2x 分段压缩
              ≤86帧  → 8x/4x/2x 分段压缩
              ≤342帧 → 16x/8x/4x/2x 分段压缩
              >342帧 → 32x/16x/8x/4x/2x 最大压缩
           │
           └─ 输出 noise_pred（只用 tail 部分）

        ③ Euler 步（只更新 tail 帧，历史帧不动）:
           new_tail = tail + (σ_{i+1} - σ_i) × pred_tail
           latent = concat([历史帧 | new_tail])

  [C] VAE 解码新帧:
      video_tail = vae.decode(latent[:, -latent_frame_zero:])
      shape: (C, frame_zero, H, W) → 像素视频帧

  [D] Self-Forcing 更新历史（关键！）:
      model_input_latent = concat([旧历史 | latent的新 tail])   ← clean latent ✅
      model_input_de     = concat([旧历史px | video_tail的新 tail]) ← 解码视频 ✅

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 5：后处理 & 持久化
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  video_cat = concat(所有 video_tail chunks, dim=time)
  → 写出 .mp4 文件（16 FPS）

  LAST["last_model_input_latent"] = model_input_latent.detach()
  LAST["last_model_input_de"]     = model_input_de.detach()
  LAST["frame_total"]             = 累计帧数
  LAST["last_video_path"]         = 视频路径
  LAST["last_prompt"]             = final_prompt
  ↑ 保存供下次"续帧"使用
```

---

### 关键数据形状追踪

```
# VAE 压缩比: 空间 8x，时间 4x
# 示例: 视频帧 33帧，分辨率 544×960
#   → latent_frame_zero = (33-1)//4 + 1 = 9
#   → latent 空间: (C=16, 9, 68, 120)

# 第1块 (I2V):
#   model_input_latent: (16, 9_hist, 68, 120)    ← 初始图片的 latent
#   noise:              (16, 9+9_new, 68, 120)
#   latent (初始):      (16, 18, 68, 120)
#   → 4步去噪后:
#   latent (clean):     (16, 18, 68, 120)
#   VAE decode tail:    (C, 33, 544, 960)        → 视频像素

# 第2块 (Self-Forcing 历史累积):
#   model_input_latent: (16, 9_hist+9_chunk1, 68, 120)  ← 加入 chunk1
#   → TSCM 对这18帧历史自动选压缩级别 (≤22帧 → 4层压缩)

# 第3块及以后:
#   model_input_latent 持续增长，TSCM 保持推理时间稳定
```

---

### Inference 模式对比

| 场景 | latent 初始化 | mask2 设置 | 备注 |
|------|-------------|-----------|------|
| T2V 首次 | 纯噪声 | 全1（全部生成） | 无历史 |
| I2V 首次 | 历史+噪声 | 历史帧=0，tail=1 | 首帧锚定 |
| 续帧（第2块起） | 累积历史+噪声 | 历史帧=0，tail=1 | Self-Forcing |

---

## 5. 性能对比

### I2V 生成质量（论文 Table 1）

| 模型 | 推理时间(s)↓ | 指令跟随↑ | 主题一致性↑ | 背景一致性↑ | 运动平滑↑ | 美观分↑ | 图质分↑ |
|------|------------|---------|----------|----------|---------|--------|--------|
| Wan-2.1 | 611 | 0.057 | 0.859 | 0.899 | 0.961 | 0.494 | 0.695 |
| MatrixGame | 971 | 0.271 | 0.911 | 0.932 | 0.983 | 0.435 | 0.750 |
| Yume 1.0 | 572 | 0.657 | 0.932 | 0.941 | 0.986 | 0.518 | 0.739 |
| **Yume 1.5** | **8** | **0.836** | **0.932** | **0.945** | **0.985** | 0.506 | 0.728 |

### 长视频生成稳定性（论文 Figure 5/6，T2V 模式，第6块）

| 指标 | 含 Self-Forcing+TSCM | 不含 |
|------|---------------------|------|
| Aesthetic Score | **0.523** | 0.442 |
| Image Quality | **0.601** | 0.542 |

### TSCM Ablation（论文 Table 2）

| 压缩方案 | 指令跟随↑ |
|---------|---------|
| TSCM（Yume1.5） | **0.836** |
| 空间压缩（Yume1.0） | 0.767 |

---

## 6. 代码文件速查

### 按创新点

| 创新点 | 核心文件 | 关键行号 |
|--------|---------|---------|
| TSCM 多级 patch_embedding | `wan23/modules/model.py` | 486-495 |
| TSCM 自适应压缩逻辑 | `wan23/modules/model.py` | 599-718 |
| Linear Attention 融合 | `wan23/modules/model.py` | DiT Block 内 |
| Self-Forcing 推理更新 | `webapp_single_gpu.py` | 836-861 |
| Self-Forcing 训练循环 | `fastvideo/distill_model.py` | 386-390 |
| MVDT Mask 训练 | `wan23/modules/model.py` | 764-848 |
| 键鼠控制词汇表 | `fastvideo/dataset/t2v_datasets.py` | 393-418 |
| GAN 蒸馏训练 | `fastvideo/distill_model.py` | 320-354 |
| ProjectedDiscriminator | `ADD/models/discriminator.py` | 176-225 |
| InternVL 加载 | `webapp_single_gpu.py` | 258-282 |
| InternVL Prompt 精炼 | `webapp_single_gpu.py` | 286-547 |

### 按论文章节

| 论文章节 | 对应代码 |
|---------|---------|
| Section 3.1（数据处理） | `fastvideo/dataset/t2v_datasets.py` |
| Section 4.1（架构基础） | `wan23/modules/model.py` + `wan23/textimage2video.py` |
| Section 4.2（TSCM） | `wan23/modules/model.py` 486-718 |
| Section 4.3（加速/Self-Forcing） | `fastvideo/distill_model.py` 320-390 + `webapp_single_gpu.py` 836-861 |
| Section 4.3（文本事件控制） | `fastvideo/dataset/t2v_datasets.py` + `webapp_single_gpu.py` |

### 按功能模块

| 功能 | 文件 |
|------|------|
| Stage 1 基础训练 | `train-5b.py` + `scripts/finetune/finetune-5b.sh` |
| Stage 2+3 蒸馏训练 | `fastvideo/distill_model.py` + `scripts/finetune/finetune.sh` |
| Web UI 推理 | `webapp_single_gpu.py` |
| 批量推理 | `fastvideo/sample/sample_5b.py` |
| VAE 定义 | `wan23/modules/vae2_2.py` |
| 数据集加载 | `fastvideo/dataset/t2v_datasets.py` |
| 判别器 | `ADD/models/discriminator.py` |
| 模型配置 | `wan23/configs/wan_ti2v_5B.py` |

---

## 架构总览图

```
YUME 1.5 完整架构
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

输入: 图片/文本 + 键盘(W/A/S/D) + 鼠标(↑↓←→)
           │
           ▼
┌─────────────────────────────┐
│  InternVL3-2B (创新点 5)    │ ← 图片 + 用户 prompt
│  Prompt 精炼（CPU→GPU→CPU）  │
└──────────────┬──────────────┘
               │ final_prompt
               ▼
┌─────────────────────────────┐
│  Action Vocab 映射 (创新点3) │ ← 键盘/鼠标控制
│  W→"camera pushes forward"  │
│  →→"camera pans to right"   │
└──────────────┬──────────────┘
               │ final_prompt（含控制描述）
               ▼
┌─────────────────────────────┐
│  T5 文本编码器（5B）          │
│  Event + Action 分离编码     │
│  Action 部分可预计算缓存      │
└──────────────┬──────────────┘
               │ context embeddings
               ▼
┌──────────────────────────────────────────────┐
│          Wan2.2-5B DiT Backbone              │
│                                              │
│  历史帧 zc ──→ [创新点1A] 时空压缩            │
│               patch_embed_Nx (自适应级别)    │
│           │                                  │
│           └──→ [创新点1B] 通道压缩(8,4,4)    │
│                + Linear Attention 融合        │
│                                              │
│  预测帧 zp ──→ 标准 patch_embed(1,2,2)       │
│                                              │
│  concat(ˆzc, ˆzp) ──→ 40个 DiT Blocks       │
│                      ──→ 输出 noise_pred      │
└──────────────────────────────────────────────┘
               │ 4步 Euler 采样
               ▼
       clean latent (新 chunk)
               │
               ├──→ VAE 解码 → video_tail (像素)
               │
               └──→ [创新点2] Self-Forcing
                    append 到 model_input_latent
                    供下一 chunk 作为历史输入 ✅

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
训练加速（创新点4）:
  ProjectedDiscriminator (DINO ViT) 对抗训练
  50 steps → 4 steps 推理
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

*本文档基于 YUME1.5.pdf + 代码库 YUME/ 综合分析生成*  
*最后更新: 2026年2月*
