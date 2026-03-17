# 🎯 YUME 创新点代码导航（可点击跳转）

> 💡 **使用方法**：在 Cursor/VS Code 中，按住 `Ctrl/Cmd` 点击下方链接即可跳转到对应代码位置

---

## 📑 目录

- [1️⃣ TSCM - 历史帧自适应压缩](#1-tscm---历史帧自适应压缩)
- [2️⃣ Self-Forcing - 模型预测作为历史](#2-self-forcing---模型预测作为历史)
- [3️⃣ Keyboard/Mouse Control - 相机控制](#3-keyboardmouse-control---相机控制)
- [4️⃣ GAN-based Distillation - 加速推理](#4-gan-based-distillation---加速推理)
- [5️⃣ InternVL3-2B - Prompt 增强](#5-internvl3-2b---prompt-增强)

---

## 1️⃣ TSCM - 历史帧自适应压缩

### 🎯 核心实现
- [wan23/modules/model.py:486](wan23/modules/model.py#L486) - patch_embedding 层定义
- [wan23/modules/model.py:597](wan23/modules/model.py#L597) - TSCM 主逻辑开始

### 📂 压缩层级实现

#### 层级 1: ≤ 6 帧 (最少压缩)
- [wan23/modules/model.py:599](wan23/modules/model.py#L599) - 条件判断
- [wan23/modules/model.py:602](wan23/modules/model.py#L602) - 首帧处理 (1x)
- [wan23/modules/model.py:605](wan23/modules/model.py#L605) - 中间帧 (2x)
- [wan23/modules/model.py:609](wan23/modules/model.py#L609) - 最后帧 (1x)

#### 层级 2: ≤ 22 帧
- [wan23/modules/model.py:618](wan23/modules/model.py#L618) - 条件判断
- [wan23/modules/model.py:621](wan23/modules/model.py#L621) - 首帧 (1x)
- [wan23/modules/model.py:624](wan23/modules/model.py#L624) - 老帧 (4x)
- [wan23/modules/model.py:629](wan23/modules/model.py#L629) - 近帧 (2x)
- [wan23/modules/model.py:630](wan23/modules/model.py#L630) - 最新帧 (1x)

#### 层级 3: ≤ 86 帧
- [wan23/modules/model.py:640](wan23/modules/model.py#L640) - 条件判断
- [wan23/modules/model.py:642](wan23/modules/model.py#L642) - 首帧 (1x)
- [wan23/modules/model.py:646](wan23/modules/model.py#L646) - 远古帧 (8x)
- [wan23/modules/model.py:650](wan23/modules/model.py#L650) - 较老帧 (4x)
- [wan23/modules/model.py:651](wan23/modules/model.py#L651) - 近帧 (2x)
- [wan23/modules/model.py:652](wan23/modules/model.py#L652) - 最新帧 (1x)

#### 层级 4: ≤ 342 帧
- [wan23/modules/model.py:664](wan23/modules/model.py#L664) - 条件判断
- [wan23/modules/model.py:666](wan23/modules/model.py#L666) - 首帧 (2x)
- [wan23/modules/model.py:669](wan23/modules/model.py#L669) - 超远古帧 (16x)
- [wan23/modules/model.py:674](wan23/modules/model.py#L674) - 远古帧 (8x)
- [wan23/modules/model.py:675](wan23/modules/model.py#L675) - 较老帧 (4x)
- [wan23/modules/model.py:676](wan23/modules/model.py#L676) - 近帧 (2x)
- [wan23/modules/model.py:677](wan23/modules/model.py#L677) - 最新帧 (1x)

#### 层级 5: > 342 帧 (最大压缩)
- [wan23/modules/model.py:690](wan23/modules/model.py#L690) - 条件判断
- [wan23/modules/model.py:693](wan23/modules/model.py#L693) - 首帧 (2x)
- [wan23/modules/model.py:696](wan23/modules/model.py#L696) - 超超远古帧 (32x = 16x+2x)
- [wan23/modules/model.py:701](wan23/modules/model.py#L701) - 超远古帧 (16x)
- [wan23/modules/model.py:702](wan23/modules/model.py#L702) - 远古帧 (8x)
- [wan23/modules/model.py:703](wan23/modules/model.py#L703) - 较老帧 (4x)
- [wan23/modules/model.py:704](wan23/modules/model.py#L704) - 近帧 (2x)
- [wan23/modules/model.py:705](wan23/modules/model.py#L705) - 最新帧 (1x)

### 🔧 辅助函数
- [wan23/modules/model.py:453](wan23/modules/model.py#L453) - patch_embedding 初始化
- [wan23/modules/model.py:905](wan23/modules/model.py#L905) - patch_embedding 权重初始化

---

## 2️⃣ Self-Forcing - 模型预测作为历史

### 🎯 推理实现 (Web UI)
- [webapp_single_gpu.py:569](webapp_single_gpu.py#L569) - `long_generate()` 函数入口
- [webapp_single_gpu.py:810](webapp_single_gpu.py#L810) - 提取尾部 latent
- [webapp_single_gpu.py:813](webapp_single_gpu.py#L813) - Euler 步更新
- [webapp_single_gpu.py:817](webapp_single_gpu.py#L817) - 拼接新旧 latent
- [webapp_single_gpu.py:836](webapp_single_gpu.py#L836) - **更新历史 latent (核心)**
- [webapp_single_gpu.py:857](webapp_single_gpu.py#L857) - 更新历史视频帧
- [webapp_single_gpu.py:872](webapp_single_gpu.py#L872) - 保存到全局状态

### 🎓 训练实现
- [fastvideo/distill_model.py:197](fastvideo/distill_model.py#L197) - `distill_one_step()` 函数
- [fastvideo/distill_model.py:385](fastvideo/distill_model.py#L385) - 计算 clean latent (x_0)
- [fastvideo/distill_model.py:390](fastvideo/distill_model.py#L390) - **拼接新预测作为历史**

### 🔄 相关逻辑
- [webapp_single_gpu.py:611](webapp_single_gpu.py#L611) - 继续生成模式检查
- [webapp_single_gpu.py:713](webapp_single_gpu.py#L713) - 生成参数传递
- [webapp_single_gpu.py:760](webapp_single_gpu.py#L760) - 视频片段拼接

---

## 3️⃣ Keyboard/Mouse Control - 相机控制

### 🎯 核心实现
- [fastvideo/dataset/t2v_datasets.py:218](fastvideo/dataset/t2v_datasets.py#L218) - `parse_txt_file()` 解析函数
- [fastvideo/dataset/t2v_datasets.py:393](fastvideo/dataset/t2v_datasets.py#L393) - 键盘控制词汇表 (vocab_k)
- [fastvideo/dataset/t2v_datasets.py:407](fastvideo/dataset/t2v_datasets.py#L407) - 鼠标控制词汇表 (vocab_c)
- [fastvideo/dataset/t2v_datasets.py:429](fastvideo/dataset/t2v_datasets.py#L429) - 相机参数附加

### 📂 数据集类
- [fastvideo/dataset/t2v_datasets.py:254](fastvideo/dataset/t2v_datasets.py#L254) - `StableVideoAnimationDataset` 类定义
- [fastvideo/dataset/t2v_datasets.py:289](fastvideo/dataset/t2v_datasets.py#L289) - 数据加载循环
- [fastvideo/dataset/t2v_datasets.py:307](fastvideo/dataset/t2v_datasets.py#L307) - TXT 文件检查
- [fastvideo/dataset/t2v_datasets.py:314](fastvideo/dataset/t2v_datasets.py#L314) - 添加到 vid_meta
- [fastvideo/dataset/t2v_datasets.py:331](fastvideo/dataset/t2v_datasets.py#L331) - `get_sample()` 方法

### 🔧 辅助函数
- [fastvideo/dataset/t2v_datasets.py:235](fastvideo/dataset/t2v_datasets.py#L235) - `parse_txt_frame()` 提取帧范围
- [fastvideo/dataset/t2v_datasets.py:182](fastvideo/dataset/t2v_datasets.py#L182) - `calculate_metrics_in_range()` 计算相机参数

### 🎨 Web UI 集成
- [webapp_single_gpu.py:648](webapp_single_gpu.py#L648) - 键盘控制词汇表
- [webapp_single_gpu.py:660](webapp_single_gpu.py#L660) - 鼠标控制词汇表
- [webapp_single_gpu.py:677](webapp_single_gpu.py#L677) - 拼接相机控制到 prompt

---

## 4️⃣ GAN-based Distillation - 加速推理

### 🎯 核心训练逻辑
- [fastvideo/distill_model.py:532](fastvideo/distill_model.py#L532) - 判别器初始化
- [fastvideo/distill_model.py:320](fastvideo/distill_model.py#L320) - **GAN Loss 计算开始**
- [fastvideo/distill_model.py:321](fastvideo/distill_model.py#L321) - 计算 clean latent (x_0 = x_t - t*ε)
- [fastvideo/distill_model.py:329](fastvideo/distill_model.py#L329) - 判别器前向 (Real)
- [fastvideo/distill_model.py:330](fastvideo/distill_model.py#L330) - 判别器前向 (Fake)
- [fastvideo/distill_model.py:337](fastvideo/distill_model.py#L337) - Hinge Loss (Real)
- [fastvideo/distill_model.py:338](fastvideo/distill_model.py#L338) - Hinge Loss (Fake)
- [fastvideo/distill_model.py:342](fastvideo/distill_model.py#L342) - 判别器损失汇总
- [fastvideo/distill_model.py:343](fastvideo/distill_model.py#L343) - 判别器反向传播
- [fastvideo/distill_model.py:344](fastvideo/distill_model.py#L344) - 梯度裁剪
- [fastvideo/distill_model.py:345](fastvideo/distill_model.py#L345) - 判别器优化器更新
- [fastvideo/distill_model.py:350](fastvideo/distill_model.py#L350) - 生成器 GAN Loss
- [fastvideo/distill_model.py:353](fastvideo/distill_model.py#L353) - Non-saturating loss
- [fastvideo/distill_model.py:354](fastvideo/distill_model.py#L354) - **总损失 = 扩散损失 + 0.01*GAN损失**

### 🏗️ 判别器架构
- [ADD/models/discriminator.py:176](ADD/models/discriminator.py#L176) - `ProjectedDiscriminator` 类定义
- [ADD/models/discriminator.py:185](ADD/models/discriminator.py#L185) - DINO ViT 初始化
- [ADD/models/discriminator.py:186](ADD/models/discriminator.py#L186) - SubPixel 上采样层
- [ADD/models/discriminator.py:190](ADD/models/discriminator.py#L190) - 多尺度判别头
- [ADD/models/discriminator.py:207](ADD/models/discriminator.py#L207) - `forward()` 方法
- [ADD/models/discriminator.py:209](ADD/models/discriminator.py#L209) - latent → RGB
- [ADD/models/discriminator.py:212](ADD/models/discriminator.py#L212) - DINO 特征提取
- [ADD/models/discriminator.py:217](ADD/models/discriminator.py#L217) - 应用判别头

### 🔧 FSDP 配置
- [fastvideo/distill_model.py:555](fastvideo/distill_model.py#L555) - 判别器 FSDP 配置
- [fastvideo/distill_model.py:574](fastvideo/distill_model.py#L574) - FSDP 包装
- [fastvideo/distill_model.py:606](fastvideo/distill_model.py#L606) - 判别器优化器初始化

### ⚙️ 训练参数
- [fastvideo/distill_model.py:829](fastvideo/distill_model.py#L829) - `--Distil` 参数定义
- [fastvideo/distill_model.py:938](fastvideo/distill_model.py#L938) - `--discriminator_learning_rate`
- [fastvideo/distill_model.py:743](fastvideo/distill_model.py#L743) - 判别器条件初始化
- [fastvideo/distill_model.py:777](fastvideo/distill_model.py#L777) - 传递 Distil 标志到训练循环

---

## 5️⃣ InternVL3-2B - Prompt 增强

### 🎯 推理脚本实现
- [fastvideo/sample/sample_5b.py:1288](fastvideo/sample/sample_5b.py#L1288) - 模型路径定义
- [fastvideo/sample/sample_5b.py:1289](fastvideo/sample/sample_5b.py#L1289) - **AutoModel 加载**
- [fastvideo/sample/sample_5b.py:1295](fastvideo/sample/sample_5b.py#L1295) - Tokenizer 加载
- [fastvideo/sample/sample_5b.py:798](fastvideo/sample/sample_5b.py#L798) - I2V prompt 构建
- [fastvideo/sample/sample_5b.py:804](fastvideo/sample/sample_5b.py#L804) - **InternVL 推理 (chat)**
- [fastvideo/sample/sample_5b.py:848](fastvideo/sample/sample_5b.py#L848) - 带用户 prompt 的精炼

### 🌐 Web UI 实现
- [webapp_single_gpu.py:91](webapp_single_gpu.py#L91) - InternVL 路径配置
- [webapp_single_gpu.py:154](webapp_single_gpu.py#L154) - caption_model 全局变量
- [webapp_single_gpu.py:258](webapp_single_gpu.py#L258) - `load_caption_model()` 函数
- [webapp_single_gpu.py:267](webapp_single_gpu.py#L267) - **AutoModel 加载 (BF16)**
- [webapp_single_gpu.py:274](webapp_single_gpu.py#L274) - Tokenizer 加载
- [webapp_single_gpu.py:276](webapp_single_gpu.py#L276) - **模型移到 CPU (节省显存)**

### 🔄 Prompt 精炼逻辑
- [webapp_single_gpu.py:286](webapp_single_gpu.py#L286) - `refine_prompt_from_image()` 函数
- [webapp_single_gpu.py:290](webapp_single_gpu.py#L290) - `dynamic_preprocess()` 图片切分
- [webapp_single_gpu.py:322](webapp_single_gpu.py#L322) - 图片 tiles 预处理
- [webapp_single_gpu.py:325](webapp_single_gpu.py#L325) - **临时移到 GPU**
- [webapp_single_gpu.py:327](webapp_single_gpu.py#L327) - 构建 prompt 精炼问题
- [webapp_single_gpu.py:330](webapp_single_gpu.py#L330) - **InternVL 推理**
- [webapp_single_gpu.py:331](webapp_single_gpu.py#L331) - **用完后移回 CPU**

### 🖼️ 图片预处理
- [webapp_single_gpu.py:290](webapp_single_gpu.py#L290) - `dynamic_preprocess()` 函数
- [webapp_single_gpu.py:304](webapp_single_gpu.py#L304) - 计算最佳切分方案
- [webapp_single_gpu.py:312](webapp_single_gpu.py#L312) - 切分图片为 tiles

### 🎨 System Prompt
- [wan23/utils/system_prompt.py:1](wan23/utils/system_prompt.py#L1) - I2V 中文 prompt
- [wan23/utils/system_prompt.py:15](wan23/utils/system_prompt.py#L15) - I2V 英文 prompt
- [wan23/utils/system_prompt.py:30](wan23/utils/system_prompt.py#L30) - T2V 中文 prompt
- [wan23/utils/system_prompt.py:45](wan23/utils/system_prompt.py#L45) - T2V 英文 prompt

### 🎯 集成点
- [webapp_single_gpu.py:679](webapp_single_gpu.py#L679) - 调用 prompt 精炼
- [fastvideo/sample/sample_5b.py:1328](fastvideo/sample/sample_5b.py#L1328) - 传递 caption_model 到采样

---

## 🗺️ 配置文件导航

### 模型配置
- [wan23/configs/wan_ti2v_5B.py:1](wan23/configs/wan_ti2v_5B.py#L1) - 5B 模型配置
- [wan23/configs/wan_ti2v_5B.py:16](wan23/configs/wan_ti2v_5B.py#L16) - VAE checkpoint 路径
- [wan23/configs/wan_ti2v_5B.py:13](wan23/configs/wan_ti2v_5B.py#L13) - T5 tokenizer 路径
- [wan23/configs/wan_ti2v_5B.py:20](wan23/configs/wan_ti2v_5B.py#L20) - Transformer 维度配置

### VAE 实现
- [wan23/modules/vae2_2.py:884](wan23/modules/vae2_2.py#L884) - `_video_vae()` 函数
- [wan23/modules/vae2_2.py:1036](wan23/modules/vae2_2.py#L1036) - `Wan2_2_VAE` 类定义
- [wan23/modules/vae2_2.py:1062](wan23/modules/vae2_2.py#L1062) - `encode()` 方法
- [wan23/modules/vae2_2.py:1070](wan23/modules/vae2_2.py#L1070) - `decode()` 方法

### 训练脚本
- [scripts/finetune/finetune.sh:1](scripts/finetune/finetune.sh#L1) - 训练启动脚本
- [scripts/finetune/finetune.sh:21](scripts/finetune/finetune.sh#L21) - MVDT 标志
- [scripts/finetune/finetune.sh:22](scripts/finetune/finetune.sh#L22) - Distil 标志
- [scripts/finetune/finetune.sh:24](scripts/finetune/finetune.sh#L24) - root_dir 数据路径
- [scripts/finetune/finetune.sh:25](scripts/finetune/finetune.sh#L25) - full_mp4 Sekai 路径

### 推理脚本
- [scripts/inference/sample_5b.sh:1](scripts/inference/sample_5b.sh#L1) - 5B 推理脚本
- [fastvideo/sample/sample_5b.py:1146](fastvideo/sample/sample_5b.py#L1146) - checkpoint 路径
- [fastvideo/sample/sample_5b.py:1149](fastvideo/sample/sample_5b.py#L1149) - Yume 模型初始化

---

## 🎯 主要类和函数导航

### WanModel (DiT Transformer)
- [wan23/modules/model.py:415](wan23/modules/model.py#L415) - `WanModel` 类定义
- [wan23/modules/model.py:446](wan23/modules/model.py#L446) - `__init__()` 初始化
- [wan23/modules/model.py:568](wan23/modules/model.py#L568) - `forward()` 前向传播
- [wan23/modules/model.py:892](wan23/modules/model.py#L892) - `init_weights()` 权重初始化

### WanAttentionBlock
- [wan23/modules/model.py:47](wan23/modules/model.py#L47) - `WanAttentionBlock` 类定义
- [wan23/modules/model.py:136](wan23/modules/model.py#L136) - `forward()` 方法
- [wan23/modules/model.py:184](wan23/modules/model.py#L184) - RoPE 应用

### Yume 高层 API
- [wan23/textimage2video.py:71](wan23/textimage2video.py#L71) - `Yume` 类定义
- [wan23/textimage2video.py:80](wan23/textimage2video.py#L80) - `__init__()` 初始化
- [wan23/textimage2video.py:164](wan23/textimage2video.py#L164) - `generate()` 方法
- [wan23/textimage2video.py:229](wan23/textimage2video.py#L229) - `t2v()` 方法
- [wan23/textimage2video.py:259](wan23/textimage2video.py#L259) - `i2v()` 方法

### Flow-Matching Scheduler
- [wan23/utils/fm_solvers.py:12](wan23/utils/fm_solvers.py#L12) - `FlowDPMSolverMultistepScheduler` 类
- [wan23/utils/fm_solvers.py:138](wan23/utils/fm_solvers.py#L138) - `step()` 方法
- [fastvideo/distill/solver.py:221](fastvideo/distill/solver.py#L221) - `PCMFMScheduler` 类

### Sequence Parallel
- [wan23/distributed/sequence_parallel.py:165](wan23/distributed/sequence_parallel.py#L165) - `sp_attn_forward()` 函数
- [wan23/distributed/ulysses.py:9](wan23/distributed/ulysses.py#L9) - `distributed_attention()` 函数

---

## 🚀 快速启动

### 启动 Web UI
- [webapp_single_gpu.py:1513](webapp_single_gpu.py#L1513) - Flask app 启动
- [bootstrap.py:1](bootstrap.py#L1) - 项目入口点

### 运行训练
- [fastvideo/distill_model.py:1078](fastvideo/distill_model.py#L1078) - `main()` 函数
- [fastvideo/distill_model.py:747](fastvideo/distill_model.py#L747) - 训练主循环

### 运行推理
- [fastvideo/sample/sample_5b.py:1143](fastvideo/sample/sample_5b.py#L1143) - 推理脚本入口

---

## 📊 数据流程关键节点

### 数据加载流程
1. [fastvideo/dataset/t2v_datasets.py:289](fastvideo/dataset/t2v_datasets.py#L289) - 扫描目录
2. [fastvideo/dataset/t2v_datasets.py:307](fastvideo/dataset/t2v_datasets.py#L307) - 解析 TXT
3. [fastvideo/dataset/t2v_datasets.py:331](fastvideo/dataset/t2v_datasets.py#L331) - 加载视频帧
4. [fastvideo/dataset/t2v_datasets.py:393](fastvideo/dataset/t2v_datasets.py#L393) - 转换控制指令

### 训练流程
1. [fastvideo/distill_model.py:197](fastvideo/distill_model.py#L197) - Batch 加载
2. [fastvideo/distill_model.py:275](fastvideo/distill_model.py#L275) - VAE 编码
3. [fastvideo/distill_model.py:295](fastvideo/distill_model.py#L295) - 添加噪声
4. [fastvideo/distill_model.py:313](fastvideo/distill_model.py#L313) - Transformer 前向
5. [fastvideo/distill_model.py:318](fastvideo/distill_model.py#L318) - 计算扩散损失
6. [fastvideo/distill_model.py:320](fastvideo/distill_model.py#L320) - GAN Loss (可选)
7. [fastvideo/distill_model.py:356](fastvideo/distill_model.py#L356) - 反向传播

### 推理流程
1. [webapp_single_gpu.py:620](webapp_single_gpu.py#L620) - 加载图片/视频
2. [webapp_single_gpu.py:633](webapp_single_gpu.py#L633) - VAE 编码
3. [webapp_single_gpu.py:679](webapp_single_gpu.py#L679) - Prompt 精炼
4. [webapp_single_gpu.py:704](webapp_single_gpu.py#L704) - 生成参数准备
5. [webapp_single_gpu.py:784](webapp_single_gpu.py#L784) - Euler 采样循环
6. [webapp_single_gpu.py:806](webapp_single_gpu.py#L806) - Transformer 推理
7. [webapp_single_gpu.py:830](webapp_single_gpu.py#L830) - VAE 解码
8. [webapp_single_gpu.py:836](webapp_single_gpu.py#L836) - 更新历史 (Self-Forcing)

---

## 💡 使用技巧

### 在 Cursor/VS Code 中
1. **Ctrl/Cmd + 点击** 链接直接跳转到代码位置
2. **Ctrl/Cmd + P** 快速打开文件
3. **Ctrl/Cmd + G** 跳转到指定行号
4. **Ctrl/Cmd + Shift + O** 查看文件大纲

### 调试流程
1. 在关键位置设置断点（点击行号左侧）
2. 使用调试器逐步执行
3. 查看变量值和调用栈

### 代码搜索
- **Ctrl/Cmd + Shift + F** 全局搜索
- 搜索函数名/类名快速定位

---

## 🎓 学习路径推荐

### 初学者
1. 先看 [Keyboard/Mouse Control](#3-keyboardmouse-control---相机控制) (最简单)
2. 再看 [InternVL3-2B](#5-internvl3-2b---prompt-增强) (独立模块)
3. 然后看 [Self-Forcing](#2-self-forcing---模型预测作为历史) (核心逻辑)

### 进阶学习
1. 深入 [TSCM](#1-tscm---历史帧自适应压缩) (最复杂)
2. 研究 [GAN Distillation](#4-gan-based-distillation---加速推理) (训练技巧)

### 实践建议
1. 运行 Web UI，观察各个模块如何协同工作
2. 修改参数（如压缩比率、GAN loss 权重），观察效果
3. 在关键位置添加 print 语句，理解数据流

---

**最后更新**: 2026-02-10  
**文档版本**: v1.0  
**维护者**: YUME Team
