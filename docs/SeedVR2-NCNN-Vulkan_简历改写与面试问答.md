# SeedVR2-NCNN/Vulkan 简历改写与面试问答

## 一、简历项目经历精简版

**SeedVR2-NCNN/Vulkan 推理部署** ｜ 腾讯犀牛鸟开源项目  

**项目简介：** 将 SeedVR2 3B 视频超分/修复模型从 PyTorch 迁移到 NCNN 推理框架，构建纯 C++ 端到端 Runtime，并基于 Vulkan Compute 实现动态算子 GPU 推理。

**技术栈：** C++17 / NCNN / Vulkan Compute / GLSL / VAE / DiT / Window Attention

- **Runtime 设计：** 封装 `SeedVR2Engine`，串联视频预处理、VAE 编解码、文本条件、Euler Sampler、DiT 推理与后处理，提供 C++ API 与 CLI。
- **显存优化：** 针对 3B DiT 权重体积大的问题，设计 32 个 Transformer Block 的流式/常驻双模式加载；默认逐层 load-forward-destroy，仅保留当前 Block 权重，峰值显存约 2.3GB；`--resident` 模式常驻权重，显存约 14.5GB，DiT 单步约 66 倍加速。
- **自定义层实现：** 实现 `DynamicFramewiseGroupNorm`、`DynamicFramewiseSpatialAttention`、`DynamicSpaceTimeShuffle`、`SeedVR2DiTInput/Block/Output` 等 6 个 NCNN 自定义层，支持动态帧数与动态分辨率。
- **Vulkan 算子：** 基于 `VkMat`、`Pipeline` 与 compute shader，将 GroupNorm reduce、QKV/Attention、softmax、Ada Modulation、SwiGLU、patchify/unpatchify、时空 shuffle 等计算迁移到 GPU，减少 CPU/GPU 数据往返。
- **精度与验证：** 构建 PyTorch 参考导出、分层 runner 与端到端 profiler，完成 30 组自定义算子数值对齐，最大误差不超过 `2e-5`，支持本地完整链路验证。

## 二、面试问题与可背诵回答

### 1. 项目简介

**问题：你这个项目主要做了什么？为什么有价值？**

**回答框架：背景 - 目标 - 难点 - 结果。**

1. 背景：SeedVR2 是 3B 规模的视频超分/修复模型，官方主要是 PyTorch 推理，部署依赖重，显存要求高，不适合普通 C++ 应用直接集成。
2. 目标：我的目标是把它迁移到 NCNN，把推理链路做成纯 C++ Runtime，并用 Vulkan Compute 跑在消费级显卡上。
3. 难点：这个模型不是常规 CNN，里面有 VAE、DiT Transformer、动态帧数/分辨率、窗口注意力、patch/unpatch 和很多 PyTorch 动态 reshape 逻辑，不能直接靠 pnnx 静态导出。
4. 我的工作：我拆分模型结构，导出 VAE 和 32 个 DiT Block，补齐 NCNN 不支持的动态算子，并实现 C++ API、CLI、Vulkan 后端和测试 runner。
5. 结果：项目可以不依赖 PyTorch 完成端到端视频修复；流式模式把显存压到约 2.3GB，常驻模式在显存充足时把 DiT 单步从约 22s 降到约 0.33s。

### 2. 技术栈

**问题：为什么选择 NCNN + Vulkan，而不是 TensorRT 或 ONNX Runtime？**

**回答框架：部署目标 - 框架特性 - 项目适配 - 代价。**

1. 部署目标：这个项目更偏端侧和跨平台 C++ 部署，不希望依赖 Python Runtime，也不希望绑定 NVIDIA 专有生态。
2. NCNN 优势：NCNN 是轻量 C++ 推理框架，模型参数格式简单，自定义层机制清晰，Vulkan 后端可以覆盖 NVIDIA、AMD、Intel 等 GPU。
3. 项目适配：SeedVR2 里有大量动态 shape 和自定义语义，NCNN 允许我把大矩阵乘复用原生 `InnerProduct`，把动态逻辑封装进自定义层。
4. TensorRT 取舍：TensorRT 性能可能更强，但插件开发和动态 shape 适配成本高，也偏 NVIDIA。这个项目的核心目标是开源、轻量、跨平台。
5. 总结：所以我选择 NCNN + Vulkan，重点解决“能部署、能跑通、能控制显存、能跨 GPU”的问题。

### 3. Runtime 设计

**问题：`SeedVR2Engine` 是怎么组织端到端推理链路的？**

**回答框架：输入 - 阶段 - 模块边界 - 输出。**

1. 输入：对外只暴露 `Video`、`TextEmbedding` 和 `RuntimeOptions`，视频是 THWC、RGB、FP32、`[0,1]`。
2. 预处理：先做中心裁剪和时间 padding，因为 VAE 要求帧数满足 `(T-1)%4==0`。
3. VAE：先把视频编码成 latent，再构造 diffusion 初始噪声和 condition latent。
4. 条件：正负文本 embedding 可以由调用方传入，也可以读取模型目录里的默认 embedding。
5. DiT：将 noise、condition 和 mask 拼成 33 通道输入，经 DiT input、32 个 block、DiT output 得到预测 latent。
6. Sampler：用 Euler Sampler 做一步或多步更新；如果 `cfg_scale > 1`，会额外跑负向条件并做 CFG。
7. 后处理：VAE decoder 解码回视频，再裁掉 padding 帧，输出 THWC RGB 数据。
8. 设计点：Runtime 用 PIMPL 隐藏 NCNN 细节，对外保持稳定 C++ API，错误通过 `last_error()` 带阶段返回。

### 4. 显存优化

**问题：你简历里写的显存优化具体怎么做的？代价是什么？**

**回答框架：问题 - 方案 - 实现 - 数据 - trade-off。**

1. 问题：SeedVR2 的 DiT 是 3B 参数，32 个 Transformer Block 权重合计约 6.4GB。如果全部常驻，再加激活、VAE 和 workspace，普通显卡很容易爆显存。
2. 方案：我把 DiT 拆成 input head、32 个 block、output head。input/output 常驻；32 个大 block 支持两种策略：默认流式、可选常驻。
3. 流式实现：在 `forward` 里按 block 下标循环，每次创建一个 NCNN `Net`，加载当前 block 的 param/bin，执行 `load-forward-destroy`，层间只传 `video_tokens`、`text_tokens`、`embedding` 和 `patched_shape`。
4. 常驻实现：打开 `--resident` 后，在 `load` 阶段一次性把 32 个 block 的 `Net` 加入 `blocks_`，推理时直接复用，避免每步反复读 6.4GB 权重。
5. 数据：默认流式模式峰值显存约 2.3GB，但 DiT 单步约 22s；常驻模式显存约 14.5GB，但 DiT 单步约 0.33s，大约 66 倍加速。
6. 取舍：流式适合低显存设备，是“空间换时间”；常驻适合 16GB 以上显卡，是“显存换吞吐”。我把它做成 RuntimeOptions 和 CLI 参数，让部署时按硬件选择。
7. 补充：激活显存也被控制在逐层传递范围内，没有把 32 层中间结果全部保存；workspace 用 NCNN allocator 统一管理。

### 5. 自定义层

**问题：为什么要写 6 个自定义层，不能直接用 NCNN 原生算子吗？**

**回答框架：动态语义 - 原生限制 - 拆分原则 - 举例。**

1. 核心原因：SeedVR2 里很多逻辑依赖运行时的 `T/H/W`，而 pnnx trace 往往会把 shape 固化；NCNN 原生算子也表达不了一些逐帧统计、时空重排和窗口 attention 语义。
2. 拆分原则：纯静态、规则的计算尽量用原生算子；动态归约、动态索引、窗口 attention、patch/unpatch 这些才做自定义层。
3. GroupNorm：PyTorch 是逐帧 GroupNorm，统计范围是单帧内的组；NCNN 原生 GroupNorm 会把时间维也统计进去，语义不对。
4. SpatialAttention：VAE bottleneck 要每帧独立做空间 attention，不能把所有帧摊平成一个序列，否则帧间会互相注意。
5. SpaceTimeShuffle：VAE decoder 要做通道到时间和空间的重排，还要删除首帧复制位，原生 PixelShuffle 只支持二维空间。
6. DiTInput/Output：负责动态 patchify/unpatchify，以及 timestep/Ada embedding 的边界处理。
7. DiTBlock：封装完整多模态 Transformer Block，包括窗口注意力、RoPE、Ada Modulation、SwiGLU 和残差。
8. 总结：自定义层不是为了重写所有算子，而是把“NCNN 原生表达不了的动态语义”封起来。

### 6. Vulkan 算子

**问题：你具体把哪些计算放到了 Vulkan 上？怎么保证不频繁回 CPU？**

**回答框架：数据结构 - 管线 - 算子 - 减少同步。**

1. 数据结构：CPU 路径用 `ncnn::Mat`，GPU 路径用 `ncnn::VkMat`。自定义层同时实现 `forward(Mat)` 和 `forward(VkMat)`。
2. 管线：在 `create_pipeline` 里把 GLSL compute shader 编译成 SPIR-V，并创建 `ncnn::Pipeline`；在 `forward(VkMat)` 里绑定输入、输出、权重和常量后 dispatch。
3. 已迁移计算：逐帧 GroupNorm 的 reduce/apply、VAE attention 的 norm/QKV/attention/output projection、SpaceTimeShuffle、DiT input patchify、DiT block 的 RMSNorm、QKV prepare、window attention、Ada residual、SwiGLU、DiT output 的 norm/unpatchify。
4. 大矩阵乘：像 QKV、MLP、projection 这类矩阵乘尽量复用 NCNN 原生 `InnerProduct` 的 Vulkan 实现，不自己手写 GEMM。
5. 减少同步：中间张量尽量保持在 `VkMat`，使用 `workspace_vkallocator` 管临时 buffer，只有很小的 shape 元信息需要 CPU 读取。
6. 结果：这样避免了“某一层回 CPU，导致 GPU/CPU 来回搬数据”的问题，VAE 和 DiT 才能形成连续 GPU 链路。

### 7. Window Attention

**问题：DiT Block 里的窗口注意力是怎么实现的？显存上有什么优化？**

**回答框架：窗口划分 - QKV - RoPE - softmax - 显存。**

1. 窗口划分：视频 token 根据运行时 `T/H/W` 划成窗口，每个窗口内部做 attention，文本 token 和视频 token 一起参与多模态交互。
2. QKV：先做 RMSNorm 和 Ada 调制，再通过线性层得到 Q/K/V；视频和文本分支有各自的权重或共享权重逻辑。
3. RoPE：对视频 Q/K 加 MM-RoPE，让 attention 感知时空位置。
4. softmax：attention 内部按窗口计算 `QK^T / sqrt(d)`，再 softmax，再乘 V。
5. 显存优化：没有物化完整 scores 矩阵长期保存，而是采用重算/分阶段方式降低 `O(seq^2)` 中间显存。代码文档里也提到这是典型的时间换空间，未来可继续向 flash-attention 式 online softmax 优化。
6. 面试总结：这部分最核心的是保持动态窗口语义正确，同时避免 scores 矩阵和中间激活把显存撑爆。

### 8. 数值验证

**问题：你怎么证明 NCNN/Vulkan 结果和 PyTorch 是对齐的？**

**回答框架：参考输出 - 分层测试 - 边界覆盖 - 指标。**

1. 参考输出：先用 PyTorch 导出权重和参考输入输出，把每个自定义层的基准结果固定下来。
2. 分层测试：为 GroupNorm、SpatialAttention、SpaceTimeShuffle、DiTInput、DiTBlock、DiTOutput 分别写 runner，CPU 与 Vulkan 都和参考逻辑对齐。
3. 覆盖边界：测试覆盖单帧/多帧、非 4 对齐宽高、非方阵、不同 channel/group、不同 attention token 数、patch/unpatch shape 等情况。
4. 指标：README 中记录 30 组自定义算子用例全部 PASS，最大误差不超过 `2e-5`。
5. 端到端：另外有 e2e profiler 和动态 VAE 报告，验证从视频预处理、VAE、DiT 到后处理的完整链路。
6. 经验：这类迁移不能只看最终图像，必须先做分层数值对齐，否则最后很难定位误差来自哪个动态算子。

### 9. 视频预处理与 VAE

**问题：视频进模型前为什么要做 padding 和裁剪？**

**回答框架：VAE 结构约束 - 处理方式 - 输出恢复。**

1. VAE 约束：SeedVR2 的因果 VAE 有时间下采样结构，要求输入帧数满足 `(T-1)%4==0`。
2. 时间处理：如果帧数不满足，就复制最后一帧补齐到 `4n+1`，这样不会引入突兀的新内容。
3. 空间处理：空间上做 DivisibleCrop 风格的中心裁剪，让尺寸满足模型下采样和 patchify 约束，不偷偷 resize。
4. 输出恢复：VAE decoder 和后处理阶段会裁掉 padding 帧，并把输出限制回 `[0,1]`。
5. 价值：这样 Runtime 可以支持动态输入视频，同时保持和官方 PyTorch 预处理语义一致。

### 10. C++ API 与 CLI

**问题：你怎么把研究代码变成可部署的工程接口？**

**回答框架：封装 - 参数 - 错误 - 易用性。**

1. 封装：对外只暴露 `include/seedvr2/engine.h`，内部 NCNN、VAE、DiT、自定义层都通过 PIMPL 隐藏。
2. 参数：`RuntimeOptions` 统一管理设备、线程数、采样步数、CFG、seed、Vulkan device、FP16 和 `dit_resident`。
3. 错误：所有阶段失败返回 `Status`，详细信息放在 `last_error()`，便于调用方定位是模型加载、VAE、DiT 还是后处理失败。
4. CLI：命令行支持输入输出视频、模型目录、steps、cfg、seed、threads、`--resident` 等参数，并处理 ffmpeg 解码/编码和音频保留。
5. 默认 embedding：模型目录可以放默认正负文本 embedding，用户不传文本条件也能开箱即用。

### 11. 项目结果

**问题：这个项目最终有什么可量化结果？**

**回答框架：功能 - 精度 - 显存 - 性能。**

1. 功能：实现了不依赖 PyTorch 的纯 C++ 端到端 SeedVR2 视频修复 Runtime。
2. 精度：6 个自定义层和 Vulkan 后端通过 30 组数值对齐，最大误差不超过 `2e-5`。
3. 显存：默认流式模式峰值显存约 2.3GB，让低显存设备也能跑；常驻模式约 14.5GB。
4. 性能：在测试配置下，DiT 单步流式约 22s，常驻约 0.33s，约 66 倍加速。
5. 工程：提供构建脚本、模型下载脚本、CLI、C++ API、测试 runner 和中文实现文档。

## 三、显存优化栏推荐背诵版

面试官问到显存优化时，可以直接背这一版：

> SeedVR2 的核心问题是 DiT 有 3B 参数，32 个 Transformer Block 权重很大。如果直接全部常驻，显存压力会非常高。所以我把 DiT 拆成 input head、32 个 block 和 output head。input/output 比较小，保持常驻；32 个大 block 做成两种加载策略。默认流式模式下，每次只加载当前 block，执行完就释放，block 之间只传 video token、text token、embedding 和动态 shape，这样峰值显存可以压到约 2.3GB。缺点是每个采样步都要重新读约 6.4GB 权重，所以速度慢。显存充足时可以打开 `--resident`，在 load 阶段一次性加载 32 个 block，避免反复 IO 和权重上传，显存约 14.5GB，但 DiT 单步从约 22 秒降到约 0.33 秒，约 66 倍加速。这个优化本质是把权重驻留策略暴露成部署选项，让低显存和高吞吐场景都能覆盖。

