# SeedVR2 NCNN Vulkan

本项目正在将 [SeedVR2](https://github.com/ByteDance-Seed/SeedVR) 3B 视频超分模型移植到
[NCNN](https://github.com/Tencent/ncnn)，最终目标是在 C++ 中通过 NCNN Vulkan 后端完成推理。

当前已经跑通 **NCNN CPU 完整推理链路**，包括动态 VAE、32 层 DiT、CFG、Euler sampler
和视频前后处理。Vulkan 运行时框架已经预留，但六个 SeedVR2 自定义层尚未实现
`forward_vkcompute`，因此当前版本不能使用 Vulkan 推理。

## 推理流程

```text
输入视频
  -> 视频预处理
  -> VAE Encoder
  -> 条件 latent 与初始噪声
  -> 32 层 DiT 流式推理
  -> CFG + Euler sampler
  -> VAE Decoder
  -> 视频后处理
  -> 输出视频
```

DiT 的 32 个 block 使用流式加载方式执行。运行时每次只加载一个 block，完成 forward 后
立即释放权重，避免 3B 模型全部常驻内存。

## 已实现内容

### 模型转换

- SeedVR2 3B VAE Encoder/Decoder 已转换为 NCNN `param/bin`。
- DiT 输入层、32 个 Transformer block 和输出层已转换为 NCNN `param/bin`。
- 普通卷积、Convolution3D、InnerProduct 等计算复用 NCNN 原生算子。
- 支持模型约束内的动态 `T/H/W`，不是固定导出尺寸。

### C++ Runtime

- 提供统一的 `SeedVR2Engine` 对外接口。
- 实现视频预处理和后处理。
- 实现 VAE posterior 采样与 latent scaling。
- 实现 32 层 DiT 流式执行。
- 实现 classifier-free guidance。
- 实现 `v_lerp` 预测类型和 Euler sampler。
- 支持预计算 positive/negative 文本 embedding。
- 提供正式 CLI 和分层测试 runner。

### CPU 自定义层

当前共有六个 SeedVR2 自定义层：

1. `DynamicFramewiseGroupNorm`
2. `DynamicFramewiseSpatialAttention`
3. `DynamicSpaceTimeShuffle`
4. `SeedVR2DiTInput`
5. `SeedVR2DiTBlock`
6. `SeedVR2DiTOutput`

这些层的 CPU `forward` 已实现。动态窗口 attention、MM-RoPE、Ada modulation、SwiGLU、
时空 shuffle 等 SeedVR2 特有逻辑目前都在这些层中执行。

### 测试与验证

- VAE Encoder/Decoder 已使用多组动态尺寸输入验证。
- DiT 输入层、单个 block、输出层和完整 32 层流程均有独立 runner。
- 完整 DiT 新旧调度实现对同一输入的输出逐字节一致。
- `SeedVR2Engine` 已使用真实模型完成 CPU 端到端 smoke test。
- PyTorch reference、中间张量和 NCNN 对齐报告保存在 `test_vectors`、`ncnn_models`
  及 `docs` 对应目录中。

## 尚未实现

### Vulkan 后端

当前主要剩余工作是为六个自定义层实现 Vulkan 版本：

- VkMat 输入输出和 pipeline 生命周期。
- GroupNorm reduction 与 affine shader。
- 动态空间 attention。
- 动态窗口划分与 shifted window。
- MM-RoPE。
- QKV、softmax 和 attention 输出。
- Ada modulation、SwiGLU 和残差路径。
- 动态 patch/unpatch 与时空 shuffle。

在全部 Vulkan 自定义层完成并通过 PyTorch/CPU 数值对齐前，Runtime 会明确拒绝 Vulkan
后端，不会静默回退到 CPU 后宣称 Vulkan 可用。

### 应用层功能

- CLI 当前使用 FP32 raw tensor，不直接读取或写入 MP4。
- C++ Runtime 不包含文本编码器，需要外部提供预计算 embedding。
- 当前目标是 SeedVR2 3B、SR 任务、batch size 1。
- 暂未实现音频保留、媒体封装和颜色修复。
- Vulkan 显存卸载和性能优化尚未完成。

## 目录结构

```text
include/seedvr2/engine.h  对外公开 API

src/
  engine.cpp              完整推理调度
  vae.cpp                 VAE Encoder/Decoder
  dit.cpp                 32 层 DiT 流式执行
  sampler.cpp             CFG 和 Euler sampler
  preprocessing.cpp       视频预处理
  postprocessing.cpp      视频后处理
  vulkan_context.cpp      Vulkan/NCNN 运行时配置

custom_layers/            六个 SeedVR2 自定义层
apps/seedvr2_cli.cpp      命令行入口
tests/                    分层和完整推理 runner
tools/                    导出、转换和数值对齐脚本
docs/                     转换及自定义层报告
pytorch_model/            本地可选的 SeedVR2 PyTorch 参考代码（Git 忽略）
```

## 构建

依赖：

- Ubuntu 24.04 / WSL2
- CMake 3.16+
- 支持 C++17 的编译器
- NCNN 源码

```bash
cd /home/czw1/ncnn_learn/seedvr2_ncnn

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DNCNN_SOURCE_DIR=/home/czw1/ncnn_learn/ncnn \
  -DSEEDVR2_ENABLE_VULKAN=OFF

cmake --build build -j8
ctest --test-dir build --output-on-failure
```

当前 Vulkan 自定义层尚未完成，建议保持 `SEEDVR2_ENABLE_VULKAN=OFF`。

## 模型目录

Runtime 默认从下面的结构加载模型：

```text
ncnn_models/
  vae_dynamic/
    seedvr2_vae_encoder_dynamic.ncnn.param
    seedvr2_vae_encoder_dynamic.ncnn.bin
    seedvr2_vae_decoder_dynamic.ncnn.param
    seedvr2_vae_decoder_dynamic.ncnn.bin

  dit_full_fp16/
    seedvr2_dit_input.ncnn.param
    seedvr2_dit_input.ncnn.bin
    seedvr2_dit_block_00.ncnn.param
    seedvr2_dit_block_00.ncnn.bin
    ...
    seedvr2_dit_block_31.ncnn.param
    seedvr2_dit_block_31.ncnn.bin
    seedvr2_dit_output.ncnn.param
    seedvr2_dit_output.ncnn.bin
```

模型权重体积较大，不提交到 Git 仓库。

## C++ 接口

```cpp
#include <seedvr2/engine.h>

seedvr2::RuntimeOptions options;
options.device = seedvr2::DeviceType::Cpu;
options.num_threads = 8;
options.sampling_steps = 1;
options.cfg_scale = 1.0f;

seedvr2::SeedVR2Engine engine;
if (engine.load("/path/to/ncnn_models", options) != 0)
    throw std::runtime_error(engine.last_error());

seedvr2::Video output;
if (engine.process(input, positive, negative, output) != 0)
    throw std::runtime_error(engine.last_error());
```

公开接口使用 FP32、`T,H,W,C` 布局的 RGB 视频，数值范围为 `[0,1]`。文本 embedding
布局为 `[tokens,5120]`。

## 后续计划

1. 实现三个 VAE 动态层的 Vulkan forward。
2. 实现 DiT Input/Output 的 Vulkan forward。
3. 将 DiT block 的 projection、Ada 和 MLP 迁移到 Vulkan。
4. 实现动态窗口 attention、shifted window 和 MM-RoPE。
5. 使用现有 PyTorch 中间张量逐层验证 Vulkan 数值。
6. 接入 MP4 输入输出并进行 Vulkan benchmark。
