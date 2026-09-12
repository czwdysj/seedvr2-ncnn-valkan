# seedvr2-ncnn-vulkan

[SeedVR2](https://github.com/ByteDance-Seed/SeedVR) 3B 一步视频修复模型的
NCNN + Vulkan 推理实现。运行时是纯 C++，不需要 Python、PyTorch 或 CUDA Toolkit；
命令行程序负责视频解码、模型推理、视频编码和音频保留。

> 本项目只发布和支持 **Linux/WSL2 x86_64** 构建，不提供 Windows 原生构建。

> 当前可用范围：短视频整段推理已经在 Ubuntu 22.04/24.04、WSL2 和 NVIDIA RTX 5090
> 上完成真实 MP4 闭环验证。AMD/Intel Vulkan 尚未验证；长视频时间分块尚未实现，
> 当前版本会把整段视频解码到内存，请先使用短片段。

## 快速开始

Ubuntu 22.04/24.04 或 WSL2：

```bash
sudo apt update && sudo apt install -y build-essential cmake git curl libvulkan-dev vulkan-tools ffmpeg
git clone https://github.com/czwdysj/seedvr2-ncnn-valkan.git
cd seedvr2-ncnn-valkan
./build.sh && ./download-models.sh
./build/seedvr2-ncnn-vulkan -i input.mp4 -o output.mp4 --resident
```

`download-models.sh` 下载约 7.3GB 权重。`--resident` 适合 32GB 显存的 RTX 5090；
显存不足时去掉该参数，程序会逐块加载 DiT，显存更低但速度明显更慢。

权重仓库：[vvzc/seedvr2-ncnn-models](https://huggingface.co/vvzc/seedvr2-ncnn-models)

## 真实视频效果

下面是同一段 5 帧视频的输入与 NCNN Vulkan 输出。动画中的输入画面按照实际预处理
规则从 480x270 中心裁剪为 480x256，因此左右两侧具有完全相同的视野。

![Big Buck Bunny 输入与 SeedVR2 NCNN Vulkan 输出对比](docs/assets/seedvr2_demo_comparison.gif)

| 原始输入 | NCNN Vulkan 输出 |
|---|---|
| [下载 480x270 输入 MP4](docs/assets/seedvr2_demo_input.mp4) | [下载 480x256 输出 MP4](docs/assets/seedvr2_demo_output.mp4) |

运行条件：RTX 5090、Vulkan resident、1 step、5 帧、25 FPS；模型推理代码基于
`v0.1.1`（`e198470`）。端到端模型阶段耗时 11.526 秒，运行时记录为一次集中上传、
零次中间下载和一次最终下载。演示素材截取自 *Big Buck Bunny*，版权归 Blender
Foundation，按 [CC BY 3.0](https://peach.blender.org/about/) 使用。

## 已实现的数据流

```text
MP4/其他视频
  -> ffmpeg RGB24 解码
  -> 中心裁剪到宽高 16 的整数倍
  -> 时间补齐到 4n+1（推理后裁回原帧数）
  -> VAE Encoder
  -> 33 通道条件构造
  -> DiT Input + 32 Transformer Blocks + DiT Output
  -> CFG / Euler sampler
  -> VAE Decoder
  -> RGB24 + H.264/AAC MP4 编码
```

Vulkan 路径中，VAE、DiT 和 sampler 的大型中间张量使用 `VkMat` 串联。一次
`Engine::process()` 只有输入集中上传和最终视频下载，不在 32 个 DiT block 之间
下载 feature tensor。

## 支持范围

| 项目 | 当前状态 |
|---|---|
| 模型 | SeedVR2 3B，fp16 NCNN 权重 |
| 任务 | 默认文本条件下的一步视频修复 |
| 输入 | ffmpeg 可解码的视频，RGB，任意宽高（至少 16） |
| 空间规则 | 宽高中心裁剪到 16 的整数倍，不做隐藏缩放 |
| 时间规则 | 任意正帧数；内部复制末帧到 `4n+1`，输出裁回原帧数 |
| 后端 | NVIDIA Vulkan 已验证；CPU 可构建但不适合完整 3B 推理 |
| 文本 | 内置官方正/负 embedding；当前 CLI 不包含 tokenizer |
| 长视频 | 尚未实现分块，宿主内存和显存随帧数增长 |

## 构建系统

`build.sh` 完成以下操作：

1. 检查 Git、C++17、CMake 3.16、ffmpeg、Vulkan headers/loader 和硬件 GPU。
2. 浅克隆仓库锁定的 ncnn 与 glslang 子模块。
3. 逐个判断 ncnn 补丁是“未应用”“已应用”还是“冲突”。
4. 配置并编译 `seedvr2-ncnn-vulkan`、静态库和测试 runner。
5. 自动运行不依赖大权重的轻量 CTest；测试未通过则构建失败。

构建参数通过环境变量配置：

```bash
BUILD_DIR=build-release BUILD_TYPE=Release JOBS=16 ./build.sh
```

仅做 CPU 编译验证：

```bash
SEEDVR2_ENABLE_VULKAN=OFF SEEDVR2_SKIP_GPU_CHECK=1 ./build.sh
```

项目包含两个必须应用到固定 ncnn 提交的补丁：`Convolution3D` Vulkan pack1 后端，
以及多核 Linux CPU cache bitmap 解析修复。构建后 `git status` 显示 ncnn 工作树有
修改是预期行为；再次执行 `build.sh` 会识别补丁已经应用。

安装到自定义前缀：

```bash
cmake --install build --prefix ./dist
```

外部 CMake 项目只需链接导出的目标：

```cmake
find_package(seedvr2_ncnn CONFIG REQUIRED)
target_link_libraries(your_app PRIVATE SeedVR2::seedvr2_ncnn)
```

配置外部项目时通过 `-DCMAKE_PREFIX_PATH=/path/to/seedvr2/dist` 指向安装目录。

## 模型下载和校验

```bash
./download-models.sh
./download-models.sh --verify-only
```

下载流程使用 `.part` 文件断点续传，并根据权重仓库发布的 `SHA256SUMS` 校验全部
DiT/VAE 文件。哈希正确后才原子替换正式文件；损坏的续传内容会自动从头重试一次。
默认优先使用 `hf-mirror.com`，失败后自动回退 Hugging Face 官方端点。

海外用户可以直接指定官方端点：

```bash
SEEDVR2_HF_ENDPOINT=https://huggingface.co ./download-models.sh
```

自定义模型目录：

```bash
SEEDVR2_MODEL_DIR=/data/seedvr2-models ./download-models.sh
./build/seedvr2-ncnn-vulkan -i input.mp4 -o output.mp4 --models /data/seedvr2-models
```

完整目录为：

```text
models/
├── dit_full_fp16/
│   ├── seedvr2_dit_input.ncnn.param/bin
│   ├── seedvr2_dit_block_00..31.ncnn.param/bin
│   ├── seedvr2_dit_output.ncnn.param/bin
│   ├── manifest.json
│   └── SHA256SUMS
├── vae_dynamic/
│   ├── seedvr2_vae_encoder_dynamic.ncnn.param/bin
│   ├── seedvr2_vae_decoder_dynamic.ncnn.param/bin
│   └── SHA256SUMS
├── default_pos_emb.bin
└── default_neg_emb.bin
```

## CLI

```console
usage: seedvr2-ncnn-vulkan -i input.mp4 -o output.mp4 [options]
  -i, --input PATH   输入视频
  -o, --output PATH  输出 MP4
  --models DIR       模型目录，默认 ./models
  --steps N          采样步数，默认 1
  --cfg X            CFG 强度，默认 1.0
  --seed S           随机种子，默认 666
  --threads N        CPU 线程数，默认 8
  -g, --gpu N        Vulkan 设备编号，默认 0
  --resident         32 个 DiT block 常驻显存
  -h, --help         显示帮助
```

输入和输出路径可以包含空格。输出父目录不存在时会自动创建。为兼容不同输入容器，
输出统一编码为 H.264 视频和 AAC 音频；无音频输入仍会正常生成视频。

阶段耗时诊断：

```bash
SEEDVR2_PROFILE=1 ./build/seedvr2-ncnn-vulkan \
  -i input.mp4 -o output.mp4 --resident
```

## 正确性证据

当前精度门禁以“排除实现错误”为目标，不要求不同浮点后端逐元素一致：

- 真实 Block 0、9、10、31 正负文本分支 Vulkan/PyTorch 最大 NRMSE `< 8e-4`。
- 完整 DiT 的 320x240 和 480x270 案例 NRMSE 为 `0.008-0.018`，cosine `>= 0.999837`。
- resident 与 streaming 的完整 DiT 输出一致。
- RTX 5090 上 320x240、5 帧 MP4 已完成 CLI 解码到编码闭环。
- 非 `4n+1` 的 6 帧输入已验证最终仍输出 6 帧。

详细数据见 [完整尺寸 Attention 精度报告](docs/attention_accuracy_validation_5090_zh.md)。

## C++ 接口

对外只暴露 [include/seedvr2/engine.h](include/seedvr2/engine.h)：

```cpp
#include "seedvr2/engine.h"
#include <stdexcept>

seedvr2::RuntimeOptions options;
options.device = seedvr2::DeviceType::Vulkan;
options.dit_resident = true;

seedvr2::SeedVR2Engine engine;
if (engine.load("models", options) != 0)
    throw std::runtime_error(engine.last_error());

seedvr2::Video input = /* THWC, RGB, FP32, [0,1] */;
seedvr2::Video output;
if (engine.process(input, {}, {}, output) != 0)
    throw std::runtime_error(engine.last_error());
```

空文本 embedding 会使用模型目录中的默认条件。应用程序不需要包含 NCNN 或自定义层
头文件；它们通过 `SeedVR2Engine` 的 PIMPL 实现隐藏。

## 常见问题

### `no hardware Vulkan GPU was found`

运行 `vulkaninfo --summary`。WSL2 NVIDIA 用户应更新 Windows NVIDIA 驱动，不要在
WSL 内额外安装会覆盖透传库的桌面显卡驱动。软件设备 llvmpipe/lavapipe 不支持推理。

### ncnn patch 冲突

不要单独更新 ncnn 子模块。恢复仓库锁定提交后重新构建：

```bash
git submodule update --init --recursive --force
./build.sh
```

该命令会丢弃 ncnn 子模块中的手工修改，只应在确认没有自己修改 ncnn 后执行。

### 下载中断或校验失败

直接再次运行 `./download-models.sh`。正确文件会跳过，`.part` 会继续下载；续传哈希
不正确时脚本会清除该临时文件并重新获取，不覆盖已有的正确模型。

### OOM

先去掉 `--resident`。当前版本尚无 VAE/时间分块，大分辨率或长视频仍可能 OOM；不要
把“动态尺寸模型”理解为任意尺寸都能在固定显存中运行。

## 项目结构

```text
include/seedvr2/  稳定公开 Engine API
src/core/         Engine 调度和 Vulkan 资源上下文
src/model/        VAE、DiT、sampler
src/layers/       六类 SeedVR2 自定义层及 Vulkan shader
src/pipeline/     视频预处理和后处理
src/cli/          正式视频命令行入口
tests/            无权重冒烟测试和分层数值 runner
tools/            PyTorch 导出与参考张量工具
scripts/          模型和发布辅助检查
patches/          固定 ncnn 提交的兼容补丁
docs/             中文实现、精度和开发报告
```

## 许可

项目代码使用 Apache-2.0，见 [LICENSE](LICENSE)。第三方项目及模型许可见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 致谢

- [ByteDance-Seed/SeedVR](https://github.com/ByteDance-Seed/SeedVR)
- [Tencent/ncnn](https://github.com/Tencent/ncnn)
- [nihui/zimage-ncnn-vulkan](https://github.com/nihui/zimage-ncnn-vulkan)
