# SeedVR2 NCNN Vulkan

[SeedVR2](https://github.com/ByteDance-Seed/SeedVR) 3B 视频修复模型的 NCNN + Vulkan 实现。

运行时为纯 C++，不依赖 Python、PyTorch 或 CUDA Toolkit。命令行程序可直接完成视频解码、模型推理、H.264 编码和音频保留。

> 当前仅支持 Linux/WSL2 x86_64，并只在 NVIDIA Vulkan GPU 上验证。

## 效果

同一段 5 帧视频的输入与 NCNN Vulkan 输出：

![Big Buck Bunny 输入与 SeedVR2 输出对比](docs/assets/seedvr2_demo_comparison.gif)

| 输入 | 输出 |
|---|---|
| [480x270 MP4](docs/assets/seedvr2_demo_input.mp4) | [480x256 MP4](docs/assets/seedvr2_demo_output.mp4) |

输出高度从 270 变为 256，是因为模型会将宽高中心裁剪到 16 的整数倍。

## 环境要求

- Ubuntu 22.04/24.04 或 WSL2 Ubuntu
- 支持 Vulkan 的 NVIDIA GPU
- CMake 3.16+ 和 C++17 编译器
- Git、curl、ffmpeg、Vulkan headers/loader
- 约 7.3GB 模型磁盘空间
- 推荐 RTX 5090 32GB 或同等级显存；显存较小时使用默认 streaming 模式

AMD、Intel GPU 和 Windows 原生环境尚未验证。

## 快速开始

```bash
sudo apt update
sudo apt install -y build-essential cmake git curl libvulkan-dev vulkan-tools ffmpeg

git clone https://github.com/czwdysj/seedvr2-ncnn-valkan.git
cd seedvr2-ncnn-valkan

./build.sh
./download-models.sh

./build/seedvr2-ncnn-vulkan -i input.mp4 -o output.mp4
```

32GB 显存设备可以让全部 DiT block 常驻显存：

```bash
./build/seedvr2-ncnn-vulkan -i input.mp4 -o output.mp4 --resident
```

模型来自 [vvzc/seedvr2-ncnn-models](https://huggingface.co/vvzc/seedvr2-ncnn-models)。下载脚本支持断点续传和 SHA256 校验，默认先尝试 Hugging Face 镜像。

使用官方 Hugging Face 端点：

```bash
SEEDVR2_HF_ENDPOINT=https://huggingface.co ./download-models.sh
```

## 命令行参数

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

查看各阶段耗时：

```bash
SEEDVR2_PROFILE=1 ./build/seedvr2-ncnn-vulkan \
  -i input.mp4 -o output.mp4 --resident
```

## 当前支持范围

| 项目 | 状态 |
|---|---|
| 模型 | SeedVR2 3B，NCNN fp16 权重 |
| 输入 | ffmpeg 可解码的视频，RGB，宽高至少 16 |
| 空间尺寸 | 动态宽高，中心裁剪到 16 的整数倍 |
| 视频帧数 | 自动复制末帧到 `4n+1`，输出裁回原帧数 |
| 文本条件 | 使用模型内置的官方正/负 embedding |
| Vulkan 数据流 | VAE、DiT、CFG/Euler 的大型中间张量保持为 `VkMat` |
| DiT 模式 | streaming 或 `--resident` |
| 长视频 | 尚未实现时序分块，当前会把整段视频读入内存 |

当前 CLI 不包含 tokenizer，也不支持用户输入自定义 prompt。长视频、超大分辨率视频仍可能因为宿主内存或显存不足而失败。

## 从源码构建

`build.sh` 会初始化锁定版本的 ncnn/glslang 子模块、应用项目补丁、构建 Release 程序并运行轻量测试：

```bash
BUILD_DIR=build-release BUILD_TYPE=Release JOBS=16 ./build.sh
```

只验证 CPU 构建：

```bash
SEEDVR2_ENABLE_VULKAN=OFF SEEDVR2_SKIP_GPU_CHECK=1 ./build.sh
```

校验已下载模型：

```bash
./download-models.sh --verify-only
```

安装 C++ 静态库：

```bash
cmake --install build --prefix ./dist
```

外部 CMake 项目：

```cmake
find_package(seedvr2_ncnn CONFIG REQUIRED)
target_link_libraries(your_app PRIVATE SeedVR2::seedvr2_ncnn)
```

公开 C++ API 位于 [`include/seedvr2/engine.h`](include/seedvr2/engine.h)。

## 文档

- [项目结构与长视频开发接手指南](docs/SeedVR2长视频开发接手指南.md)
- [PyTorch 到 NCNN 转换报告](docs/seedvr2_pytorch_to_ncnn_conversion_report_zh.md)
- [自定义层总览](docs/custom_layers/overview_zh.md)
- [Attention 精度验证](docs/attention_accuracy_validation_5090_zh.md)
- [VAE 3D 卷积优化](docs/v2.0开发优化/VAE_3D卷积优化详解.md)

## 许可

项目代码使用 [Apache-2.0](LICENSE)。第三方项目和模型许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 致谢

- [ByteDance-Seed/SeedVR](https://github.com/ByteDance-Seed/SeedVR)
- [Tencent/ncnn](https://github.com/Tencent/ncnn)
- [nihui/zimage-ncnn-vulkan](https://github.com/nihui/zimage-ncnn-vulkan)
