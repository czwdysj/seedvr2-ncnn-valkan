# seedvr2-ncnn-vulkan

[SeedVR2](https://github.com/ByteDance-Seed/SeedVR) 3B 一步视频修复（超分/去噪/去压缩）
模型的 **ncnn + Vulkan** 移植版。纯 C++ 推理，不依赖 PyTorch —— clone、下权重、
一条命令，在自己的显卡上跑视频修复。

```bash
git clone --recursive https://github.com/czwdysj/seedvr2-ncnn-valkan.git
cd seedvr2-ncnn-vulkan
./build.sh                 # 一键构建
./download-models.sh       # 下载 7.5GB ncnn 权重（默认走国内 hf-mirror）
./build/seedvr2-ncnn-vulkan -i input.mp4 -o output.mp4
```

权重仓库：[vvzc/seedvr2-ncnn-models](https://huggingface.co/vvzc/seedvr2-ncnn-models)

## 特性

- **单命令视频修复**：`-i in.mp4 -o out.mp4`，自动处理解码/编码/音频保留/帧数对齐
- **纯 C++ / ncnn Vulkan 推理**，无 Python 运行时，开箱即用的默认文本条件
- **6 个自定义 Vulkan 算子**（Conv3D / GroupNorm / SpatialAttention / SpaceTimeShuffle /
  DiT Input/Block/Output）与官方 PyTorch 数值对齐（30 用例全 PASS，max err ≤ 2e-5）
- **双模式加载**：流式（显存 ~3GB，慢）与常驻 `--resident`（显存 ~15GB，快约 66 倍）
- 支持动态分辨率/帧数，不限定导出尺寸

官方 PyTorch 参考实现建议 A100-80G（100 帧 720p）；本移植版把权重转成 fp16 并
提供流式加载，消费级显卡也能跑。

## 依赖

| 平台 | 依赖 |
|---|---|
| Linux / WSL2 | `build-essential` `cmake ≥3.16` `git` `libvulkan-dev` `vulkan-tools` `ffmpeg` |
| Windows | Visual Studio 2022、[Vulkan SDK](https://vulkan.lunarg.com/)、git、[ffmpeg](https://ffmpeg.org)（加入 PATH） |

显卡：任意支持 Vulkan 的 GPU（NVIDIA / AMD / Intel）。运行前建议用
`vulkaninfo --summary` 确认识别到独立显卡（而不是 llvmpipe 软渲染）。

## 构建

Linux / WSL2：

```bash
./build.sh        # 初始化 ncnn submodule → 自动应用 ncnn patches → cmake → 编译
```

Windows：

```bat
build.bat
```

构建会自动完成三件事：拉取 ncnn submodule（官方 Tencent/ncnn，固定在已验证提交）、
应用 `patches/` 里的两个补丁（Convolution3D Vulkan 后端 + 多核 CPU 位图解析修复）、
编译出 `build/seedvr2-ncnn-vulkan`。

## 下载模型

```bash
./download-models.sh
```

- 默认从 `hf-mirror.com`（国内加速）下载；海外可
  `SEEDVR2_HF_ENDPOINT=https://huggingface.co ./download-models.sh`
- 自定义仓库：`SEEDVR2_HF_REPO=<user>/<repo> ./download-models.sh`
- 已下载过的文件自动跳过，可中断重跑

`models/` 组装完成后：

```text
models/
├── dit_full_fp16/          # 3B DiT，fp16，34 对 param/bin（约 6.5GB）
├── vae_dynamic/            # VAE encoder + decoder（约 1GB）
├── default_pos_emb.bin     # 官方默认正文本条件（内置，随仓库分发）
└── default_neg_emb.bin     # 官方默认负文本条件（内置，随仓库分发）
```

## 使用

```bash
./build/seedvr2-ncnn-vulkan -i input.mp4 -o output.mp4 [选项]
```

| 选项 | 说明 | 默认 |
|---|---|---|
| `-i PATH` | 输入视频（ffmpeg 支持的任意格式） | 必填 |
| `-o PATH` | 输出视频（mp4，保留原音频） | 必填 |
| `--models DIR` | 模型目录 | `./models` |
| `--steps N` | 采样步数 | `1`（官方 one-step） |
| `--cfg X` | CFG 强度（>1 启用负向文本） | `1.0` |
| `--seed S` | 随机种子 | `666` |
| `--threads N` | CPU 线程数 | `8` |
| `--resident` | 32 个 DiT block 权重常驻显存 | 关闭 |

**显存与速度参考**（RTX PRO 6000，64×64×5 帧，1 step）：

| 模式 | 显存峰值 | DiT 单步耗时 |
|---|---|---|
| 流式（默认） | ~2.3GB | ~22s |
| `--resident` | ~14.5GB | ~0.33s（66×） |

流式模式每次采样步都要重新加载 6.4GB 权重，适合显存紧张的场景；显存 ≥16GB 建议
始终开启 `--resident`。步数 >1 时差距按步数成倍放大。

## 在你自己的 C++ 项目里使用

核心库 `seedvr2_ncnn` 是纯静态库，对外只暴露一个头文件：

```cpp
#include "seedvr2/engine.h"

seedvr2::RuntimeOptions opts;
opts.device = seedvr2::DeviceType::Vulkan;
opts.sampling_steps = 1;
opts.dit_resident = true;                 // 显存充足时建议开启

seedvr2::SeedVR2Engine engine;
engine.load("models/", opts);

seedvr2::Video input = /* THWC、RGB、fp32、[0,1] */;
seedvr2::Video output;
engine.process(input, {}, {}, output);    // 文本传空 → 自动用默认 embedding
```

完整字段见 [`include/seedvr2/engine.h`](include/seedvr2/engine.h)。错误信息通过
`engine.last_error()` 获取；设置环境变量 `SEEDVR2_PROFILE=1` 可输出分阶段耗时。

## 源码结构

```text
src/
├── core/       # engine 调度 + 运行时上下文 + Vulkan 配置
├── model/      # VAE / DiT / sampler
├── layers/     # 6 个自定义算子层（含 Vulkan compute shader）
├── pipeline/   # 视频前后处理
└── cli/        # 命令行入口
include/        # 对外公开头文件（seedvr2/engine.h）
tests/          # 分层数值对齐 runner（30 用例）+ 端到端 profiler
tools/          # PyTorch → ncnn 权重导出脚本、默认 embedding 转换
patches/        # 对官方 ncnn 的两个补丁（构建时自动应用）
docs/           # 算子实现与转换过程报告
```

## 从 PyTorch 权重重新导出（可选）

正常使用只需 `download-models.sh`。若想从官方 fp32 检查点自己生成 ncnn 权重：

```bash
python tools/export_seedvr2_vae_dynamic_ncnn.py ...   # VAE
python tools/export_seedvr2_dit_ncnn.py --storage fp16 ...  # DiT
python tools/export_default_embeddings.py             # 默认文本 embedding
```

## 性能说明

DiT 的 32 个 block 逐个串行执行。流式模式下每个采样步都会重新从磁盘加载
6.4GB 权重（为低显存设备设计）；`--resident` 一次性加载后常驻显存。激活值
逐层计算逐层释放，显存占用主要由「权重策略 + 分辨率×帧数」决定。

## 致谢

- [ByteDance-Seed/SeedVR2](https://github.com/ByteDance-Seed/SeedVR) —— 原模型与官方权重（Apache-2.0）
- [Tencent/ncnn](https://github.com/Tencent/ncnn) —— 推理框架
- [nihui 的 ncnn-vulkan 系列项目](https://github.com/nihui)（realesrgan-ncnn-vulkan 等）—— 项目形态参考

## License

Apache-2.0（与上游一致）
