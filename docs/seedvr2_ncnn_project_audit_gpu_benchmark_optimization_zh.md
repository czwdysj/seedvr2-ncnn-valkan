# SeedVR2 NCNN 项目评估与 GPU 测试优化方案

## 文档目的

本文评估 `/home/czw1/ncnn_learn/seedvr2_ncnn` 在 2026 年 9 月 11 日的实际状态，回答四个问题：项目已经完成到什么程度、目前还缺少什么、下一步如何进行可信的 GPU 测试、以及 ncnn Vulkan 相比 PyTorch 是否有性能提升空间。

本文面向后续开发和测试执行者。除特别标注为“建议”或“待测”的内容外，项目状态均来自当前代码、构建缓存、测试输出或本机环境探测。本文不把尚未在真实 GPU 上复现的数据写成已验证结论。

## 结论摘要

当前项目已经具备完整的工程骨架：动态 VAE、完整 32 层 DiT、六类 SeedVR2 自定义 Vulkan 层、Convolution3D Vulkan 补丁、C++ Engine、命令行程序、模型下载脚本、算子测试以及流式和常驻两种 DiT 权重策略均已存在。它不是空壳，也不再只是模型转换实验。

更准确的成熟度定位是：**端到端链路和自定义 Vulkan 算子基本完成的正确性优先原型**。目前还不能把它定义为“已经在 NVIDIA GPU 上完成验证并证明性能优势的发布版”，主要原因如下：

1. 当前 WSL2 的 Vulkan 设备是 `llvmpipe` 软件实现，不是 RTX 5060。现有 30 个 Vulkan 用例虽然全部通过，但不能证明 NVIDIA GPU 上的正确性和性能。
2. Engine 主链仍通过 `ncnn::Mat` 连接 VAE、DiT input、32 个 block、DiT output 和 sampler。大张量可能在 Net 边界反复上传、下载和同步，距离真正的端到端 `VkMat` 流水线还有明显差距。
3. 当前 ncnn 与 PyTorch benchmark 的 token 数、步数、精度、attention 后端、warmup 和统计方式不一致，不能直接计算可信的框架加速比。
4. 现有 Vulkan 实现以 FP32、pack1 和易验证的直接算法为主。DiT attention、VAE attention 和 Conv3D 都存在明确的 kernel 优化空间。
5. README 中的“66 倍”是 ncnn 常驻模式相对 ncnn 流式模式的加速，不是 ncnn 相对 PyTorch 的加速；仓库中缺少对应原始日志和可复现结果文件，应重新测量后再引用。

对于“能不能比 PyTorch 快”，当前最合理的判断是：

- **现在的实现大概率不能稳定超过使用 CUDA、cuBLAS、Apex 和 FlashAttention 的官方优化 PyTorch。** 当前实现的 CPU/GPU 往返、FP32 pack1、逐窗口调度和 correctness-first kernel 都会限制速度。
- **ncnn 已经具备部署侧优势。** 它不需要 Python 运行时，模型可用流式权重策略降低显存峰值，并且 Vulkan 具备跨 NVIDIA、AMD、Intel 平台的潜力。
- **完成全程 VkMat、权重缓存、FP16、window bucket、online softmax 和 Conv3D 优化后，在 batch 1、短视频、小分辨率或 PyTorch fallback 基线上，ncnn 有机会接近或超过 PyTorch。** 是否能超过官方 FlashAttention 基线必须由同机实测决定，不能预先承诺。

## 事实 推断和待测项的标记

本文使用以下三类标记避免混淆：

- **已观测**：当前机器命令输出、当前代码或当前测试直接证明。
- **工程判断**：根据代码结构和运行机制推导，需用 profiler 验证具体占比。
- **待测**：必须在原生 Linux 独立 GPU 上执行后才能形成结论。

## 1 当前项目审计

### 1.1 基线快照

| 项目 | 当前状态 | 证据或说明 |
| --- | --- | --- |
| 主仓库提交 | `7742903` | 当前 `main`，相对 `server/main` ahead 17 |
| ncnn 提交 | `81e8cbff` | 子模块包含项目补丁后的 dirty 状态 |
| 构建类型 | `Release` | `build/CMakeCache.txt` |
| Vulkan 构建 | 已开启 | `SEEDVR2_ENABLE_VULKAN=ON`、`NCNN_VULKAN=ON` |
| 模型资产 | 约 7.3GB | `models/dit_full_fp16`、`models/vae_dynamic`、默认 embedding |
| ncnn 测试 | 8 个 CTest、30 个算子 case 全通过 | 运行设备为 llvmpipe |
| 当前笔记本 GPU | RTX 5060 Laptop，约 8GB | CUDA 可见，compute capability 12.0，BF16 可用 |
| 当前 WSL 内存 | 约 7.6GB，可用约 6.5GB | 主机物理内存约 16GB |
| WSL Vulkan | 仅 llvmpipe | `vulkaninfo --summary` 未枚举 NVIDIA/Dozen |
| PyTorch | Miniforge base 中可用 | `torch 2.13.0+cu130`，CUDA smoke 通过 |
| PyTorch checkpoint | 本地缺失 | 未找到 `seedvr2_ema_3b.pth`、`ema_vae.pth` |
| 权威测试目标 | 原生 Linux，NVIDIA，显存至少 16GB | 具体 GPU 型号不预设 |

### 1.2 已完成能力

以下能力已经落入当前仓库：

- `SeedVR2Engine` 公开 API、运行时选项、错误码和 CLI 封装。
- 动态 VAE encoder/decoder，以及 GroupNorm、SpatialAttention、SpaceTimeShuffle 三个动态层。
- DiT input、32 个 DiT block、DiT output 和 Euler/CFG 调度。
- 六类自定义层的 CPU 与 Vulkan 入口。
- ncnn `Convolution3D` Vulkan 补丁及 5 个参数组合测试。
- DiT 单 block 流式加载和 32 block 常驻两种模式。
- 默认正负文本 embedding，可在不运行文本编码器的情况下推理。
- 两组真实尺寸 PyTorch reference 和 CPU 数值验证资产。
- 算子 benchmark、端到端 profile 和 PyTorch benchmark 的初始框架。
- Linux/WSL2 一键构建脚本，以及 Hugging Face 权重下载脚本。

主要代码入口：

- 公共 API：[`../include/seedvr2/engine.h`](../include/seedvr2/engine.h)
- Engine 调度：[`../src/core/engine.cpp`](../src/core/engine.cpp)
- Vulkan 上下文：[`../src/core/vulkan_context.cpp`](../src/core/vulkan_context.cpp)
- VAE：[`../src/model/vae.cpp`](../src/model/vae.cpp)
- DiT：[`../src/model/dit.cpp`](../src/model/dit.cpp)
- sampler：[`../src/model/sampler.cpp`](../src/model/sampler.cpp)
- 算子性能入口：[`../tests/perf_bench.cpp`](../tests/perf_bench.cpp)
- ncnn 端到端 profile：[`../tests/e2e_profile.cpp`](../tests/e2e_profile.cpp)
- PyTorch benchmark：[`../tools/pytorch_bench.py`](../tools/pytorch_bench.py)

### 1.3 当前成熟度评级

评分使用 5 分制。它是工程评审结果，不等同于模型质量评分。

| 维度 | 评分 | 评价 |
| --- | ---: | --- |
| 功能完整性 | 4.0 | 主链、权重、CLI 和主要自定义层齐全，但长视频和生产级 I/O 尚未完成 |
| 数值验证 | 3.0 | CPU/PyTorch 历史参考较扎实，Vulkan 有 30 个 case；缺真实 GPU、FP16 和完整模型自动回归 |
| GPU 就绪度 | 2.0 | Vulkan 代码与构建已完成，但当前本机只验证到软件 Vulkan，主链仍有 Mat 边界 |
| 性能工程 | 1.5 | 已有 benchmark 入口，但口径不公平、无 GPU timestamp/显存/JSON，热点 kernel 仍为正确性方案 |
| 发布质量 | 2.0 | 有 README 和一键脚本，但缺根 LICENSE、项目 CI、模型校验和长视频可靠性 |

综合定位：**Alpha 阶段的正确性原型，已具备进入真实 GPU 验证和系统优化阶段的条件。**

### 1.4 当前测试能证明什么

当前 `ctest -V` 共运行 8 个测试目标，内部包含 30 个 case：

| 模块 | case 数 | 当前结果 |
| --- | ---: | --- |
| DynamicSpaceTimeShuffle | 4 | 全通过 |
| DynamicFramewiseGroupNorm | 5 | 全通过 |
| DynamicFramewiseSpatialAttention | 4 | 全通过 |
| SeedVR2DiTInput | 4 | 全通过 |
| SeedVR2DiTOutput | 4 | 全通过 |
| SeedVR2DiTBlock | 4 | 全通过 |
| Convolution3D | 5 | 全通过 |
| 合计 | 30 | 全通过 |

这些测试证明：

- shader 能被当前 Vulkan loader 编译和执行；
- toy shape 下 Vulkan 输出与当前 ncnn CPU 实现一致；
- 若干非方形、非 4 对齐、stride、dilation 和 shifted-window 边界已覆盖。

这些测试尚不能证明：

- shader 在 NVIDIA 驱动上的正确性、稳定性和速度；
- 真实 2560 hidden size、20 heads 和完整 32 block 的表现；
- Vulkan 结果与 PyTorch reference 一致；
- FP16 storage、FP16 arithmetic 或 packing 正确；
- 端到端主链没有 CPU fallback 或大张量往返；
- README 中的显存和延迟数字可在当前提交上复现。

### 1.5 WSL2 GPU 状态

当前 WSL2 同时存在两个不同结论：

- CUDA 正常：`nvidia-smi` 和 PyTorch 均可访问 RTX 5060。
- Vulkan 不正常：`vulkaninfo --summary` 只枚举 `llvmpipe (LLVM 20.1.2)`，`deviceType` 是 CPU。

Ubuntu 24.04 当前安装的 `mesa-vulkan-drivers 25.2.8` 包含 `libvulkan_lvp.so`，但没有 `libvulkan_dzn.so` 和 `dzn_icd.json`。因此 CUDA 可用不代表 Vulkan 可用。当前 WSL2 可以继续用于代码开发、CPU 回归、CUDA/PyTorch smoke 和软件 Vulkan接口验证，但不能用于正式 ncnn Vulkan 性能结论。

本文不建议为了本轮测试在该环境中临时引入第三方 Mesa PPA。Dozen 是 Vulkan 到 D3D12 的转换路径，会额外引入兼容性和性能变量；本轮权威数据应来自原生 Linux Vulkan 驱动。

参考资料：

- [ncnn Vulkan notes](https://github.com/Tencent/ncnn/wiki/vulkan-notes)
- [ncnn layer support behavior](https://github.com/Tencent/ncnn/wiki/layer-support-behavior)
- [WSLg GPU acceleration](https://github.com/microsoft/wslg/blob/main/README.md)
- [NVIDIA CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/)

### 1.6 关键实现缺口

#### 主链不是完整的 VkMat 流水线

`src/core/engine.cpp`、`src/model/vae.cpp`、`src/model/dit.cpp` 和 `src/model/sampler.cpp` 的模块接口仍以 `ncnn::Mat` 为主。DiT input 输出到 CPU `Mat`，32 个 block 分别创建 Extractor 并输出 `Mat`，DiT output 再接收 `Mat`。流式模式还会为每个 block 重新创建 Net、加载参数和模型。

**工程判断：** 即使每个自定义层内部实现了 `forward_vkcompute`，这些模块边界仍可能触发上传、下载、command submit 和同步。是否是当前最大瓶颈需要用 profiler 确认，但在做复杂 kernel 优化前，应先完成全链路存储和同步审计。

#### CPU 上仍有主数据流计算

以下计算当前明确在 CPU `Mat` 上执行：

- VAE posterior 采样和 scaling；
- initial noise 与 augment noise 生成；
- condition latent 构造；
- 33 通道 DiT 输入拼接；
- CFG、CFG rescale 和 Euler endpoint；
- 部分预处理与全部最终 THWC 转换。

这些计算本身未必都很重，但会强制数据离开 GPU 或增加中间内存占用。

#### 精度优化尚未验收

`RuntimeOptions` 已经包含 `use_fp16_storage` 和 `use_fp16_arithmetic`，但：

- CLI 没有暴露对应参数；
- `e2e_profile` 没有打开它们；
- 所有现有 Vulkan runner 都显式关闭 FP16 和 packing；
- 自定义 shader 在不同设备上的半精度路径没有回归结果。

因此“模型 bin 为 fp16”不能等同于“GPU 计算已经采用完整、高效且验证过的 FP16 路径”。

#### Attention 和 Conv3D 仍是正确性优先实现

- DiT block attention 和 VAE SpatialAttention 使用多遍 QK 点积重算换显存，计算量偏高。
- window 处理存在较多独立 buffer 和 dispatch，真实分辨率下调度成本可能明显。
- Convolution3D 使用直接卷积，通道较大时无法充分利用高性能 GEMM 路径。
- 多个 elementwise、norm、Ada、RoPE、SwiGLU 和 residual 阶段尚未按 profiler 结果融合。

### 1.7 当前 benchmark 的公平性问题

#### ncnn 端

- `seedvr2_e2e_profile` 默认只使用 8 个随机文本 token，而默认正文本 embedding 是 58 个 token。
- `seedvr2_perf_bench` 使用 `dim=1024`、8 heads 等缩小参数，不是完整 3B DiT 的 `dim=2560`、20 heads。
- 每项只有 3 或 5 次计时，只报告 median。
- 没有 P90、原始样本、峰值显存、GPU timestamp、驱动和硬件元数据。
- 单次 warmup 不足以稳定区分 shader 编译、pipeline cache、模型加载和热运行。
- 数值失败不会形成结构化结果，也没有标准 JSON 供回归比较。

#### PyTorch 端

- `install_runtime_fallbacks()` 在导入项目模块前向 `sys.modules` 写入 Apex 和 FlashAttention fallback，因此即使服务器安装了真实扩展，当前脚本也会优先使用 fallback。
- fallback 的 varlen attention 逐 segment 调用 PyTorch SDPA，不能代表官方 FlashAttention 性能。
- `--steps` 虽然可配置，但采样循环第一轮后无条件 `break`；标签仍按总 timestep 数打印。
- PyTorch 读取 58 token 的官方 `pos_emb.pt`，与 ncnn profile 默认 8 token 不一致。
- PyTorch 开启 BF16 和 TF32，而 ncnn benchmark 强制 FP32 pack1。
- 两端随机数生成器不同，仅使用相同 seed 无法得到相同 latent/noise。
- 两端没有统一 cold load、warm inference 和端到端统计边界。

在修复以上问题前，任何“ncnn 比 PyTorch 快 X 倍”的结论都应视为无效。

### 1.8 发布和产品化缺口

#### 根许可证与 CI

README 声明 Apache-2.0，但主仓库根目录没有实际 `LICENSE` 文件。仓库也没有项目级 GitHub Actions；子模块 ncnn 自带的 CI 不能替代本项目 CI。

#### 模型下载完整性

`download-models.sh` 只判断目标文件是否非空，没有 SHA256、预期大小、临时文件或原子重命名。下载中断后留下的部分文件可能在下次执行时被误判为完整模型。

建议采用：

```text
下载到 filename.part
 -> 校验 Content-Length 或 manifest 中的 size
 -> 校验 SHA256
 -> 原子 rename 为正式文件
```

#### CLI 补帧问题

CLI 在送入 Engine 前已经把帧数补到 `4n+1`，随后覆盖了本地 `frame_count`。Engine 再把这个补齐后的帧数当作原始帧数，最终输出可能保留额外复制帧，与“编码前裁回”的注释不一致。

#### 视频内存和安全性

- CLI 一次性把完整视频解码为 `rgb24` 并复制为 FP32 `Video`，内存复杂度随视频总帧数线性增长。
- Engine 还会产生预处理 tensor、latent、condition、noise 和输出等副本。
- `ffmpeg` 与 `ffprobe` 命令通过 shell 字符串拼接，包含双引号或 shell 特殊字符的路径存在失败和命令注入风险。

#### 文档不一致

- clone URL 的仓库名为 `seedvr2-ncnn-valkan`，下一行却进入 `seedvr2-ncnn-vulkan`。
- README 表述“支持动态分辨率”，但预处理实际中心裁剪到 16 的倍数，并可能丢弃边缘像素。
- “纯 Vulkan 推理”容易让读者误认为全程为 `VkMat` 且没有 CPU 主数据流处理。
- `docs/seedvr2_vulkan_development_plan_zh.md` 的若干“尚未实现”描述已经过时，需要后续单独更新。
- 发布支持范围需要明确限定为 Linux/WSL2，避免把源码层面的可移植性误解为已验证平台。

## 2 原生 Linux GPU 测试操作手册

### 2.1 测试原则

正式测试必须满足：

- 原生 Linux；
- NVIDIA 独立 GPU，显存至少 16GB；
- `vulkaninfo` 枚举真实 NVIDIA 设备；
- ncnn 输出的 GPU 名称不是 llvmpipe/lavapipe；
- ncnn 与 PyTorch 在同一台机器、同一驱动上运行；
- 性能、显存和精度数据都保留原始记录。

若服务器只有 16GB，先运行流式模式；常驻模式可能因 workspace 和驱动分配超过可用显存。不要因为 GPU 标称 16GB 就默认常驻一定成功。

### 2.2 目录变量

以下命令假设项目、ncnn 权重和 PyTorch 项目在同一仓库中。按服务器实际路径修改第一行即可：

```bash
export SEEDVR2_ROOT=/path/to/seedvr2_ncnn
export SEEDVR2_MODELS="$SEEDVR2_ROOT/models"
export SEEDVR2_PYTORCH="$SEEDVR2_ROOT/pytorch_model"
export SEEDVR2_RESULTS="$SEEDVR2_ROOT/results/gpu-$(date +%Y%m%d-%H%M%S)"

mkdir -p "$SEEDVR2_RESULTS"
cd "$SEEDVR2_ROOT"
```

### 2.3 环境门禁

先记录完整环境：

```bash
{
  date --iso-8601=seconds
  uname -a
  lscpu
  free -h
  nvidia-smi
  nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,power.limit \
    --format=csv,noheader
  vulkaninfo --summary
  cmake --version
  g++ --version
  glslangValidator --version || true
  git rev-parse HEAD
  git status --short
  git -C ncnn rev-parse HEAD
  git -C ncnn status --short
} 2>&1 | tee "$SEEDVR2_RESULTS/environment.txt"
```

执行硬门禁：

```bash
vulkaninfo --summary >"$SEEDVR2_RESULTS/vulkan-summary.txt" 2>&1

if grep -qiE 'llvmpipe|lavapipe|deviceType[[:space:]]*=[[:space:]]*PHYSICAL_DEVICE_TYPE_CPU' \
  "$SEEDVR2_RESULTS/vulkan-summary.txt"; then
  echo 'ERROR: software Vulkan detected; stop GPU benchmark.' >&2
  exit 1
fi

if ! grep -qiE 'NVIDIA|vendorID[[:space:]]*=[[:space:]]*0x10de' \
  "$SEEDVR2_RESULTS/vulkan-summary.txt"; then
  echo 'ERROR: NVIDIA Vulkan device not found; stop GPU benchmark.' >&2
  exit 1
fi
```

通过条件：

- `deviceName` 对应目标 NVIDIA GPU；
- `deviceType` 是 `PHYSICAL_DEVICE_TYPE_DISCRETE_GPU`；
- vendor ID 为 `0x10de`；
- 没有 llvmpipe/lavapipe；
- 后续 runner 启动日志中的 ncnn GPU 名称与 `vulkaninfo` 一致。

### 2.4 准备模型

ncnn 权重：

```bash
cd "$SEEDVR2_ROOT"
./download-models.sh
du -sh "$SEEDVR2_MODELS"
find "$SEEDVR2_MODELS" -type f -size 0 -print
```

`find` 不应输出任何文件。正式测试前还应完成 P0 中建议的 SHA256 manifest；在该功能实现前，至少保存文件大小列表：

```bash
find "$SEEDVR2_MODELS" -type f -printf '%s  %P\n' \
  | sort >"$SEEDVR2_RESULTS/model-files.txt"
```

PyTorch 权重应包含：

```text
pytorch_model/ckpts/seedvr2_ema_3b.pth
pytorch_model/ckpts/ema_vae.pth
```

按官方 SeedVR2 README 使用 Hugging Face snapshot 下载，下载后保存实际文件名、大小和 SHA256。若服务器实际模型文件名不同，以 `configs_3b/main.yaml` 和 `configure_dit_model()` 读取路径为准。

### 2.5 独立构建目录

Profile 构建用于 validation、调试和 profiler：

```bash
cmake -S "$SEEDVR2_ROOT" -B "$SEEDVR2_ROOT/build-gpu-profile" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DSEEDVR2_ENABLE_VULKAN=ON \
  -DSEEDVR2_BUILD_TESTS=ON \
  -DSEEDVR2_BUILD_APPS=ON

cmake --build "$SEEDVR2_ROOT/build-gpu-profile" \
  --parallel "$(nproc)"
```

Release 构建用于最终性能：

```bash
cmake -S "$SEEDVR2_ROOT" -B "$SEEDVR2_ROOT/build-gpu-release" \
  -DCMAKE_BUILD_TYPE=Release \
  -DSEEDVR2_ENABLE_VULKAN=ON \
  -DSEEDVR2_BUILD_TESTS=ON \
  -DSEEDVR2_BUILD_APPS=ON

cmake --build "$SEEDVR2_ROOT/build-gpu-release" \
  --parallel "$(nproc)"
```

如果服务器内存不足，不要盲目使用全部 CPU 核心编译；将 `--parallel` 改为 4 或 8。

### 2.6 第一层 算子正确性

先在 profile 构建中运行详细测试：

```bash
ctest --test-dir "$SEEDVR2_ROOT/build-gpu-profile" \
  -V 2>&1 | tee "$SEEDVR2_RESULTS/ctest-vulkan-fp32.txt"
```

检查日志顶部的 ncnn device 名称。30 个 case 全部 PASS 只是第一层出口；若 GPU 名称不正确，即使 CTest 返回 0 也不能继续。

建议初始精度阈值：

| 算子类别 | Vulkan FP32 对 ncnn CPU |
| --- | ---: |
| 纯索引和重排 | bit exact 或 `max_abs <= 1e-6` |
| elementwise 与线性层编排 | `max_abs <= 1e-4` |
| norm 和 reduction | `max_abs <= 5e-4` |
| 单次 attention 或 DiT block | `NRMSE <= 1e-3` 且 `cosine >= 0.99999` |
| 完整 32 block 新增误差 | `NRMSE <= 2e-3` |

第一次真实 GPU 测量后冻结阈值。后续不能为让测试通过而临时放宽。

### 2.7 第二层 PyTorch reference 对齐

当前 CPU reference 工具可用于确认历史基线，但现有 runner 不是完整 GPU reference 测试。运行前先激活已经安装 PyTorch、NumPy 等依赖的 SeedVR2 Conda 环境，并确认 `python -c 'import torch'` 成功；不要直接使用缺少 torch 的系统 Python。然后保留以下 CPU 回归：

```bash
python "$SEEDVR2_ROOT/tools/test_seedvr2_vae_dynamic_ncnn.py" \
  --runner "$SEEDVR2_ROOT/build-gpu-release/seedvr2_vae_dynamic_runner" \
  --model-dir "$SEEDVR2_MODELS/vae_dynamic" \
  --max-abs 5e-4

python "$SEEDVR2_ROOT/tools/test_seedvr2_dit_ncnn.py" \
  --reference-dir "$SEEDVR2_ROOT/test_vectors/example_001_320x240" \
  --reference-dir "$SEEDVR2_ROOT/test_vectors/example_002_480x270" \
  --model-dir "$SEEDVR2_MODELS/dit_full_fp16" \
  --block-runner "$SEEDVR2_ROOT/build-gpu-release/seedvr2_dit_block_runner" \
  --full-runner "$SEEDVR2_ROOT/build-gpu-release/seedvr2_dit_full_runner" \
  --output "$SEEDVR2_RESULTS/dit-cpu-reference.json"
```

正式 GPU 验收前必须补充等价 Vulkan runner，至少覆盖：

- VAE encoder 与 decoder；
- DiT input；
- block 0、9、10、31；
- 完整 32 blocks；
- DiT output；
- Engine 端到端。

选择 0、9、10、31 的原因：它们覆盖独立权重阶段、MM 与 shared-weight 边界、shifted window 和最后一层 `vid_only` 特例。

### 2.8 第三层 端到端 smoke

先使用 5 帧、64×64、58 token、1 step：

```bash
SEEDVR2_PROFILE=1 \
"$SEEDVR2_ROOT/build-gpu-release/seedvr2_e2e_profile" \
  "$SEEDVR2_MODELS" 5 64 64 58 \
  2>&1 | tee "$SEEDVR2_RESULTS/e2e-smoke-stream.txt"
```

显存允许时再测常驻：

```bash
SEEDVR2_PROFILE=1 \
SEEDVR2_DIT_RESIDENT=1 \
"$SEEDVR2_ROOT/build-gpu-release/seedvr2_e2e_profile" \
  "$SEEDVR2_MODELS" 5 64 64 58 \
  2>&1 | tee "$SEEDVR2_RESULTS/e2e-smoke-resident.txt"
```

当前 `e2e_profile` 是一次性运行，适合功能 smoke，不适合作为最终统计。完成 P0 benchmark 改造后再运行 3 次 warmup 和至少 10 次正式采样。

### 2.9 第四层 真实尺寸

固定 5 帧、58 token、CFG 1.0、1 step，按顺序增加尺寸：

| 场景 | T | H | W | 用途 |
| --- | ---: | ---: | ---: | --- |
| smoke | 5 | 64 | 64 | 快速验证完整链路 |
| 标准小视频 | 5 | 240 | 320 | 对齐 example 1 |
| 宽画幅 | 5 | 256 | 480 | 对齐 example 2 有效尺寸 |
| 时间扩展 | 9 | 240 | 320 | 验证时间维增长和内存曲线 |

每个 shape 依次测试：

1. 流式 FP32；
2. 常驻 FP32，显存允许时；
3. FP16 storage + FP32 accumulation；
4. FP16 arithmetic；
5. 后续的预算缓存模式。

若某配置 OOM，记录 OOM 时的空闲显存、已分配显存、shape、模式和阶段，不要只写“运行失败”。

### 2.10 冷启动 热运行和显存

最终 benchmark 必须区分：

- 进程启动；
- Vulkan instance 与 shader/pipeline 创建；
- 模型文件读取；
- 权重上传；
- 第一次推理；
- warm inference；
- 输出下载和后处理。

每种配置执行：

```text
warmup = 3
measured runs = 10 或更多
报告 = min、P50、P90、mean、stdev
```

显存至少记录：

- 空闲基线；
- model load 后；
- VAE encode 峰值；
- DiT 峰值；
- VAE decode 峰值；
- 进程退出后是否恢复。

`nvidia-smi` 适合外部粗粒度采样；精确的 Vulkan allocator、workspace 和 GPU 时间应由 ncnn allocator 统计、Vulkan timestamp query 或 Nsight 提供。

### 2.11 结果文件要求

每次运行保存一个 JSON，不只保留汇总表。建议 schema：

```json
{
  "schema_version": 1,
  "status": "pass",
  "git": {
    "seedvr2": "full commit",
    "ncnn": "full commit",
    "dirty": false
  },
  "hardware": {
    "gpu": "detected name",
    "driver": "detected version",
    "vram_mib": 0,
    "cpu": "detected name",
    "ram_mib": 0
  },
  "config": {
    "backend": "ncnn-vulkan",
    "precision": "fp16-storage-fp32-accum",
    "weight_mode": "stream",
    "frames": 5,
    "height": 240,
    "width": 320,
    "text_tokens": 58,
    "steps": 1,
    "cfg_scale": 1.0,
    "warmup": 3,
    "runs": 10
  },
  "samples_ms": [],
  "summary_ms": {
    "min": 0.0,
    "p50": 0.0,
    "p90": 0.0,
    "mean": 0.0
  },
  "stages_ms": {},
  "peak_vram_mib": 0,
  "accuracy": {
    "finite": true,
    "max_abs": 0.0,
    "mean_abs": 0.0,
    "rmse": 0.0,
    "nrmse": 0.0,
    "cosine": 1.0
  }
}
```

## 3 PyTorch 公平对比规范

### 3.1 必须保留两个 PyTorch 基线

#### 官方优化基线

要求实际加载并报告：

- 官方 SeedVR2 代码和 checkpoint；
- Apex fused norm；
- FlashAttention varlen kernel；
- BF16 autocast；
- TF32 状态；
- PyTorch、CUDA、cuDNN、FlashAttention 和 Apex 版本。

#### 兼容 fallback 基线

使用当前 Python RMSNorm 和逐窗口 SDPA fallback。该基线用于兼容性与数值定位，必须在结果中写为 `pytorch-fallback`，不能写成“官方 PyTorch 性能”。

### 3.2 当前 PyTorch benchmark 必须先修复

P0 应至少完成：

1. 增加 `--attention-backend official|fallback`，选择 official 时禁止注入 fallback，扩展缺失则明确失败。
2. 移除采样循环中的无条件 `break`，确保 `--steps N` 实际执行 N 步。
3. 增加 `--warmup`、`--runs`、`--output-json`、`--dtype`、`--text-embedding` 和 `--condition-noise-scale`。
4. 将模型加载、VAE encode、prepare、DiT、VAE decode 和总耗时使用同一计时规范。
5. 每个 CUDA 阶段计时前后正确 synchronize；同时保留 CUDA Event 时间和 wall time。
6. 输出 `max_memory_allocated` 与 `max_memory_reserved`，测试前重置 peak stats。
7. 使用实际 58 token 正文本 embedding，或显式把 token 数写入结果。

### 3.3 公平比较固定项

同一对比组必须固定：

- 同一台机器和同一块 GPU；
- 相同驱动；
- 相同输入 shape 和有效帧数；
- 正文本 token 数为 58；
- CFG 1.0；
- 1 sampling step；
- 相同 condition noise scale；
- 相同 VAE 随机/确定性策略；
- 相同计时边界；
- 相同 warmup 和 runs；
- GPU 处于相近功耗、温度和空闲状态。

不得把以下结果直接相除：

- ncnn FP32 与 PyTorch BF16；
- ncnn warm inference 与 PyTorch cold load；
- ncnn 8 token 与 PyTorch 58 token；
- ncnn stream 与 PyTorch resident，而不同时说明显存策略；
- PyTorch fallback 与“官方 FlashAttention”；
- WSL2 Dozen Vulkan 与原生 Linux CUDA。

### 3.4 精度组和性能组分开

性能测试只要求输入分布和 shape 相同，可以使用固定随机输入。

精度测试必须使用相同 tensor，不能只设置相同 seed。PyTorch CUDA 使用 Philox，C++ 当前使用 `std::mt19937_64`，两端随机序列不会一致。建议测试资产包含：

- 预处理后视频 tensor；
- VAE posterior mean/logvar，或直接使用 mean 路径；
- initial noise；
- augment noise；
- condition latent；
- 正负文本 embedding；
- timestep；
- block 0、9、10、31 输入输出；
- 最终 DiT latent；
- VAE decode 输出。

每个文件必须带 shape、dtype、布局、来源 commit 和 SHA256。测试时优先注入预生成 tensor，避免 RNG 差异掩盖计算差异。

### 3.5 精度与质量指标

中间 tensor：

- finite 检查；
- max absolute error；
- mean absolute error；
- RMSE；
- NRMSE；
- cosine similarity。

最终视频：

- PSNR；
- SSIM；
- 可选 LPIPS；
- 固定帧编号的 PNG 对照；
- 时间一致性检查，避免只看单帧。

FP16 对 PyTorch 的建议验收方式：完整 DiT 的 NRMSE 不比已冻结的 ncnn CPU/PyTorch 基线恶化超过 10%，或绝对增加不超过 0.002，取更严格者；cosine 降低不超过 `1e-5`。阈值应在第一轮真实 GPU 数据后冻结。

### 3.6 性能指标

必须同时报告：

- cold model load；
- warm model load 或 pipeline cache load；
- VAE encode；
- DiT input；
- 32 blocks；
- DiT output；
- sampler；
- VAE decode；
- 端到端；
- FPS；
- 每百万输入像素毫秒数；
- 峰值显存；
- 权重策略和实际缓存 block 数。

建议结果表：

| 引擎 | 后端 | 精度 | 权重模式 | Shape | P50 | P90 | 峰值显存 | NRMSE | 备注 |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| PyTorch | CUDA official | BF16/TF32 | resident/offload | 待测 | 待测 | 待测 | 待测 | reference | FlashAttention |
| PyTorch | CUDA fallback | BF16/TF32 | resident/offload | 待测 | 待测 | 待测 | 待测 | reference | SDPA fallback |
| ncnn | Vulkan | FP32 | stream | 待测 | 待测 | 待测 | 待测 | 待测 | correctness baseline |
| ncnn | Vulkan | FP32 | resident | 待测 | 待测 | 待测 | 待测 | 待测 | VRAM permitting |
| ncnn | Vulkan | FP16 storage | cache | 待测 | 待测 | 待测 | 待测 | 待测 | optimized candidate |

### 3.7 如何解释最终结果

如果 ncnn stream 显存更低但延迟更高，应表述为“用 I/O 和权重上传换显存”，不能表述为全面性能提升。

如果 ncnn 超过 fallback PyTorch，但没有超过官方 FlashAttention，应表述为“超过兼容基线，尚未超过官方优化基线”。

如果小 shape 超过 PyTorch、真实 shape 落后，应说明框架启动/调度开销与大规模 kernel 效率的分界，不能只选最好看的数据。

如果常驻比流式快 66 倍，只能说明权重加载策略的收益；必须另列 PyTorch 数据才能回答框架对比。

## 4 分阶段优化路线

### P0 基准可信化

**目标：** 在做性能优化前建立不会误导的测量系统。

实施内容：

- 修正 PyTorch step、token、condition 和 fallback 选择。
- ncnn/PyTorch 输出统一 JSON schema。
- 增加硬件、驱动、commit、dirty 状态、精度、shape 和 token 元数据。
- 增加 warmup、runs、原始样本、P50/P90 和峰值显存。
- 增加真实 GPU 门禁，发现 llvmpipe/lavapipe 时性能程序返回非零。
- 为 CPU、Vulkan、PyTorch 共用 reference manifest 和指标计算。
- 把 GPU runner 和历史 PyTorch reference 接入 CTest 或单独的 GPU test suite。

预期收益：不会直接提高推理速度，但能防止后续优化建立在错误数据上。

主要风险：不同后端计时边界仍可能不一致。解决方法是同时报告 wall time、GPU event/timestamp 和明确的阶段边界。

验收条件：

- 同一配置连续两轮 P50 差异不超过 5%，否则先解释温度、功耗或后台负载；
- JSON 可由脚本重新生成汇总表；
- 多步配置实际执行指定步数；
- official 和 fallback PyTorch 结果名称不可混淆；
- 软件 Vulkan 无法生成标记为 GPU 的成绩。

### P1 全程 VkMat

**目标：** 输入上传一次，大张量在 VAE、DiT、sampler 和 decoder 之间保持在 GPU。

实施内容：

- RuntimeContext 统一管理 `VkAllocator`、`VkCompute` 和 staging allocator。
- 为 VAE、DiT 和 sampler 增加内部 `VkMat` 路径，公共 `SeedVR2Engine` API 保持不变。
- DiT input、32 blocks 和 output 通过 `VkMat` 直接串联。
- 把 posterior、noise、condition、33 通道拼接、CFG 和 Euler 移到 GPU。
- 只允许小型 shape metadata 留在 CPU。
- 最终 postprocess 前才下载视频 tensor。
- 增加传输审计日志和 profiler 标记。

预期收益：减少 PCIe/DXG 传输、submit_and_wait 和中间 CPU 内存；这是当前最应该先验证的系统级优化。

主要风险：allocator 生命周期、command 未完成时释放权重、多个 Net 共享 Vulkan device 和 OOM 清理。

验收条件：

- profiler 中 32 block 之间没有大 tensor download/upload；
- 主链大 tensor 上传和下载次数接近输入一次、输出一次；
- FP32 结果满足冻结精度阈值；
- 重复处理 20 次无显存增长；
- 若 P50 没有改善，保留 profiler 证据并重新定位瓶颈，而不是宣称成功加速。

### P2 权重与 pipeline 生命周期

**目标：** 在单 block 流式与全常驻之间增加可控的显存性能折中。

实施内容：

- 增加 `vram_budget_mb` 或 `dit_cache_blocks`。
- 实现 N-block LRU 或顺序滑动缓存。
- 复用 Net、pipeline、allocator 和 workspace，避免每次 forward 重建。
- 使用 ncnn/Vulkan pipeline cache 降低冷启动 shader/pipeline 成本。
- 流式模式预取下一 block，实现磁盘读取、host decode、权重上传和当前 block 计算的受控重叠。
- 将 cold load、cache miss、cache hit 和 warm inference 分开计时。

预期收益：16GB 级 GPU 可避免全常驻 OOM，同时显著减少每步重复读取和上传 6.4GB 权重的成本。

主要风险：双缓冲提高瞬时显存；错误的 fence 或引用生命周期会导致随机崩溃或错误结果。

验收条件：

- 实际显存不超过配置预算加已记录的固定 workspace 裕量；
- cache hit/miss 可观测；
- 结果与无缓存模式一致；
- 只有在相关阶段 P50 至少改善 5%，或显存至少降低 15% 且延迟代价可接受时保留复杂策略。

### P3 FP16 与 packing

**目标：** 减少权重和激活带宽，并提高 GPU 算术吞吐。

推荐顺序：

1. FP16 storage + FP32 accumulation；
2. FP16 packed storage；
3. pack4；
4. FP16 arithmetic；
5. 仅对验证稳定的子阶段开启更激进精度。

实施内容：

- 让 CLI 和 benchmark 显式控制并打印实际精度。
- 为每个自定义层增加 FP16 与 packing case。
- norm、softmax、方差和累加默认保持 FP32。
- 每次只改变一个精度变量，避免误差来源不可定位。
- 分别记录权重显存、activation workspace 和总峰值。

预期收益：降低权重和激活带宽；有机会让更多 block 驻留在 16GB GPU。

主要风险：半精度 reduction、exp、RoPE 和长序列 attention 的误差放大，以及 shader buffer 类型与 elempack 不匹配。

验收条件：

- 所有 shape 和输出有限；
- 单层和完整模型均满足冻结阈值；
- 峰值显存相对 FP32 有清晰下降；
- 性能没有改善且精度变差的配置不设为默认值。

### P4 热点 kernel

#### DiT attention

- 将相同 window shape 分桶，减少逐窗口独立 dispatch。
- 使用 tiled online softmax，单遍或分块维护 row max、sum 和输出，避免三遍 QK 重算。
- Q/K norm、RoPE、gather 尽量融合或共享中间 buffer。
- 评估 joint video/text varlen attention 的批量布局。

#### VAE SpatialAttention

- 先确认它在真实 VAE profile 中的占比。
- 若是热点，采用逐帧 tiled online softmax，不物化完整 `N×N` scores。
- 保留小尺寸 materialized 或 CPU 路径作为 oracle。

#### Convolution3D

- 统计 VAE 中全部 Conv3D 的唯一参数族和耗时占比。
- 对 1×1×1 使用专门 pointwise/GEMM 路径。
- 对 3×3×3 热点评估 im2col + GEMM、时间维展开 Conv2D 或实际参数族 specialized shader。
- 不先实现覆盖所有 group/dilation 的复杂通用 kernel，除非参数统计证明有必要。

#### 融合策略

按 profiler 结果依次评估：

- RMSNorm + Ada；
- Q/K norm + RoPE；
- projection + bias + gate；
- SwiGLU；
- residual add。

预期收益：减少重复计算、全局内存流量和 dispatch 数。

主要风险：过早融合会增加寄存器压力、降低 occupancy，并显著增加验证难度。

验收条件：

- 优化 kernel 与 reference 的数值阈值不退化；
- Nsight 或 timestamp 证明被优化阶段确实是热点；
- 单 kernel 变快但端到端无收益时不夸大结果；
- 复杂融合只在相关阶段 P50 至少改善 5% 后保留。

### P5 长视频与发布质量

**目标：** 从可演示原型推进到可重复使用的工具。

实施内容：

- 视频按时间和空间分块处理，增加 overlap 和融合策略。
- VAE 支持 memory state/cache，避免整段 clip 和所有 FP32 帧驻留内存。
- 修复 CLI 双重补帧和最终帧数裁剪。
- 对空间中心裁剪给出明确日志，后续提供 pad/crop 策略选项。
- 使用安全的进程参数 API 或 ffmpeg 库，停止拼接 shell 命令。
- 模型下载增加 size/SHA256 manifest、`.part`、断点续传和原子重命名。
- 添加根 `LICENSE`、项目 CI、CPU nightly、真实 GPU nightly 和 release checklist。
- README 的功能、限制和性能数字自动引用最新 benchmark 结果。

验收条件：

- 处理长视频时 CPU 内存和显存由 chunk 大小控制，不再随总帧数线性增长；
- 输入文件名包含空格、引号和非 ASCII 字符时行为正确；
- 输出帧数、FPS 和音频时长与原视频一致；
- 任意损坏模型在 load 前被校验拒绝；
- 全新机器能按 README 从 clone 到 smoke test 完成复现。

## 5 推荐接口与测试资产

本报告不修改公共 API。后续建议在保持 `SeedVR2Engine` 简洁的前提下增加以下入口。

### 5.1 CLI

建议生产 CLI 增加：

```text
--gpu INDEX
--fp16-storage
--fp16-arithmetic
--vram-budget-mb N
--profile-json PATH
```

建议 benchmark 程序增加：

```text
--warmup N
--runs N
--frames N
--height N
--width N
--text-tokens N
--precision fp32|fp16-storage|fp16-arithmetic
--weight-mode stream|resident|budget
--output-json PATH
```

`RuntimeOptions` 已有 `vulkan_device_index`、`use_fp16_storage` 和 `use_fp16_arithmetic`，CLI 只需正确映射并打印。显存预算需要新增字段；现有 `--resident` 保留为全常驻快捷方式。

### 5.2 Reference manifest

建议每套 reference 使用一个 manifest：

```json
{
  "schema_version": 1,
  "source_commit": "PyTorch commit",
  "model": "SeedVR2-3B",
  "config": {
    "frames": 5,
    "height": 240,
    "width": 320,
    "text_tokens": 58,
    "steps": 1,
    "cfg_scale": 1.0
  },
  "files": [
    {
      "path": "initial_noise.pt",
      "shape": [16, 2, 30, 40],
      "dtype": "float32",
      "layout": "C,T,H,W",
      "sha256": "..."
    }
  ]
}
```

### 5.3 自动测试分层

```text
Level 0  环境和真实 GPU 门禁
Level 1  primitive 与 30 个 toy case
Level 2  自定义层和实际参数族
Level 3  VAE 与 DiT 子图
Level 4  32 block 与完整 Engine
Level 5  真实视频质量 性能 显存 稳定性
```

上层失败时先回到下一层定位，不能只比较最终视频。

## 6 建议的下一步执行顺序

### 第一阶段 建立可信基线

1. 在原生 Linux 16GB 以上 NVIDIA 服务器完成环境门禁。
2. 保存当前提交的 30 个真实 GPU FP32 case 结果。
3. 完成 P0 benchmark 修复和 JSON schema。
4. 准备官方优化与 fallback 两套 PyTorch 环境。
5. 固定 64×64×5、320×240×5、480×256×5 reference 和 manifest。

阶段出口：能够回答“同一配置重复运行结果是否稳定”，但暂不要求 ncnn 更快。

### 第二阶段 消除系统级瓶颈

1. 完成 P1 VkMat 主链。
2. 用 profiler 对比优化前后的上传、下载、同步和阶段延迟。
3. 完成 P2 显存预算缓存和 pipeline 复用。
4. 再次运行完整精度、性能和内存回归。

阶段出口：大 tensor 不在 block 间回到 CPU，16GB 级 GPU 有可用的 stream/cache 策略。

### 第三阶段 精度和 kernel 优化

1. FP16 storage + FP32 accumulation。
2. packing 与 FP16 arithmetic。
3. 根据 profile 优化 DiT attention。
4. 优化 Conv3D 参数族。
5. 必要时优化 VAE SpatialAttention 和 elementwise 融合。

阶段出口：形成 ncnn FP32、ncnn mixed precision、PyTorch official、PyTorch fallback 四套可复现数据。

### 第四阶段 发布化

完成长视频切片、I/O 安全、下载校验、CI、LICENSE、README 校正和正式性能报告。

## 7 最终验收清单

### GPU 正确性

- [ ] Vulkan device 是目标 NVIDIA 独显，不是 llvmpipe/lavapipe。
- [ ] 30 个 FP32 case 在真实 GPU 全部通过。
- [ ] FP16 storage 和 FP16 arithmetic 各自有单层与完整模型回归。
- [ ] VAE、block 0/9/10/31、32 block 和 Engine 均与 reference 对齐。
- [ ] validation layer 无错误。
- [ ] 连续运行 20 次无崩溃、NaN 或显存增长。

### 性能

- [ ] 同机、同 GPU、同 shape、同 token、同 step、同 CFG。
- [ ] PyTorch official 与 fallback 分开报告。
- [ ] FP32 与混合精度分开报告。
- [ ] warmup 至少 3 次，正式样本至少 10 次。
- [ ] 保留原始样本、P50、P90、显存和阶段数据。
- [ ] cold load 和 warm inference 分开。
- [ ] stream、resident 和 budget cache 明确区分。
- [ ] “66 倍”等结论能从归档 JSON 重新计算。

### 发布质量

- [ ] 根 LICENSE 与 README 声明一致。
- [ ] 模型下载有 SHA256 和原子完成机制。
- [ ] CLI 不保留补齐帧，输出时长正确。
- [ ] 长视频内存由 chunk 控制。
- [ ] 路径处理不通过不安全 shell 拼接。
- [ ] 项目 CI 与 GPU nightly 存在。
- [ ] README 的命令、目录名、尺寸语义和性能数据可复现。

## 8 最终评价

这个项目当前最有价值的部分不是某个未经复核的加速数字，而是已经完成了一个难度较高的 SeedVR2 3B 到 ncnn 的模型拆分、动态 shape、自定义算子和 Vulkan 移植骨架。六类自定义层、完整 DiT、动态 VAE、流式权重策略和分层 reference 使它具备继续优化的基础。

下一步不应立刻追求更多融合 shader，也不应先宣传超过 PyTorch。最优先的工作是：在真实 NVIDIA Vulkan 环境建立公平 benchmark，然后用 profiler 证明 `Mat/VkMat` 往返、权重加载、attention 或 Conv3D 各自占多少。完成 P0 和 P1 后，性能数据才足以决定 FP16、缓存和 kernel 优化的投入顺序。

如果最终结果显示 ncnn 没有超过官方 PyTorch，也不代表项目失败。只要能够量化说明部署依赖、显存、冷启动、跨平台性和延迟之间的取舍，并给出 profiler 驱动的优化过程，这个项目仍然是完整且有说服力的模型部署工程案例。
