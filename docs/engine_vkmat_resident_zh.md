# SeedVR2Engine 大张量 VkMat 常驻实现报告

## 1. 目标与边界

本阶段把一次 `SeedVR2Engine::process()` 内的主要数据流改为 Vulkan resident tensors：

```text
CPU 预处理视频 + 可复现噪声 + 文本
    -> 一批集中上传
VAE Encoder -> posterior sample -> condition/latent
    -> DiT positive/negative -> CFG -> Euler（循环 N 步）
    -> VAE Decoder
    -> 一次最终视频下载
CPU 后处理视频
```

这里的“零中间往返”指大型 feature tensor 不做 device-to-host/host-to-device 往返。
shape、timestep 等极小控制数据仍允许上传；输入视频、CPU RNG 生成的噪声和文本属于
推理入口数据，最终视频属于出口数据。

## 2. 核心设计

### 2.1 共享执行上下文

`VulkanExecutionContext` 在一次 process 中持有同一 Vulkan device、blob allocator 和
staging allocator。VAE、DiT 和 sampler 返回的 `VkMat` 因此可以跨组件存活，不会因
单个 Extractor 析构而失效。

### 2.2 统一 token-major 布局

扩散阶段统一使用二维 `[tokens, channels]`：

- latent、condition、prediction：`w=16, h=T*H*W, elempack=1`
- DiT 输入：`w=33, h=T*H*W, elempack=1`
- text：`w=5120, h=text_tokens, elempack=1`

ncnn 上传二维 Mat 时可能自动改成 pack4，所以入口显式转换到 pack1。这样自定义
融合 shader 的地址是稳定的 `token * channels + channel`，不依赖隐式 packing。

### 2.3 sampler Vulkan pipeline

新增六个 pipeline：condition 线性混合、33 通道拼接、CFG、CFG 统计归约、CFG
rescale、Euler 更新。CFG reduction 使用一个 256 线程 workgroup 计算 positive 和
guided prediction 的 sum/square_sum；Euler 利用公式化简为：

```text
x_next = x_current + (next_timestep - timestep) / 1000 * prediction
```

### 2.4 VAE Vulkan 边界

Encoder 的 32 通道 moments 不再下载。posterior shader 直接读取 mean/logvar，应用
clamp、exp、噪声和 scaling，并同时从 C,T,H,W 转成 token-major。Decoder 边界执行
逆 scaling 和反向布局变换，随后直接把 `VkMat` 输入 decoder Net。

CPU 仍生成 posterior、initial、augment 三组高斯噪声，顺序与旧实现一致。这保证 seed
可复现；改成 GPU RNG 会改变参考结果，应作为单独功能处理。

## 3. 提交与验证

### 提交一：共享 VkMat 执行边界

- commit：`fd1d2e6`
- RTX 6000D Vulkan CTest：8/8
- 320x240 对应完整 DiT 与旧调度输出逐字节一致

### 提交二：采样循环 VkMat 常驻

- commit：`b997d6c`
- Vulkan sampler 专项测试覆盖 condition、拼接、CFG rescale、Euler
- 全量 CTest：9/9
- 两步 CFG=2.5、rescale=0.4 的完整 Engine 运行通过

### VAE 边界专项结果

测试尺寸为输入 `T=5,H=64,W=64`：

| 边界 | max_abs_error | cosine |
|---|---:|---:|
| Encoder Mat vs VkMat | 1.19209e-7 | 1.0 |
| Decoder Mat vs VkMat | 2.95788e-6 | 1.0 |

旧 VAE Mat 边界与全 VkMat Engine 最终输出比较：NRMSE 0.003641、cosine 0.99999344。
边界本身的误差很小，最终差异来自微小浮点误差经过 DiT 和 decoder 放大。

### 动态尺寸与性能冒烟测试

服务器硬件为 NVIDIA RTX 6000D 86GB，FP32 storage/arithmetic，resident DiT，一步采样：

| 输入 T,H,W | VAE encode | DiT | VAE decode | 总耗时 |
|---|---:|---:|---:|---:|
| 5,64,64 | 125.5 ms | 220.5 ms | 266.4 ms | 612.6 ms |
| 9,80,96 | 342.1 ms | 363.4 ms | 800.0 ms | 1507.2 ms |
| 5,240,320 | 1968.3 ms | 1719.8 ms | 4685.5 ms | 8381.2 ms |

`5,64,64` streaming 模式约 18.35 秒，主要消耗在每次读取 32 个 block 权重，不适合作为
高性能默认路径，但 VkMat 数据流仍可运行。

## 4. 传输审计

`process_vulkan()` 中的 feature tensor 传输为：

```text
入口：1 个批次，上传 video/posterior noise/initial noise/augment noise/text
VAE encode -> sampler -> 所有 DiT step -> VAE decode：0 次中间下载
出口：1 次，下载最终 decoded video
```

兼容 API `SeedVR2DiT::forward_vulkan(Mat, Mat, Mat&)` 仍包含入口上传和出口下载，供测试
和旧调用方使用；Engine 使用的是下层 `forward_vkmat()`，不会走兼容边界。

## 5. 当前不足与下一步

1. 完整尺寸 DiT Vulkan 相对 PyTorch 的既有 NRMSE 约 0.21，主要问题仍在 window
   attention 数值实现，不是本阶段的调度传输造成。
2. resident block 之间仍有 GPU queue submit/wait，用来保护短生命周期 workspace；
   它不是 CPU/GPU 数据传输，但会增加同步开销。
3. 当前是 FP32 pack1 主路径。开启 FP16 storage/arithmetic 后，新增 sampler/VAE boundary
   shader 仍需单独执行数值和性能验收。
4. 预处理与后处理仍在 CPU，但只位于 API 首尾，不造成模型组件之间的往返。
5. 下一性能优先级应为修复/重写 window attention，然后减少 resident block 同步，最后
   优化 VAE Convolution3D 和 FP16/packing。
