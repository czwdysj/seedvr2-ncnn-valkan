# DiT Mat/VkMat 往返消除实现与验证报告

本文记录 SeedVR2 NCNN 单次 DiT 的 Vulkan 零中间传输改造。输入是 CPU 侧
`latent[33,T,H,W]`、`text[L,5120]` 和 timestep；输出是 CPU 侧
`latent[16,T,H,W]`。报告覆盖调度结构、资源生命周期、真实 RTX 5090 验证、
已知数值限制和下一阶段工作，可直接用于项目复盘与面试说明。

## 1. 改造前的问题

旧调度在 input head、32 个 Transformer block 和 output head 之间使用 `Mat`。
每个 `Extractor::extract(Mat&)` 都会把结果下载到 CPU，下一个子网又重新上传。
此外三个自定义层的 Vulkan forward 会下载 shape/timestep，并调用
`submit_and_wait()`，因此单次 DiT 存在大量隐藏同步和数据传输。

## 2. 最终数据流

```text
CPU latent/text/timestep/shape
  -> 一批入口上传
  -> DiT input head (VkMat)
  -> block 00 ... block 31 (VkMat)
  -> DiT output head (VkMat)
  -> 一次最终下载
  -> CPU output
```

`SeedVR2DiT::forward()` 根据 `RuntimeOptions.device` 自动选择 CPU 或 Vulkan；
公开 `SeedVR2Engine` API 没有变化。Vulkan 路径集中取得同一设备的 blob/staging
allocator，并显式设置给每个 extractor，保证跨 Net 的 `VkMat` 在 extractor
析构后仍然有效。

三个自定义层新增 CPU 运行时元数据注入：

- input：原始 `T/H/W` 和 timestep；
- block：patch 后 `T/H/W`；
- output：patch 后 `T/H/W`。

shape/timestep blob 为兼容现有 param 保留，但 Vulkan shader 不再下载它们。
三个层中已经没有 `record_download()`、`submit_and_wait()` 或 `reset()`。

## 3. Resident 与 Streaming

两种模式都保持中间 feature 为 `VkMat`，审计结果均为：

```text
entry_upload_commands=1
intermediate_download_commands=0
final_download_commands=1
queue_submissions=33
```

streaming 每层执行后必须等待 GPU，随后才能销毁临时 Net 的 pipeline 和权重。
resident 虽然权重常驻，但当前 block 内部有大量短生命周期 workspace；把 32 层
全部延迟到一次提交会出现 workspace 生命周期冲突并产生错误结果。因此当前也在
block 边界提交，只取消数据下载。后续若要合并提交，必须先让命令对象持有全部
workspace，或为跨 block 命令批次设计不可复用的 arena。

## 4. 零拷贝暴露的 attention 问题

窗口 attention 为每个窗口分配一段 `vid_out_ws`，但只写该窗口覆盖的 token；
`window_sum` 随后会读取所有窗口段。旧实现依赖未初始化显存恰好为零，独立
extractor allocator 偶尔掩盖了问题。共享 allocator 后旧值被稳定复用，完整 DiT
误差明显扩大。

修复方式是新增 `zero_buffer` Vulkan shader，在 attention 前显式清空
`vid_out_ws`。修复后 320×240 示例的新 VkMat 输出与旧 Mat 边界输出 bit-exact，
resident 与 streaming 输出也 bit-exact。480×270 示例中，清零后的结果比依赖
未初始化显存的旧路径稳定；旧路径不再适合作为该尺寸的正确性参考。

## 5. RTX 5090 验证结果

环境：NVIDIA GeForce RTX 5090、Vulkan 1.4、FP32 storage/arithmetic、58 个文本
token、timestep=1000。

| 验证项 | 结果 |
|---|---:|
| DiT input Vulkan 单层 | PASS |
| DiT block Vulkan 动态窗口用例 | 4/4 PASS |
| DiT output Vulkan 单层 | PASS |
| 示例 1，latent `[33,2,30,40]` | 完整 32 层 PASS |
| 示例 2，latent `[33,2,32,60]` | 完整 32 层 PASS |
| 示例 1 resident vs streaming | bit-exact |
| 示例 1 新 VkMat vs 旧 Mat 边界 | bit-exact |
| 中间 GPU -> CPU 下载 | 0 |

性能使用示例 1、2 次预热、5 次正式运行取中位数：

| 模式 | 旧 Mat 边界 | 新 VkMat | 变化 |
|---|---:|---:|---:|
| resident | 1555.34 ms | 1383.05 ms | 加速约 11.1% |
| streaming | 未做同口径旧版 5 次统计 | 20494.2 ms | 权重重载仍是主体 |

## 6. 数值限制

本次目标是消除调度层往返，不能把既有 Vulkan attention 的完整模型累计误差
算作已经解决。对 PyTorch reference：

| 示例 | NRMSE | cosine |
|---|---:|---:|
| 320×240 | 0.212671 | 0.977159 |
| 480×270 | 0.215568 | 0.976619 |

CPU NCNN 基线在示例 1 的 NRMSE 为 0.016998。因此当前 VkMat 调度已经通过
“不引入额外误差”和“零中间下载”验证，但整个 Vulkan DiT 尚未达到最终 PyTorch
精度标准。下一数值瓶颈是完整尺寸 window attention，而不是 Mat/VkMat 调度。

## 7. 后续顺序

1. 用 block 0、9、10、31 的真实权重和参考张量定位 Vulkan attention 误差。
2. 修正 attention 后重新冻结完整 DiT 的 NRMSE/cosine 阈值。
3. 为 resident 设计命令期 workspace arena，再尝试多 block 单次提交。
4. 将 CFG 和 Euler sampler 改为 VkMat，消除采样步之间的 DiT 出口下载和入口上传。
