# SeedVR2 DiT 完整尺寸 Attention 精度验证报告

本文记录 RTX 5090 上 SeedVR2 3B DiT attention 语义修复后的数值证据，用于判断当前 NCNN CPU/Vulkan 实现是否存在算子、张量布局或条件分支错误。本文不把不同后端经过 32 层后的浮点累积差异当作性能问题，也不以逐元素完全一致作为当前验收条件。

## 1. 结论

当前完整尺寸 window attention 未发现实现语义错误，可以进入下一阶段的 Linux 构建和项目可用性完善。

- Block 0、9、10、31 的正负文本分支均通过，Vulkan 对 PyTorch 的最大 NRMSE 为 `7.66201e-4`。
- 320x240 与 480x270 两个完整 DiT 案例均通过，Vulkan 对 PyTorch 的 NRMSE 分别为 `0.0180403` 和 `0.00807038`。
- 两个完整案例的 cosine similarity 分别为 `0.999837262` 和 `0.999967437`。
- 负文本完整分支 NRMSE 为 `0.0140261`，cosine similarity 为 `0.999901629`。
- resident 与 streaming 的完整 DiT 输出逐元素一致。
- 所有参与验收的输出均为有限值，不含 NaN 或 Inf。

## 2. 问题根因与修复

诊断初期曾按旧版 `models/dit` 推断文本 Q/K 不应执行 RoPE。检查实际导出所用的 `models/dit_v2` 后确认，当前 3B 模型使用 `NaMMRotaryEmbedding3d(mmrope3d)`：

- 视频 token 使用 `(text_length + t_local, y_local, x_local)` 三轴坐标；
- 文本 token 使用 `(text_index, text_index, text_index)` 三轴坐标；
- 视频和文本 Q/K 都执行 RoPE。

因此没有错误地删除文本 RoPE。最终定位出的真实问题是 Vulkan QKV shader 将视频分支的 `norm_q/norm_k` gamma 同时用于文本分支。修复后，视频和文本分别绑定各自训练得到的归一化参数，CPU 与 Vulkan 保持同一模型语义。

## 3. 验收门槛

自动化工具 `tools/test_seedvr2_dit_ncnn.py` 使用以下默认门槛：

| 检查项 | 门槛 | 目的 |
|---|---:|---|
| 单个 block 对 PyTorch | NRMSE <= `0.002` | 捕获 QKV、RoPE、窗口布局、归一化、残差等实现错误 |
| 完整 32-block DiT 对 PyTorch | NRMSE <= `0.02` | 容纳 FP16 权重、BF16 attention 与 GEMM 累加顺序差异 |
| 完整 32-block DiT 对 PyTorch | cosine >= `0.9998` | 防止方向和整体语义发生明显偏移 |
| 所有输出 | finite | 直接拒绝 NaN 和 Inf |

旧错误曾产生约 `0.21` 的 NRMSE，仍会被当前门槛明确拦截。当前门槛不是降低正确性要求，而是把“算子语义错误”和“不同后端浮点累计差异”分开判断。

## 4. 单 Block 对齐

### 4.1 正文本分支

Block 0、9、10、31 的 CPU/PyTorch 和 Vulkan/PyTorch 比较全部通过 `0.002` 门槛，Vulkan 各输出的最大 NRMSE 低于 `7.8e-4`。详细逐项结果由测试 JSON 保存；此处只记录已经复核的范围，避免把不同测试轮次的打印值混为同一张表。

### 4.2 负文本分支

| Block | CPU vid NRMSE | CPU txt NRMSE | Vulkan vid NRMSE | Vulkan txt NRMSE |
|---:|---:|---:|---:|---:|
| 0 | `1.10481e-4` | `5.14395e-5` | `1.54868e-4` | `7.07152e-5` |
| 9 | `2.12764e-4` | `3.10286e-4` | `2.42408e-4` | `2.73717e-4` |
| 10 | `2.86034e-4` | `3.10199e-6` | `3.35057e-4` | `3.15996e-6` |
| 31 | `5.59160e-4` | `1.22409e-5` | `7.66201e-4` | `1.24388e-5` |

QKV projection、Q/K RMSNorm 和 mmrope3d 候选坐标也做了中间张量诊断。其中 RoPE 后 Q、K 的 NRMSE 分别为 `1.28e-6` 和 `2.19e-5`，说明坐标构造和旋转公式与 PyTorch 一致。

## 5. 完整 DiT 对齐

| 案例 | 分支 | Vulkan/PyTorch NRMSE | cosine | CPU/PyTorch NRMSE |
|---|---|---:|---:|---:|
| 320x240，5 帧 | 正文本 | `0.0180403` | `0.999837262` | `0.0169980` |
| 480x270，5 帧 | 正文本 | `0.00807038` | `0.999967437` | `0.0123472` |
| 320x240，5 帧 | 负文本 | `0.0140261` | `0.999901629` | 未单独记录完整输出 |

320x240 案例中 Vulkan NRMSE 比 CPU 基线高 `0.0010423`；480x270 案例中 Vulkan 比 CPU 基线低 `0.00427682`。两者均未显示 Vulkan 存在系统性语义偏差。

完整 Vulkan 与完整 CPU 的直接 NRMSE 为 `0.012665`（320x240）和 `0.011504`（480x270）。考虑到独立 block 对 PyTorch 均低于 `8e-4`，且两个后端各自都能稳定对齐 PyTorch，这一差异归类为 32 层 FP16/BF16 和 GEMM 累加顺序造成的后端数值漂移，而不是 attention 实现错误。后续可以单独做数值精度优化，但不阻塞项目正确性建设。

## 6. 测试范围与限制

本阶段已经覆盖：

- RTX 5090 NVIDIA Vulkan；
- 两个真实视频对应的完整模型尺寸；
- 空间窗口、时间窗口和文本联合 attention；
- 正文本和负文本条件；
- resident 与 streaming 两种模型调度；
- CPU、Vulkan 和 PyTorch 三方比较。

本阶段尚未覆盖完整长视频分块、所有 sampler step、VAE 与最终视频编码的端到端验收，也未验证 AMD/Intel Vulkan。这些属于后续项目完整性任务，不能由本报告外推为已经完成。

## 7. 复现方式

在模型、参考张量和 runner 路径准备完成后执行：

```bash
SEEDVR2_DEVICE=vulkan SEEDVR2_DIT_RESIDENT=1 \
python tools/test_seedvr2_dit_ncnn.py \
  --blocks 0,9,10,31 \
  --branch neg \
  --reference-dir /path/to/reference \
  --model-dir /path/to/dit_full_fp16 \
  --block-runner /path/to/seedvr2_dit_block_runner \
  --full-runner /path/to/seedvr2_dit_full_runner \
  --output /tmp/seedvr2_accuracy.json
```

工具默认同时检查数值有限性、单 block NRMSE、完整 DiT NRMSE 和 cosine similarity，退出码非零表示至少一项未通过。

## 8. 下一步

下一项只处理 Linux/WSL2 项目构建完整性：从干净 clone 开始验证依赖检查、ncnn patch 幂等应用、CMake 构建和轻量 CTest。该阶段仍不做性能优化。
