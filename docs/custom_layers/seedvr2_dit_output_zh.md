# SeedVR2DiTOutput 自定义层报告

## 1. 层的职责

`SeedVR2DiTOutput` 是 32 个 Transformer Block 之后的动态输出头。它把 2560 维
视频 token 转换为 16 通道 diffusion prediction，并把输入头生成的 `2x2` patch
重新展开到空间网格。

该层包含 affine RMSNorm、output Ada、原生 Linear、动态 unpatch 和输出 shape
传播。文本 token 不进入该层。

对应实现：

```text
src/seedvr2_dit_output.h
src/seedvr2_dit_output.cpp
cpp_runtime/dit_output_runner.cpp
```

## 2. 输入接口

| blob | 逻辑形状 | NCNN `Mat` | 类型 |
| --- | --- | --- | --- |
| `vid` | `[T*H*W,2560]` | `dims=2,w=2560,h=T*H*W` | FP32 |
| `emb` | `[15360]` | `dims=1,w=15360` | FP32 |
| `vid_shape` | `[T,H,W]` | `dims=1,w=3` | INT32 |

这里的 `H/W` 是 patch 后尺寸，必须满足 `vid.h == T*H*W`。

## 3. 输出接口

| blob | 逻辑形状 | NCNN `Mat` | 类型 |
| --- | --- | --- | --- |
| `vid_out` | `[T*(2H)*(2W),16]` | `dims=2,w=16,h=T*2H*2W` | FP32 |
| `output_shape` | `[T,2H,2W]` | `dims=1,w=3` | INT32 |

若还原成 PyTorch 视频布局，`vid_out` 可解释为 `[1,16,T,2H,2W]`。该层输出是
DiT 的 16 通道预测，不是 RGB 视频；后续还要经过 sampler 和 VAE Decoder。

## 4. 参数和权重

固定配置：

```text
dim             = 2560
output_channels = 16
norm_eps        = 1e-5
```

`.bin` 顺序：

```text
norm_weight  [2560]
output_shift [2560]
output_scale [2560]
projection.weight [64,2560]
projection.bias   [64]
```

`64 = 16 * 2 * 2`，即每个 patch token 先预测四个空间位置的 16 通道。projection
复用 NCNN 原生 `InnerProduct(2560,64)`。

## 5. 为什么需要自定义实现

### 5.1 unpatch 输出行数依赖运行时 `T/H/W`

projection 之后必须从 `[T*H*W,64]` 动态重排到 `[T*2H*2W,16]`。pnnx trace
生成的 Reshape/Permute 很容易固定某一个 token 数和网格，无法复用到不同分辨率。

### 5.2 必须显式传播输出 shape

`T*H*W` 不能唯一确定 H 和 W。输出层需要输入 INT32 `vid_shape` 并输出
`[T,2H,2W]`，后续 sampler/布局恢复不能只依赖扁平 token 数猜测。

### 5.3 必须兼容真实 PyTorch Cache 行为

完整 PyTorch 图中，`vid_out_ada` 与 Block 0 使用相同 cache key
`emb_repeat_0_vid`。实际执行时输出 Ada 复用了按 `[2560,2,3]` 解释的 Block 0
embedding，而不是按模块表面定义的 `[5120,1,3]` 重新排列。

如果按照“看起来更合理”的声明形状实现，单独模块可能能运行，但不能与真实完整图
对齐。自定义层明确按 `embedding[channel*6 + offset]` 读取真实缓存语义。

因此 Linear 本身不需要自定义，动态 unpatch、shape 传播和 cache 兼容语义必须自定义。

## 6. 实现逻辑

### 6.1 affine RMSNorm

对每个 token 独立计算：

```text
inv_rms = 1 / sqrt(mean(x^2) + 1e-5)
normalized[c] = x[c] * inv_rms * norm_weight[c]
```

平方和使用 `double` 累加。

### 6.2 output Ada

```text
ada[c] = normalized[c] * (emb[c*6+1] + output_scale[c])
                         + emb[c*6+0] + output_shift[c]
```

### 6.3 projection

```text
[L_vid,2560] -> InnerProduct -> [L_vid,64]
```

### 6.4 2x2 unpatch

每个 projection 行的 64 个数按以下顺序解释：

```text
dy -> dx -> output_channel
```

目标行索引：

```text
patch = (t*H + y)*W + x
output_index = (t*(2H) + y*2 + dy)*(2W) + x*2 + dx
```

最终写入 `output[output_index,channel]`，并输出 shape `[T,2H,2W]`。

## 7. 原生算子与自定义边界

- 原生：`2560 -> 64` 的 `InnerProduct`。
- 自定义：RMSNorm、cache-compatible Ada、动态 unpatch、INT32 shape 输出。

将大矩阵乘法留给 NCNN，避免在自定义层中维护另一套 GEMM；自定义代码只负责模型
特有语义和动态布局。

## 8. 验证证据

输出头在两种 patch 网格上独立验证：

| 输入网格 | 输出网格 | NRMSE | max abs |
| --- | --- | ---: | ---: |
| `[2,15,20]` | `[2,30,40]` | `7.76631e-5` | `< 6.84e-4` |
| `[2,16,30]` | `[2,32,60]` | `7.80274e-5` | `< 6.84e-4` |

两种网格输出 token 数分别为 2400 和 3840，证明 unpatch 不是固定尺寸实现。

## 9. 当前限制和 Vulkan 迁移

- batch=1、FP32 `Mat` 接口；权重文件允许 FP16 storage。
- 尚未实现 `VkMat` forward。
- Vulkan 版本应复用原生 Vulkan `InnerProduct`，增加 RMSNorm+Ada kernel 和 unpatch
  scatter kernel。
- `vid_shape` 可作为小型 CPU push constant，但 2560 维 token 不应下载回 CPU。
- Vulkan 对齐必须单独覆盖 cache-compatible embedding 索引，避免只验证 shape 正确。
