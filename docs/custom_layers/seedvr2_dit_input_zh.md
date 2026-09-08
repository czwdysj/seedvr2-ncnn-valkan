# SeedVR2DiTInput 自定义层报告

## 1. 层的职责

`SeedVR2DiTInput` 是 SeedVR2 3B DiT 的统一输入边界。它把图外准备好的视频
latent/condition、文本 embedding、扩散 timestep 和原始 latent 形状转换为 32 个
Transformer Block 使用的三类特征：2560 维视频 token、2560 维文本 token 和
15360 维 Ada embedding。

对应实现：

```text
src/seedvr2_dit_input.h
src/seedvr2_dit_input.cpp
cpp_runtime/dit_input_runner.cpp
```

## 2. 输入接口

该层有四个输入 blob，batch 固定为 1：

| blob | 逻辑形状 | NCNN `Mat` | 类型 | 含义 |
| --- | --- | --- | --- | --- |
| `vid` | `[T*H*W,33]` | `dims=2,w=33,h=T*H*W` | FP32 | 16 通道 noisy latent、16 通道 condition latent 和 1 通道条件标记 |
| `txt` | `[L_txt,5120]` | `dims=2,w=5120,h=L_txt` | FP32 | 预计算正向或负向文本 embedding |
| `timestep` | `[1]` | `dims=1,w>=1` | FP32 | 当前扩散时间步 |
| `vid_shape` | `[3]` | `dims=1,w=3` | INT32 | 原始 latent 网格 `[T,H,W]` |

`vid` 的行顺序为 `t -> y -> x`，每行保存该位置的 33 个通道。要求：

```text
vid.h == T * H * W
H % 2 == 0
W % 2 == 0
T,H,W > 0
```

`L_txt` 动态，不写死为某个提示词长度。

## 3. 输出接口

| blob | 逻辑形状 | NCNN `Mat` | 类型 |
| --- | --- | --- | --- |
| `vid_out` | `[T*(H/2)*(W/2),2560]` | `dims=2,w=2560,h=L_vid` | FP32 |
| `txt_out` | `[L_txt,2560]` | `dims=2,w=2560,h=L_txt` | FP32 |
| `emb` | `[15360]` | `dims=1,w=15360` | FP32 |
| `patched_shape` | `[T,H/2,W/2]` | `dims=1,w=3` | INT32 |

其中 `15360 = 2560 * 6`，每个通道对应 attention 和 MLP 的 shift、scale、gate
六个调制量。

## 4. 参数、权重和原生算子

层参数：

```text
dim            = 2560
video_channels = 33
text_channels  = 5120
sinusoidal_dim = 256
embedding_dim  = 15360
```

`.bin` 中五组线性层按以下顺序加载：

```text
video_projection       132  -> 2560，带 bias
text_projection        5120 -> 2560，带 bias
time_projection_in      256 -> 2560，带 bias
time_projection_hidden 2560 -> 2560，带 bias
time_projection_out    2560 -> 15360，带 bias
```

这些大矩阵乘法均复用 NCNN 原生 `InnerProduct`。自定义代码负责动态 patchify、
sin/cos timestep embedding、SiLU 调度和 shape blob。

## 5. 为什么需要自定义实现

### 5.1 patchify 的 token 数和索引依赖运行时形状

`T/H/W` 由输入视频决定，`L_vid=T*(H/2)*(W/2)` 也是运行时值。pnnx trace 容易把
reshape、permute 和 patch 数固化为示例网格，不能稳定覆盖 `[2,15,20]` 与
`[2,16,30]` 等不同 patch 网格。

### 5.2 shape 是后续窗口 attention 的控制输入

普通 tensor 图只传递数值，后续 Block 还必须知道 token 对应的三维 `[T,H,W]`。
该层显式输出 INT32 `patched_shape`，避免依赖静态 param 或从 token 数猜测空间比例。

### 5.3 timestep 路径包含动态图外调度

sin/cos 频率由标量 timestep 在运行时计算，然后依次经过三层 Linear 和两次 SiLU。
Linear 可以原生转换，但动态标量构造、频率排列和最终 Ada 布局需要一个稳定边界。

因此并不是 5 个 Linear 必须手写，而是 patchify、shape 传播和 timestep 调度必须由
自定义层统一管理。

## 6. 实现逻辑

### 6.1 视频 patchify

每个 `2x2` 空间 patch 保持时间维不变：

```text
[T,H,W,33]
  -> [T,H/2,W/2,2,2,33]
  -> [T*(H/2)*(W/2),132]
  -> InnerProduct(132,2560)
```

patch 内展开顺序是：

```text
dy -> dx -> channel
```

即 `(0,0)` 的 33 通道、`(0,1)` 的 33 通道、`(1,0)`、`(1,1)`。该顺序必须与
PyTorch einops 和 checkpoint 的 projection 权重一致。

### 6.2 文本投影

```text
[L_txt,5120] -> InnerProduct -> [L_txt,2560]
```

文本长度保持不变。

### 6.3 timestep embedding

对 `i=0..127`：

```text
frequency_i = exp(-log(10000) * i / 128)
sinusoidal[i]       = sin(timestep * frequency_i)
sinusoidal[i + 128] = cos(timestep * frequency_i)
```

之后执行：

```text
[256]
  -> Linear 256->2560 -> SiLU
  -> Linear 2560->2560 -> SiLU
  -> Linear 2560->15360
```

### 6.4 shape 输出

```text
[T,H,W] -> [T,H/2,W/2]
```

## 7. 验证证据

输入头在两个真实视频网格上独立对齐：

| patch 网格 | video NRMSE | text NRMSE | embedding NRMSE |
| --- | ---: | ---: | ---: |
| `[2,15,20]` | `2.00045e-4` | `1.10514e-5` | `2.20952e-5` |
| `[2,16,30]` | `2.01292e-4` | `1.10514e-5` | `2.20952e-5` |

两个网格具有不同视频 token 数，证明输出不是固定 shape 模型。

## 8. 当前限制和 Vulkan 迁移

- batch=1，输入 `H/W` 必须为偶数。
- 文本 embedding 必须在图外预计算。
- 当前只有 CPU `forward`，内部原生 `InnerProduct` 也按 CPU Option 创建。
- Vulkan 版本需要为 patchify、sin/cos、SiLU 和 shape 传播提供 GPU/CPU 明确边界；
  五个 Linear 应直接复用 NCNN Vulkan `InnerProduct`，不应重写 GEMM shader。
- INT32 shape 很小，可保留 CPU 控制数据，但需要避免为大型 video/txt tensor 往返拷贝。
