# SeedVR2DiTBlock 自定义层报告

## 1. 层的职责

`SeedVR2DiTBlock` 实现 SeedVR2 3B 的一个完整多模态 Transformer Block。模型共有
32 个 Block，全部复用同一份 C++ 类，但每个 `.param/.bin` 保存各自配置和权重。

该层同时更新视频 token 和文本 token，包含：

- RMSNorm 和 AdaSingle 调制。
- 动态普通/shifted 三维窗口划分。
- 视频与文本 joint attention。
- Q/K head-wise RMSNorm 和 MM-RoPE。
- attention 输出投影、gate 和残差。
- SwiGLU MLP、gate 和残差。
- Block 10 以后的权重共享语义，以及 Block 31 的 `vid_only` 特例。

对应实现：

```text
src/seedvr2_dit_block.h
src/seedvr2_dit_block.cpp
cpp_runtime/dit_block_runner.cpp
cpp_runtime/dit_full_runner.cpp
```

## 2. 输入与输出

每个 Block 有四个输入、两个输出：

| blob | 逻辑形状 | NCNN `Mat` | 类型 |
| --- | --- | --- | --- |
| `vid` | `[L_vid,2560]` | `dims=2,w=2560,h=L_vid` | FP32 |
| `txt` | `[L_txt,2560]` | `dims=2,w=2560,h=L_txt` | FP32 |
| `emb` | `[15360]` | `dims=1,w=15360` | FP32 |
| `vid_shape` | `[T,H,W]` | `dims=1,w=3` | INT32 |
| `vid_out` | `[L_vid,2560]` | `dims=2,w=2560,h=L_vid` | FP32 |
| `txt_out` | `[L_txt,2560]` | `dims=2,w=2560,h=L_txt` | FP32 |

必须满足：

```text
L_vid = T * H * W
15360 = 2560 * 6
```

`T/H/W` 是 patch 后网格；`L_txt` 动态。Block 不改变 token 数或 hidden dimension。

## 3. 固定模型配置

| 参数 | 值 | 含义 |
| --- | ---: | --- |
| `dim` | 2560 | token hidden dimension |
| `heads` | 20 | attention head 数 |
| `head_dim` | 128 | 每个 head 的维度 |
| `mlp_hidden` | 6912 | SwiGLU 中间维度 |
| `norm_eps` | `1e-5` | RMSNorm epsilon |

每个分支保存六个 Ada bias：

```text
attn_shift, attn_scale, attn_gate
mlp_shift,  mlp_scale,  mlp_gate
```

并保存 QKV、attention 输出、Q/K norm、SwiGLU 三个 projection 的权重。

## 4. 32 个 Block 的差异

| Block | vid/txt 权重 | 窗口 | 特殊行为 |
| --- | --- | --- | --- |
| 0-9 | 独立权重 | 偶数普通、奇数 shifted | 标准双分支 |
| 10-30 | 共享权重 | 偶数普通、奇数 shifted | vid/txt 调用同一组权重 |
| 31 | 共享权重 | shifted | `vid_only` Ada/MLP 文本特例 |

Block 10-31 的 `.bin` 只保存一份 `all` 分支权重，C++ 的 `text_branch` 指向
`vid_weights`，不是复制第二份权重。

## 5. 为什么必须自定义实现

### 5.1 窗口数量和边界由运行时网格决定

窗口不是固定 `7x7` 或固定 token 数。它先按当前 `H*W` 计算一个面积归一化尺度，再
生成时间和空间窗口；奇数 Block 还使用半窗口偏移。窗口个数、边界窗口大小和每个
attention 序列长度都会随输入变化。

pnnx 无法把 Python list slice、动态窗口列表和变长 flatten/unflatten 转为一张通用
静态图。

### 5.2 每个视频窗口都要融合完整文本

对每个视频窗口，attention 序列为：

```text
[该窗口 video tokens, 全部 text tokens]
```

视频 token 只属于一个当前窗口；文本 token 会在每个窗口重复参与 attention，最后
文本结果再按窗口数求平均。这不是普通全局 MHA，也不是简单 cross-attention。

### 5.3 MM-RoPE 位置规则是多模态且动态的

视频使用局部三维 `(t,h,w)` 位置，文本在三个轴上使用语言 token 位置。视频时间
位置还加上 `L_txt` 偏移。普通 RoPE 算子无法直接表达这套三轴拼接规则。

### 5.4 必须复现精度边界和特殊 Block

PyTorch 在 attention 前显式转 BF16。CPU 基线需要在相同边界执行
round-to-nearest-even；Block 31 还包含看似异常但真实存在的文本双倍残差语义。
这些行为不能依赖通用转换器猜测。

因此该层是整个移植中最核心、最难替代的自定义边界。

## 6. 总体实现逻辑

一个标准 Block 的完整数据流为：

```text
vid/txt 输入
  -> 分支 RMSNorm
  -> attention Ada shift/scale
  -> vid/txt QKV InnerProduct
  -> 动态窗口生成与视频 token gather
  -> 每个窗口拼接完整文本 token
  -> Q/K head-wise RMSNorm + MM-RoPE + BF16 舍入
  -> joint scaled dot-product attention
  -> 视频 scatter + 文本跨窗口平均
  -> attention output InnerProduct
  -> attention gate + residual
  -> 分支 RMSNorm
  -> MLP Ada shift/scale
  -> SwiGLU 三个 InnerProduct
  -> MLP gate + residual
  -> vid_out/txt_out
```

实现中大型矩阵乘法由 NCNN 原生层完成，动态索引、精度边界、窗口调度和模型特例由
自定义 C++ 负责。

## 7. 动态窗口算法

对 patch 网格 `[T,H,W]`：

```text
scale = sqrt(3600 / (H*W))
resized_h = round_to_even(H * scale)
resized_w = round_to_even(W * scale)
window_h = ceil(resized_h / 3)
window_w = ceil(resized_w / 3)
window_t = ceil(min(T,30) / 4)
```

奇数 Block 在某个窗口尺寸小于对应输入尺寸时使用 `0.5 * window_size` 偏移。边界窗口
通过 clamp 截断，因此不同窗口可以具有不同 video token 数。

窗口遍历顺序为 `w -> h -> t`，窗口内 token 顺序为 `t -> h -> w`。该顺序影响
RoPE 位置和输出回填，不能随意调整。

## 8. Attention 实现结构

### 8.1 Attention 前 Ada

先分别对 vid/txt 执行 token-wise RMSNorm，再使用 `emb` 和分支 bias：

```text
x_attn[c] = norm(x)[c] * (emb[c*6+1] + attn_scale[c])
                         + emb[c*6+0] + attn_shift[c]
```

QKV projection 使用原生 NCNN `InnerProduct`：

```text
[L,2560] -> [L,7680]
```

### 8.2 Q/K RMSNorm 和 MM-RoPE

Q/K 按 20 个 head 拆成 128 维，每个 head 独立 RMSNorm 并乘学习 gamma。

RoPE 使用三个轴，每个轴旋转 42 维，即 21 对 sin/cos：

```text
axis 0: dimension 0..41
axis 1: dimension 42..83
axis 2: dimension 84..125
dimension 126..127 不旋转
```

视频位置：

```text
(L_txt + local_t, local_h, local_w)
```

文本位置：

```text
(text_index, text_index, text_index)
```

Q/K/V 在进入 attention 前舍入到 BF16 可表示值，但仍存放在 FP32 `Mat/vector` 中。

### 8.3 Joint attention

每个 head、每个窗口执行：

```text
sequence = concat(window_video, full_text)
scores   = Q * K^T / sqrt(128)
probs    = stable_softmax(scores)
context  = probs * V
```

视频结果直接回填到原 token 索引。文本结果在所有窗口累加，最后除以窗口数。attention
context 在写出边界再次执行 BF16 舍入，以匹配 PyTorch SDPA 附近的精度行为。

### 8.4 Attention 输出和残差

输出 projection 使用原生 `InnerProduct(2560,2560)`，随后：

```text
y[c] = projected[c] * (emb[c*6+2] + attn_gate[c]) + residual[c]
```

## 9. MLP 实现结构

MLP 是 SwiGLU：

```text
gate  = Linear_gate(x)  [L,6912]
value = Linear_in(x)    [L,6912]
fused = SiLU(gate) * value
out   = Linear_out(fused) [L,2560]
```

MLP 前使用 `emb[c*6+3]` 和 `emb[c*6+4]` 做 shift/scale，输出使用
`emb[c*6+5]` 和 learned `mlp_gate` 做 gate，再加残差。三个 Linear 均复用原生
`InnerProduct`。

## 10. Block 31 的兼容行为

最后一层的 PyTorch Ada 模块配置为 `vid_only=True`：

- 文本分支仍做 RMSNorm、QKV、joint attention 和输出 projection。
- 文本不应用 attention Ada shift/scale/gate。
- 文本不执行普通 MLP。
- 为匹配原始 MMModule 外层残差路径，最终文本输出为 attention 残差结果的 2 倍。

这不是通用 Transformer 设计，而是当前 checkpoint 和源码的兼容要求。

## 11. 验证证据

两组真实网格的 32 个 Block 均使用各自 PyTorch 输入独立验证，共 64 次：

| 示例 | 最差 Block | video NRMSE | 同项 text NRMSE |
| --- | ---: | ---: | ---: |
| `[2,15,20]` | 27 | `6.26984e-4` | `2.83983e-6` |
| `[2,16,30]` | 30 | `6.54065e-4` | `2.15035e-6` |

本机下载后重新执行 Block 00，得到：

```text
video NRMSE = 1.10496e-4
text  NRMSE = 7.76658e-5
```

完整 32 层串联结果：

| 网格 | NRMSE | cosine |
| --- | ---: | ---: |
| `[2,15,20]` | `0.016998` | `0.999855` |
| `[2,16,30]` | `0.012347` | `0.999924` |

独立层误差很低，完整串联误差主要是 BF16 舍入、CPU/CUDA GEMM 和 SDPA 累积差异。

## 12. 当前限制和 Vulkan 迁移

- batch=1，文本 embedding 预计算。
- CPU attention 是正确性基线，窗口内复杂度为 `O(S^2*head_dim)`。
- 当前只有 `Mat` forward；即使内部 Linear 可创建 pipeline，也没有 `VkMat` 调度。
- Vulkan 不能写成一个巨大 shader。建议拆为 Ada/RMSNorm、QKV GEMM、RoPE、窗口
  gather、分块 attention、scatter/text reduce、输出 GEMM、SwiGLU 和残差多个 pipeline。
- 必须让 vid/txt/embedding 在 32 层间保持 GPU 常驻；若每层回落 CPU，传输开销会
  抵消全部加速收益。
- 需要同时验证普通窗口、shifted 窗口、边界小窗口、共享权重和 Block 31。
