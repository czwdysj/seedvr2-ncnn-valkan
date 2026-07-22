# DynamicFramewiseSpatialAttention 自定义层报告

## 1. 层的职责

`DynamicFramewiseSpatialAttention` 实现 VAE bottleneck 中的逐帧空间自注意力。
Encoder 和 Decoder 各有一个实例，当前模型通道数均为 512。

该层只在单帧的 `H*W` 空间 token 之间建立关系，不跨时间帧做 attention。它同时
包含 attention 前 GroupNorm、Q/K/V 投影、scaled dot-product attention、输出投影
和残差连接，是一个完整的 VAE attention block。

对应实现：

```text
custom_layers/dynamic_framewise_spatial_attention.h
custom_layers/dynamic_framewise_spatial_attention.cpp
```

## 2. 输入与输出

| 项目 | PyTorch 逻辑形状 | NCNN 物理表示 | 数据类型 |
| --- | --- | --- | --- |
| 输入 `bottom_blob` | `[1,C,T,H,W]` | `Mat(w=W,h=H,d=T,c=C)` | FP32 |
| 输出 `top_blob` | `[1,C,T,H,W]` | `Mat(w=W,h=H,d=T,c=C)` | FP32 |

运行时定义：

```text
C = 512
N = H * W
frame_tokens = [N,C]
```

输入和输出形状完全相同。`T/H/W` 动态；每一帧独立执行一次长度为 `N` 的单头
attention。

输入约束：`dims == 4`、`elempack == 1`、`C % 32 == 0`，且 `C` 必须与权重一致。

## 3. 参数和权重

`.bin` 的读取顺序必须与 pnnx module operator 写入顺序完全一致：

```text
norm_bias    [C]
norm_weight  [C]
k_bias       [C]
k_weight     [C,C]
out_bias     [C]
out_weight   [C,C]
q_bias       [C]
q_weight     [C,C]
v_bias       [C]
v_weight     [C,C]
```

GroupNorm 使用 32 组、`eps=1e-6`。attention scale 为：

```text
scale = 1 / sqrt(C)
```

当前是单头空间 attention，因此 head dimension 就是 `C`。

## 4. 为什么需要自定义实现

### 4.1 token 数由运行时 `H*W` 决定

原始 PyTorch 路径先把 `[B,C,T,H,W]` 变成 `[B*T,C,H,W]`，再把每帧空间展开
成 `H*W` token。pnnx trace 容易把 reshape 的 `B*T`、token 数和每帧 slice 固化为
示例尺寸，无法让同一 param 同时处理不同 `T/H/W`。

### 4.2 不是普通固定长度 MultiHeadAttention

该模块包含逐帧 GroupNorm、独立 Q/K/V 线性层、单头缩放、输出投影和残差。直接替换
为 NCNN 通用 MHA 不仅要解决动态序列长度，还必须匹配原权重布局、单头语义和残差
边界，pnnx 当前不会自动得到这一等价图。

### 4.3 必须保持“不跨帧”

把 `[T*H*W,C]` 当成一条序列虽然容易实现，却会让一个 frame 读取其他 frame，改变
VAE 的空间 bottleneck 语义。自定义层通过显式 frame 循环保证 attention mask 的
实际效果是严格的 block diagonal。

## 5. 实现逻辑

每个 frame 执行以下流程：

```text
输入 [C,H,W]
  -> 32 组逐帧 GroupNorm
  -> 转为 token-major [N,C]
  -> Q = X * Wq^T + bq
  -> K = X * Wk^T + bk
  -> V = X * Wv^T + bv
  -> scores = Q * K^T / sqrt(C)
  -> stable softmax(scores)
  -> attended = probabilities * V
  -> projected = attended * Wo^T + bo
  -> projected + 原始输入残差
  -> 写回 [C,H,W]
```

稳定 softmax 的计算顺序：

```text
m_i = max_j(scores[i,j])
p[i,j] = exp(scores[i,j] - m_i) / sum_k(exp(scores[i,k] - m_i))
```

点积、GroupNorm 统计量和 softmax 分母使用 `double` 累加，存储仍为 FP32。Q/K/V
和输出线性层权重按 `[out_channel,in_channel]` 解释。

## 6. 复杂度和内存

设 `N=H*W`：

- Q/K/V 和输出投影复杂度约为 `O(T*N*C^2)`。
- attention 点积和加权求和约为 `O(T*N^2*C)`。
- 当前 CPU 实现为每个 query 临时分配长度 `N` 的 score 数组，优先保证可读性和对齐。

VAE bottleneck 的空间已经被下采样，因此 `N` 小于原视频像素数，但该层仍是需要重点
迁移到 Vulkan 的热点。

## 7. 原生算子与自定义边界

当前 CPU 基线把完整 attention block 放在自定义层内部，未调用 NCNN 原生
`InnerProduct`。这是为了首先固定动态逐帧语义。后续 Vulkan 版本可以把四个线性投影
拆成原生 Vulkan GEMM，custom shader 只负责 frame 调度、softmax 和布局转换。

## 8. 验证证据

Encoder 和 Decoder 各包含一个该层，均随完整动态 VAE 验证：

- Encoder 最坏 max abs：`5.53131e-5`。
- Decoder 最坏 max abs：`8.49962e-5`。
- Decoder 最坏 RMSE：`9.93329e-6`。

这些是整网误差，不是该 attention 层的独立误差。当前不足是尚未保存本层 Q/K/V、
softmax 和投影后的独立 PyTorch/NCNN 对齐张量；实现 Vulkan 前应补充这组测试。

## 9. 当前限制和 Vulkan 迁移

- batch=1、FP32、pack1。
- 普通 `O(N^2)` 单头 attention，没有 FlashAttention。
- 不支持跨帧 attention，这是模型语义而不是功能缺失。
- 尚无 `VkMat` 权重上传和 `forward_vkcompute`。
- Vulkan 建议拆成 GroupNorm、QKV GEMM、分块 `QK^T`、稳定 softmax、`PV`、输出
  GEMM 和残差 kernel，并保持每帧独立 dispatch。
