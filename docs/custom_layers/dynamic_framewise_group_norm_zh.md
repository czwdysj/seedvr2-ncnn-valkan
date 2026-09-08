# DynamicFramewiseGroupNorm 自定义层报告

## 1. 层的职责

`DynamicFramewiseGroupNorm` 实现 SeedVR2 VAE 中的“逐帧 GroupNorm”。它对视频的
每一个时间帧独立计算归一化统计量，不能把不同帧放进同一个 group 里共同计算均值
和方差。

动态 VAE 模型中共有 50 个该层：Encoder 21 个，Decoder 29 个。不同实例的通道数
可能是 128、256 或 512，但都使用 32 组和 FP32 CPU 计算。

对应实现：

```text
src/dynamic_framewise_group_norm.h
src/dynamic_framewise_group_norm.cpp
```

## 2. 输入与输出

batch 固定为 1，因此 NCNN 张量不保存 batch 维。

| 项目 | PyTorch 逻辑形状 | NCNN 物理表示 | 数据类型 |
| --- | --- | --- | --- |
| 输入 `bottom_blob` | `[1,C,T,H,W]` | `Mat(w=W,h=H,d=T,c=C)` | FP32 |
| 输出 `top_blob` | `[1,C,T,H,W]` | `Mat(w=W,h=H,d=T,c=C)` | FP32 |

输入约束：

- `dims == 4`。
- `elempack == 1`。
- `C` 必须等于该层权重的通道数。
- `C % 32 == 0`。
- `T/H/W` 在运行时读取，不写死在 `.param` 中。

该层不改变形状，只改变数值。`Mat.d` 明确解释为时间帧数，不能把它当作普通
三维特征的 depth 并参与同一次归一化。

## 3. 参数和权重

导出脚本把 PyTorch `GroupNorm` 保留为 pnnx module operator。NCNN `.param` 的
shape 参数用于恢复通道数，`.bin` 按以下顺序保存：

```text
bias_data   [C]
weight_data [C]
```

运行时固定配置：

```text
groups = 32
eps    = 1e-6
```

`weight_data` 是 affine scale，`bias_data` 是 affine bias。

## 4. 为什么需要自定义实现

### 4.1 原模型的统计范围不是普通 5D GroupNorm

PyTorch 导出包装先执行：

```text
[B,C,T,H,W]
  -> permute [B,T,C,H,W]
  -> reshape [B*T,C,H,W]
  -> GroupNorm
  -> reshape/permute 回 [B,C,T,H,W]
```

也就是说每个 frame 都是一个独立样本。若直接对 NCNN 的 4D
`Mat(w=W,h=H,d=T,c=C)` 使用普通 GroupNorm，常见实现会把 `T*H*W` 一起放入
统计范围，导致不同帧相互影响，语义已经改变。

### 4.2 pnnx 无法稳定表达动态 `B*T`

若让 trace 把 `T` 个 frame 拆成固定 slice，再分别执行 GroupNorm，导出图只对 trace
时的帧数有效。换成另一个 `T` 后，slice 数量、reshape 常量和 concat 数量都不再匹配。

### 4.3 不能仅靠手写图外逻辑

GroupNorm 在 VAE 残差块内部出现 50 次。如果每次都把 tensor 拉出 NCNN 图、在调用
层手工处理再送回，会破坏子图执行和后续 Vulkan 的 GPU 常驻设计。把动态语义封装为
正式 NCNN layer，才能保持 `.param` 拓扑和运行时接口稳定。

因此必须自定义的是“逐帧动态统计边界”，不是 GroupNorm 数学公式本身。

## 5. 实现逻辑

每个 `(frame, group)` 可以独立计算，CPU 代码在这两个维度并行：

```text
输入 [C,T,H,W]
  -> 遍历 frame = 0..T-1
  -> 遍历 group = 0..31
  -> 读取该帧该组的 [C/32,H,W]
  -> 计算 sum 和 square_sum
  -> mean = sum / N
  -> variance = square_sum / N - mean^2
  -> inv_std = 1 / sqrt(max(variance,0) + eps)
  -> 对组内各通道应用 affine
  -> 写回相同位置
输出 [C,T,H,W]
```

其中：

```text
N = (C / groups) * H * W
y[c,t,h,w] = (x[c,t,h,w] - mean[t,g]) * inv_std[t,g]
             * weight[c] + bias[c]
```

均值和平方和使用 `double` 累加，最终结果写回 FP32。这是为了降低不同 CPU 归约顺序
造成的误差，而不是改变模型精度定义。方差在开平方前执行 `max(variance, 0)`，防止
浮点消减误差产生很小的负值。

## 6. 原生算子与自定义边界

该层完整负责动态 GroupNorm；前后的 `Convolution3D`、残差加法和 Swish 仍由 NCNN
原生算子执行。自定义层没有卷积权重，也不负责因果时间 padding。

## 7. 验证证据

该层当前没有单独逐实例 dump 50 组输出，验证来自完整动态 VAE：

| 测试 | 输入 | Encoder/Decoder 最坏 max abs |
| --- | --- | ---: |
| case 0 | `[1,3,6,37,53]` | `7.43121e-5` |
| case 1 | `[1,3,10,41,67]` | `8.49962e-5` |

四项动态 Encoder/Decoder 测试都满足 `max_abs <= 5e-4`。这能证明包含本层的完整路径
与 PyTorch 对齐，但不能把上述整网误差归因成某一个 GroupNorm 实例的独立误差。

## 8. 当前限制和 Vulkan 迁移

- 仅支持 batch=1、FP32、pack1。
- 仅实现 CPU `forward(const Mat&, Mat&, Option&)`。
- 尚未实现 `upload_model`、`create_pipeline` 和 `forward_vkcompute`。
- Vulkan 版本应以 `(frame,group)` 为归约单元，先计算 mean/variance，再执行 affine；
  不能直接调用会跨 `T` 归约的原生 GroupNorm shader。
- Vulkan 对齐应至少覆盖 `T=1`、多个 `T`、奇数 `H/W` 和不同通道数。
