# DynamicSpaceTimeShuffle 自定义层报告

## 1. 层的职责

`DynamicSpaceTimeShuffle` 实现 VAE Decoder 上采样中的 learned `1x1x1 Conv3D`
和 channel-to-space-time 重排。它相当于同时支持空间 PixelShuffle 和时间
PixelShuffle，并复现 SeedVR2 时间上采样后删除重复头帧的特殊规则。

Decoder 中共有三个实例：前两个同时进行时间和空间 2 倍上采样，最后一个只进行
空间 2 倍上采样。

对应实现：

```text
src/dynamic_space_time_shuffle.h
src/dynamic_space_time_shuffle.cpp
```

## 2. 输入与输出

设空间倍率 `s=2`，时间倍率为 `r_t`：

| 项目 | PyTorch 逻辑形状 | NCNN 物理表示 |
| --- | --- | --- |
| 输入 | `[1,C,T,H,W]` | `Mat(w=W,h=H,d=T,c=C)` |
| projection 临时结果 | `[1,C*s*s*r_t,T,H,W]` | 层内计算，不输出 |
| 输出 | `[1,C,T_out,H*s,W*s]` | `Mat(w=W*s,h=H*s,d=T_out,c=C)` |

输出时间长度：

```text
r_t == 1: T_out = T
r_t == 2: T_out = 2*T - 1
```

当前三个实例的通道和倍率为：

| 实例 | 输入通道 | projection 通道 | `r_t` | 输出通道 |
| --- | ---: | ---: | ---: | ---: |
| upsample 0 | 512 | 4096 | 2 | 512 |
| upsample 1 | 512 | 4096 | 2 | 512 |
| upsample 2 | 256 | 1024 | 1 | 256 |

## 3. 参数和权重

pnnx 在 `.param` 中写入 bias 和 Conv3D weight 的 shape。C++ 由 shape 恢复：

```text
projected_channels = bias_shape[0]
in_channels        = weight_shape[1]
r_t = projected_channels / (in_channels * s * s)
```

`.bin` 顺序：

```text
bias_data   [projected_channels]
weight_data [projected_channels,in_channels]
```

因为卷积核为 `1x1x1`，权重在每一个 `(t,h,w)` 位置只混合通道，不读取邻域。

## 4. 为什么需要自定义实现

### 4.1 NCNN 的普通 PixelShuffle 只表达空间重排

这里除了 `H/W`，还要把 projection 通道的一部分展开到时间轴。普通二维
PixelShuffle 无法表达 `C*s^2*r_t -> C` 的三轴重排。

### 4.2 输出时间是动态表达式 `2*T-1`

pnnx 可以在一个 trace 尺寸下生成固定 Reshape/Permute/Crop，但 Crop 的结束位置和
输出 depth 会绑定示例 `T`。更换视频长度后，固定图无法自动得到新的 `2*T-1`。

### 4.3 删除的不是简单最后一帧

时间展开产生 raw frame `0..2*T-1`。原模型保留 raw frame 0，删除 raw frame 1，
再保留 raw frame 2 以后的位置。这个“删除重复头帧”规则必须在通道映射时准确实现，
不能用普通 resize 或尾部 crop 替代。

## 5. 实现逻辑

对每个输入位置 `(c_in,t,y,x)`，先计算所有 projection channel：

```text
projection[p,t,y,x] = bias[p] + sum_c(input[c,t,y,x] * weight[p,c])
```

当前实现没有显式分配 projection tensor，而是在写目标位置时直接完成通道混合。
projection channel 的编码顺序为：

```text
p = ((offset_y * s + offset_x) * r_t + offset_t) * C + output_channel
```

对应输出坐标：

```text
raw_t = t * r_t + offset_t
out_y = y * s + offset_y
out_x = x * s + offset_x
```

当 `r_t > 1` 时：

```text
raw_t == 1: 跳过
raw_t > 1 : output_t = raw_t - 1
其他      : output_t = raw_t
```

完整数据流：

```text
输入 [C,T,H,W]
  -> learned 1x1x1 channel projection
  -> 解析 projection channel 为 [dy,dx,dt,c]
  -> 写入放大的 T/H/W 坐标
  -> 时间倍率为 2 时删除 raw frame 1
输出 [C,2*T-1,2H,2W] 或 [C,T,2H,2W]
```

## 6. 为什么没有对应的下采样自定义层

Encoder 下采样可由原生 `Convolution3D` 的 stride、因果前置 padding 和普通裁剪表达，
输出尺寸由卷积规则自然计算，不需要执行 channel-to-time 重排，也没有删除 raw frame 1
的特殊语义。因此下采样继续使用 NCNN 原生算子，只有 Decoder 的时空 shuffle 需要
自定义。

## 7. 验证证据

该层随完整 Decoder 的两组动态 latent 验证：

| Decoder 输入 | Decoder 输出 | max abs |
| --- | --- | ---: |
| `[1,16,2,4,6]` | `[1,3,5,32,48]` | `7.43121e-5` |
| `[1,16,3,5,8]` | `[1,3,9,40,64]` | `8.49962e-5` |

输出帧数分别满足 `4*2-3=5` 和 `4*3-3=9`，说明两个时间 2 倍层串联后的因果长度
规则正确。当前数值是完整 Decoder 误差，尚未为三个 shuffle 实例分别保存独立输出。

## 8. 当前限制和 Vulkan 迁移

- batch=1、FP32、pack1，空间倍率固定为 2。
- 当前 projection 是 C++ 手写通道点积，没有复用原生 `Convolution3D` pipeline。
- Vulkan 版本可采用两个阶段：原生/自定义 `1x1x1` GEMM，再执行无分支的重排 shader；
  也可融合为一个 kernel，但必须评估重复读取权重的代价。
- 动态输出分配必须使用运行时 `T/H/W`，并保留删除 raw frame 1 的规则。
