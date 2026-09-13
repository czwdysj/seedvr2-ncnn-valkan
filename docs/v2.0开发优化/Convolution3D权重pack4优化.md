# Convolution3D 权重 pack4 优化

## 1. 本阶段解决什么问题

第一阶段已经让一个 Vulkan invocation 同时计算四个输出通道，但权重仍沿用原始
`[out,in,kernel]` 排列。四个输出通道的权重位于四段相距很远的显存中，所以 shader
必须执行四次标量读取：

```cpp
wt.x = buffer_ld1(weight, (out + 0) * output_stride + offset);
wt.y = buffer_ld1(weight, (out + 1) * output_stride + offset);
wt.z = buffer_ld1(weight, (out + 2) * output_stride + offset);
wt.w = buffer_ld1(weight, (out + 3) * output_stride + offset);
```

这段代码虽然使用 `vec4` 做计算，但显存访问仍是四条离散 load，无法形成一次自然
对齐的 16 字节向量读取。

本阶段只改变 GPU 权重副本的布局，不修改模型文件、param、激活张量布局和卷积数学
含义。因此它可以在不改其他 VAE layer 的情况下独立验证。

## 2. 优化步骤

### 第一步：保留原始模型权重

模型加载得到的 CPU 权重仍是：

```text
[output_channel][input_channel][kernel_d][kernel_h][kernel_w]
```

这保证现有 `.bin` 文件完全兼容，不需要重新导出模型。

### 第二步：上传前重排权重

`upload_model()` 在 CPU 上创建只供 Vulkan 使用的 pack4 副本：

```cpp
for (int output_group = 0; output_group < ceil(out / 4); ++output_group)
    for (int input_channel = 0; input_channel < channels; ++input_channel)
        for (int kernel_index = 0; kernel_index < maxk; ++kernel_index)
            for (int lane = 0; lane < 4; ++lane)
                packed[output_group][input_channel][kernel_index][lane]
                    = original[output_group * 4 + lane][input_channel][kernel_index];
```

优化后 GPU 权重布局为：

```text
[output_group][input_channel][kernel_index][4 output lanes]
```

不足四通道的最后一组补零。这个重排只在模型上传时执行一次，不进入每次 forward 的
计时路径。

### 第三步：shader 改为一次向量读取

优化后的核心代码是：

```cpp
uint group = output_channel_base / 4u;
uint packed_index = group * (channels * maxk) + kernel_offset;
afpvec4 weight4 = buffer_ld4(weight_blob, packed_index);
sum4 += input_value * weight4;
```

权重 buffer 声明同步从标量数组改为向量数组：

```cpp
// 优化前
readonly buffer weight_data { sfp weight_blob[]; };

// 优化后
readonly buffer weight_data { sfpvec4 weight_blob[]; };
```

### 第四步：保留尾通道保护

最终 RGB 卷积只有三个输出通道。上传阶段会将第四 lane 补零，写回阶段仍检查真实
`num_output`，因此不会创建或写入不存在的第四输出通道。

## 3. 为什么会更快

以一次输入标量参与四个输出通道计算为例：

| 项目 | 第一阶段 | 权重 pack4 后 |
|---|---:|---:|
| 权重 load 指令 | 4 次标量 load | 1 次向量 load |
| 权重地址 | 相隔 `channels×kernel` | 连续 16 字节 |
| shader 权重索引 | 4 套输出地址计算 | 1 套 group 地址计算 |
| 数学乘加次数 | 4 | 4 |
| 计算结果 | FP32 | FP32 |

优化没有减少必要的乘加次数，收益来自更连续的显存访问、更少的 load 指令和更少的地址
计算。VAE decoder 有 35 个 Convolution3D，且高分辨率层会为大量空间位置重复执行这段
内循环，所以单次循环节省会被显著放大。

## 4. RTX 5090 实测结果

统一条件：FP32、预热 2 次、正式 5 次取中位数，输入为确定性随机 latent。

| 输出 | 原始单输出 | 四输出并行 | 再加权重 pack4 | 本阶段提升 | 累计提升 |
|---|---:|---:|---:|---:|---:|
| 5×240×320 | 4074.497 ms | 2067.058 ms | 1403.545 ms | 1.473× | 2.903× |
| 5×256×480 | 6529.968 ms | 3272.562 ms | 2262.042 ms | 1.447× | 2.887× |

代表性 Conv3D 微基准为 `1.496 ms`；第一阶段是 `1.683 ms`，原始实现是
`2.237 ms`。完整 decoder 的收益更明显，因为它包含 35 层卷积和更大的空间尺寸。

三个阶段 checksum 分别保持：

- 320×240：`-207777.197`
- 480×256：`-336291.263`

## 5. 正确性验证

`seedvr2_conv3d_vulkan_runner` 五组测试全部通过，CPU/Vulkan 最大绝对误差为 0。
测试覆盖 1×1×1、3×3×3、stride、dilation、SAME/非对称 padding，以及非四倍数
输出通道。两个完整 decoder 尺寸的 checksum 与前两阶段一致。

## 6. 为什么本阶段没有直接启用输入 pack4

当前三个 VAE 动态层都明确要求 `elempack=1`：

- `DynamicFramewiseGroupNorm`
- `DynamicFramewiseSpatialAttention`
- `DynamicSpaceTimeShuffle`

如果只让 Convolution3D 输出 pack4，ncnn 必须在这些动态层前后反复执行 pack4↔pack1
转换。这样会增加完整激活张量的读写，可能抵消卷积收益，而且错误的 4D channel/cstep
解释会直接破坏数值。

正确的下一步是先让三类动态层理解 pack4 的通道语义，再让连续卷积、激活和归一化
链路保持 pack4；只有模型输入和最终 RGB 输出需要转换。该工作必须作为独立提交测试，
不能通过简单设置 `support_vulkan_packing=true` 完成。
