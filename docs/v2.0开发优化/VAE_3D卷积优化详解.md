# SeedVR2 VAE 3D 卷积优化详解

本文是 `docs/v2.0开发优化/` 中唯一的 VAE 优化文档，按实际开发顺序记录
Convolution3D Vulkan 的瓶颈、代码变化、正确性测试、性能数据和失败实验。

## 1. 测试条件

- GPU：NVIDIA GeForce RTX 5090，32607 MiB
- 驱动：595.71.05
- 计算：ncnn Vulkan，FP32 storage/FP32 arithmetic
- 模型：动态 VAE decoder，35 个 Convolution3D
- 输入一：`[16,2,30,40]`，输出 `[3,5,240,320]`
- 输入二：`[16,2,32,60]`，输出 `[3,5,256,480]`
- 每项预热 2 次，正式运行 5 次，取中位数

## 2. 原始瓶颈

原始 shader 的一个 invocation 只计算一个输出标量：

```cpp
for (uint q = 0; q < channels; q++)
    for (uint k = 0; k < kernel_size; k++)
        sum += input[q][k] * weight[out][q][k];
```

不同输出通道重复读取相同输入；权重是离散标量读取；每个线程包含很长的
`Cin×Kd×Kh×Kw` 循环。高分辨率 decoder 将这些问题放大。

## 3. 优化一：四输出通道并行

### 修改前

```cpp
afp sum = bias[out];
sum += input_value * buffer_ld1(weight_blob, weight_index);
buffer_st1(output_blob, output_address, sum);
```

### 修改后

```cpp
afpvec4 sum4 = load_four_biases();
sum4 += input_value * afpvec4(weight0, weight1, weight2, weight3);
store_valid_output_lanes(sum4);
```

一个输入值同时贡献给四个输出通道，线程数约减为四分之一，输入读取被四个输出
复用。尾组逐 lane 判断，因此 RGB 三通道仍然正确。

结果：

| 输出 | 原始 | 四输出并行 | 提升 |
|---|---:|---:|---:|
| 5×240×320 | 4074.497 ms | 2067.058 ms | 1.971× |
| 5×256×480 | 6529.968 ms | 3272.562 ms | 1.996× |

## 4. 优化二：权重 pack4

四输出版本仍需从四段相距很远的权重内存执行四次 load：

```cpp
wt.x = buffer_ld1(weight, (out + 0) * stride + offset);
wt.y = buffer_ld1(weight, (out + 1) * stride + offset);
wt.z = buffer_ld1(weight, (out + 2) * stride + offset);
wt.w = buffer_ld1(weight, (out + 3) * stride + offset);
```

模型上传时将 CPU 权重从 `[out,in,kernel]` 重排为
`[out/4,in,kernel,4]`：

```cpp
for (int output_group = 0; output_group < ceil(out / 4); ++output_group)
    for (int input_channel = 0; input_channel < channels; ++input_channel)
        for (int kernel_index = 0; kernel_index < maxk; ++kernel_index)
            for (int lane = 0; lane < 4; ++lane)
                packed[group][input][kernel][lane]
                    = original[group * 4 + lane][input][kernel];
```

shader 随后只需一次连续 16 字节读取：

```cpp
uint packed_index = group * channels * maxk + input_channel * maxk + kernel_index;
afpvec4 weight4 = buffer_ld4(weight_blob, packed_index);
sum4 += input_value * weight4;
```

权重重排只在模型上传时执行一次，不进入 forward。它把四次离散标量 load 变成一次
连续向量 load，同时减少三套权重地址计算。

| 输出 | 四输出并行 | 权重 pack4 | 本阶段提升 | 相对原始 |
|---|---:|---:|---:|---:|
| 5×240×320 | 2067.058 ms | 1403.545 ms | 1.473× | 2.903× |
| 5×256×480 | 3272.562 ms | 2262.042 ms | 1.447× | 2.887× |

## 5. 优化三：workgroup 调优

RTX 5090 subgroup 为 32。原 `local_size_x=64` 每个 workgroup 只有两个 subgroup；
调整为 128 后每组包含四个 subgroup，让调度器在该长循环 kernel 上获得稍好的占用率。

```cpp
// 修改前
pipeline->set_optimal_local_size_xyz(64, 1, 1);

// 修改后
pipeline->set_optimal_local_size_xyz(128, 1, 1);
```

| 输出 | local=64 | local=128 | 提升 | checksum |
|---|---:|---:|---:|---:|
| 5×240×320 | 1403.545 ms | 1396.789 ms | 1.005× | -207777.197 |
| 5×256×480 | 2262.042 ms | 2245.263 ms | 1.007× | -336291.263 |

收益只有约 0.5–0.7%，属于小幅调度优化。累计相对原始版本约 2.9 倍。

## 6. 正确性验证

`seedvr2_conv3d_vulkan_runner` 五组 CPU/Vulkan 对齐测试最大绝对误差均为 0，覆盖：

- 1×1×1 和 3×3×3；
- stride 1/2、dilation 2；
- SAME 和非对称 padding；
- 非四倍数输入输出通道。

完整 decoder 两个尺寸的 checksum 在三项保留优化中保持一致。

## 7. 复现命令

```bash
cmake --build build --target seedvr2_conv3d_vulkan_runner seedvr2_vae_decoder_bench -j2
./build/seedvr2_conv3d_vulkan_runner

./build/seedvr2_vae_decoder_bench \
  models/vae_dynamic/seedvr2_vae_decoder_dynamic.ncnn.param \
  models/vae_dynamic/seedvr2_vae_decoder_dynamic.ncnn.bin 2 30 40 16
```

## 8. 下一步

主激活仍是 pack1。真正的激活 pack4 需要 Convolution3D、GroupNorm、SpatialAttention
和 SpaceTimeShuffle 共同支持 pack4；只改卷积会在动态层前后产生大张量 pack/unpack。
下一阶段应先为三个动态层增加独立 pack4 数值测试，再启用 VAE 网络级 packing。
