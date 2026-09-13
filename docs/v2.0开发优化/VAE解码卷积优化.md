# VAE decoder 瓶颈与 Convolution3D 优化

## 结论

VAE decoder 的首要瓶颈是项目自定义的 ncnn `Convolution3D_vulkan`。decoder 图中
共有 35 个 `Convolution3D`；旧 shader 让每个 invocation 只计算一个输出标量，
不同输出通道会重复读取相同输入。RTX 5090 上仅修改该 shader 后，完整 decoder
在两个真实输出尺寸上分别提升 1.97 倍和 2.00 倍，证明判断成立。

本轮没有启用 FP16，没有改变权重文件和网络 param，也没有引入近似算法。

## 原实现为什么慢

旧实现的一个线程对应 `(输出通道, T, H, W)` 中的一个元素，并串行遍历全部输入
通道和 `kernel_d × kernel_h × kernel_w`。它存在以下问题：

- 输出通道之间无法共享同一输入元素，输入被重复从全局显存读取。
- pack1 标量权重读取和标量累加无法充分利用 RTX 5090 的向量吞吐。
- 线程内循环很长，包含输入边界判断、地址计算和权重索引计算。
- 没有 shared memory tiling、im2col/GEMM 或矩阵计算单元路径。
- decoder 在高分辨率阶段反复调用该算子，空间尺寸放大后代价快速增长。

GroupNorm、空间注意力和时空 shuffle 也有优化空间，但只替换卷积 shader 就让完整
decoder 接近 2 倍加速，因此它们不是本阶段的第一瓶颈。

## 本轮实现

修改文件为 `patches/0001-feat-Convolution3D-Vulkan-pack1.patch`。张量仍保持
`elempack=1`，每个 invocation 改为在寄存器中的 `afpvec4` 同时累加最多四个输出
通道：

```text
同一个输入值 v
    ├─ v * weight[out + 0] -> sum.x
    ├─ v * weight[out + 1] -> sum.y
    ├─ v * weight[out + 2] -> sum.z
    └─ v * weight[out + 3] -> sum.w
```

这样线程数约降为四分之一，并让四个输出通道复用一次输入读取。输出通道不是 4 的
倍数时使用边界判断，因此最终 RGB 三通道卷积和任意通道数仍可执行。计算次序在每个
输出通道内没有改变，权重布局也保持 `[out, in, kd, kh, kw]`。

## 测试环境

- GPU：NVIDIA GeForce RTX 5090，32607 MiB
- 驱动：595.71.05
- 后端：ncnn Vulkan，FP32 storage / FP32 arithmetic，packing 关闭
- 模型：`vvzc/seedvr2-ncnn-models` 的动态 VAE decoder
- 权重 SHA256：`94893f4171a5bfc338e4793cc33d565e1b5be0be5fff08feb768629c86c2ee6c`
- 方法：预热 2 次，正式运行 5 次，取中位数

benchmark 输入是确定性随机的 `[C=16,T=2,H,W]` latent。计时覆盖 ncnn Extractor
的 decoder 推理和相同的边界传输，因此新旧版本可直接比较；不包含视频编解码、DiT
和 sampler。

## 性能对比

| 输出尺寸 | 旧 pack1 单输出 | 四输出通道 | 加速比 | 输出 checksum |
|---|---:|---:|---:|---:|
| 5×240×320 | 4074.497 ms | 2067.058 ms | 1.971× | -207777.197 / 一致 |
| 5×256×480 | 6529.968 ms | 3272.562 ms | 1.996× | -336291.263 / 一致 |

代表性 `128→128, 3×3×3, T×H×W=8×16×16` 微基准从 `2.237 ms` 降到
`1.683 ms`，提升 1.329 倍。完整 decoder 收益更高，说明高分辨率层中的输入复用
更能覆盖调度和固定开销。

## 正确性验证

`seedvr2_conv3d_vulkan_runner` 的 5 组 CPU/Vulkan 对齐测试全部通过，最大绝对误差
均为 0，覆盖：

- 3×3×3、1×1×1；
- stride 1、stride 2；
- dilation 2；
- SAME padding 和非对称 padding；
- 输出通道为 1、2、3、4，包括非 4 倍数尾通道。

两个完整 decoder A/B 测试的 checksum 逐位显示一致。该结果证明新 shader 没有改变
当前 FP32 路径的输出；后续仍应在完整视频闭环中保留 PyTorch reference 对比。

## 复现

```bash
cmake --build build --target seedvr2_vae_decoder_bench seedvr2_conv3d_vulkan_runner -j2
./build/seedvr2_conv3d_vulkan_runner

# 320×240 输出
./build/seedvr2_vae_decoder_bench \
  models/vae_dynamic/seedvr2_vae_decoder_dynamic.ncnn.param \
  models/vae_dynamic/seedvr2_vae_decoder_dynamic.ncnn.bin 2 30 40

# 480×256 输出
./build/seedvr2_vae_decoder_bench \
  models/vae_dynamic/seedvr2_vae_decoder_dynamic.ncnn.param \
  models/vae_dynamic/seedvr2_vae_decoder_dynamic.ncnn.bin 2 32 60
```

## 尚未解决

当前只是寄存器级四输出并行，主张量仍为 pack1，权重仍进行四次标量读取，也没有利用
Tensor Core。下一步优先实现输入/权重 pack4，并按 3×3×3 主路径做专用 shader。
im2col+GEMM 理论吞吐更高，但会增加大尺寸临时显存；shared-memory direct convolution
显存低但调优复杂。两条路线应在逐层 timestamp 数据完成后再选择。
