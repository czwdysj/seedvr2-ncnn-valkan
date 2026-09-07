# Convolution3D 的 Vulkan 后端实现（面试文档）

## 1. 功能与背景

给 ncnn 上游的 **Convolution3D（3D 卷积）补上 Vulkan 后端**，使 SeedVR2 的 VAE（encoder/decoder 共 **62 个 3D 卷积**）不再 fallback 到 CPU，实现「全程不回退 CPU」。

**背景**：ncnn 原生只有 2D/1D/DepthWise 卷积的 Vulkan 实现，`layer_registry_vulkan[]` 里 `{"Convolution3D", 0}`（Vulkan 创建函数为 nullptr）。VAE 是视频生成模型，用大量 3×3×3 时空卷积（权重最大 512×512×3×3×3 = 7077888 float ≈ 28MB/层）。若无 Vulkan 后端，每次 forward 都会触发 GPU↔CPU 搬运 + CPU 计算，Vulkan 加速对 VAE 基本失效。

## 2. 输入输出定义

| 项 | 定义 |
|---|---|
| 输入 | 4D Mat `[w, h, d, c]`（w 宽、h 高、d 深度/帧、c 通道）|
| 输出 | 4D Mat `[outw, outh, outd, num_output]` |
| 权重 | `[num_output, channels, maxk]`，maxk = kd·kh·kw，内层顺序 (kd, kh, kw) |
| 偏置 | `[num_output]`（bias_term 时）|

参数：kernel/stride/dilation 各三维 + pad（left/right/top/bottom/front/behind，支持显式与 SAME -233/-234）+ pad_value + bias_term。

## 3. 实现方法

### 3.1 注册机制（关键：零侵入）

ncnn 的层注册是**构建期自动生成**的。`ncnn_add_layer(Convolution3D)` 宏会检查 `layer/vulkan/convolution3d_vulkan.cpp` 是否存在：

```cmake
set(LAYER_VULKAN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/layer/vulkan/${name}_vulkan.cpp)
if(NCNN_VULKAN AND EXISTS ${LAYER_VULKAN_SRC})
    set(WITH_LAYER_${name}_vulkan 1)   # 存在即注册
```

所以**只需创建 `convolution3d_vulkan.cpp` + `.h`，重新 cmake 就会自动把 `{"Convolution3D", 0}` 变成 `{"Convolution3D", Convolution3D_vulkan_layer_creator}`**，无需手动改注册表、layer_declaration.h。

### 3.2 核心 shader：pack1 直接卷积（核心代码块）

每个 work item 计算一个输出元素，遍历 kernel 邻域与全部输入通道，边界判断填 pad_value（等价于 CPU 先 make_padding 再卷积）：

```glsl
void main()
{
    uint gi = gl_GlobalInvocationID.x;
    if (gi >= p.total) return;

    // 解码 (p_out, z_out, i_out, j_out)
    uint spatial_out = p.outw * p.outh * p.outd;
    uint p_out = gi / spatial_out;
    uint rem = gi % spatial_out;
    uint z_out = rem / (p.outh * p.outw);
    uint rem2 = rem % (p.outh * p.outw);
    uint i_out = rem2 / p.outw;
    uint j_out = rem2 % p.outw;

    // 输出地址用 cstep 对齐（VkMat 4D 通道间有 padding，非稠密！）
    uint out_addr = p_out * p.out_cstep + z_out * (p.outh * p.outw) + i_out * p.outw + j_out;

    afp sum = 0.0;
    if (p.bias_term == 1u) sum = buffer_ld1(bias_blob, p_out);

    for (uint q = 0; q < p.channels; q++)
        for (uint kd = 0; kd < p.kernel_d; kd++)
            for (uint kh = 0; kh < p.kernel_h; kh++)
                for (uint kw = 0; kw < p.kernel_w; kw++)
                {
                    int iz = int(z_out * p.stride_d + kd * p.dilation_d) - int(p.pad_front);
                    int ih = int(i_out * p.stride_h + kh * p.dilation_h) - int(p.pad_top);
                    int iw = int(j_out * p.stride_w + kw * p.dilation_w) - int(p.pad_left);

                    afp v = p.pad_value;          // 越界填 pad_value
                    if (iz >= 0 && iz < int(p.d) && ih >= 0 && ih < int(p.h) && iw >= 0 && iw < int(p.w))
                    {
                        uint addr = q * p.cstep + uint(iz) * (p.h * p.w) + uint(ih) * p.w + uint(iw);
                        v = buffer_ld1(bottom_blob, addr);
                    }
                    uint widx = p_out * (p.channels * p.maxk) + q * p.maxk
                        + (kd * p.kernel_h * p.kernel_w + kh * p.kernel_w + kw);
                    sum += v * buffer_ld1(weight_blob, widx);
                }
    buffer_st1(top_blob, out_addr, sum);
}
```

### 3.3 forward 里 SAME padding 的折算

`forward(VkMat)` 把 SAME padding（-233/-234）折算成显式 pad 量（与 CPU `make_padding` 一致），再传给 shader：

```cpp
if (pad_left == -233 && ...) {   // SAME_UPPER
    int wpad = kernel_extent_w + (w-1)/stride_w*stride_w - w;
    pad_left_ = wpad/2; pad_right_ = wpad - wpad/2;
    // ... h/d 同理
}
outw = (w + pad_left_ + pad_right_ - kernel_extent_w) / stride_w + 1;
```

## 4. 关键难点（面试最有价值）

### 4.1 难点 1：VkMat 4D 的 cstep padding（最大的坑）

**ncnn 的 4D Mat `[w,h,d,c]` 的 buffer 布局不是稠密的**：每个通道之间有 cstep 对齐 padding（cstep = alignSize(w·h·d, 4)）。当 `w·h·d` 不是 4 的倍数时（如 3×3×3=27 → cstep=28），通道间有 1 个元素的空隙。

我最初用稠密索引 `gi` 直接写 output（`buffer_st1(top_blob, gi, sum)`），在 `w·h·d` 是 4 的倍数时碰巧正确（无 padding），否则第 2 个通道起全部错位。**修复：输出地址用 `p_out * out_cstep + z_out*(outh*outw) + ...`**（cstep 对齐）。

**教训**：ncnn 4D Mat 的通道寻址必须用 cstep，不能用 `c * (w*h*d)` 这种稠密假设——这和之前 shuffle 算子踩的「cstep 对齐」坑同源，但这次发生在「写输出」而非「读输入」。

### 4.2 难点 2：bin 文件的 weight flag 格式

ncnn 的 bin 格式里，`mb.load(size, type=0)` 的权重**前面有 4 字节 flag**（fp32 = `0x00000000`，fp16 = `0x01306B47`，int8 = `0x000D4B38`）。`type=1`（bias 等）则直接读 float 无 flag。

我最初测试 bin 漏写 flag，导致 weight 被误判为量化数据而读取失败（`read quantization_value failed`）。**教训**：写 ncnn 测试 bin 时，type=0 权重前必须写 4 字节 flag。

### 4.3 难点 3：测试比较逻辑要跳过 padding

4D Mat 的 buffer 有 cstep padding，用稠密 `total()` 遍历比较会把 padding 间隙算进去（padding 是未初始化垃圾）。必须逐 `channel(ch).depth(d).row(y)` 访问比较。

## 5. 优化方向

| 优先级 | 优化 | 收益 |
|---|---|---|
| 1 | **im2col + gemm**（3D im2col 展开成矩阵乘，复用卷积 gemm shader）| 直接卷积每输出 512×27 次乘加，gemm 可用 tensor core/pack4，吞吐大幅提升 |
| 2 | **pack4**（通道 4 对齐时用 vec4 load/store）| 指令数↓、内存事务规整 |
| 3 | **1×1×1 pointwise 专用 gemm shader** | pointwise 本质是矩阵乘，走 gemm 比直接卷积快 |

当前实现是「正确性优先」的直接卷积（每输出元素遍历全部 kernel 邻域 + 通道），对 VAE 的 512 通道 3×3×3 卷积性能有限，但**保证不回退 CPU**。性能优化（im2col+gemm）是后续重点。

## 6. 测试验证

`tests/conv3d_vulkan_runner.cpp` 5 组参数：

```
[case 0] 3x3x3 s1 pad1 c2->3        : max|diff| = 0.000000 PASS
[case 1] 1x1x1 pointwise c4->2      : max|diff| = 0.000000 PASS
[case 2] 3x3x3 s2 pad1 c2->4        : max|diff| = 0.000000 PASS
[case 3] 3x3x3 SAME c1->1           : max|diff| = 0.000000 PASS
[case 4] 3x3x3 dil2 c3->2 非对称     : max|diff| = 0.000000 PASS
```

覆盖：3×3×3 与 1×1×1、stride 1/2、dilation、显式 pad 与 SAME pad、bias、多通道、非 4 倍数的 cstep 对齐边界。**全部 max|diff|=0，证明 Convolution3D 走 Vulkan 路径且与 CPU 逐元素一致（注册表 + support_vulkan=true 保证不回退）**。

## 7. 面试问答

**Q1：为什么不改注册表就能加 Vulkan 后端？**
ncnn 的层注册是构建期自动生成的：`ncnn_add_layer` 宏检测 `layer/vulkan/<name>_vulkan.cpp` 是否存在，存在就把 `layer_registry_vulkan` 里的 `0` 换成 `<name>_vulkan_layer_creator`。所以只需新建源文件 + 重新 cmake，零侵入。

**Q2：这个 shader 的性能瓶颈在哪？**
直接卷积每个输出元素要遍历全部 kernel 邻域（maxk）和全部输入通道，VAE 的 512 通道 3×3×3 卷积每输出 512×27≈1.4 万次乘加，是 compute-bound。优化方向是 im2col+gemm（把卷积转成矩阵乘，复用 gemm shader 和 tensor core）。

**Q3：为什么用边界判断而不是先 padding？**
3D 卷积的 padding 涉及深度维（pad_front/behind），ncnn 没有现成的 3D padding Vulkan 层。shader 里边界判断填 pad_value 等价于「先 pad 再卷积」，且省去一次中间张量搬运。SAME padding 在 forward 里折算成显式 pad 量。

**Q4：怎么保证「全程不回退 CPU」？**
`Convolution3D_vulkan` 构造函数设 `support_vulkan = true`，Net 在 forward 时据此走 Vulkan 路径；只有 activation_type != 0（融合激活，VAE 不使用）时才回退。加上数值对齐测试证明走 Vulkan 且正确。

**Q5：这个算子和前面六个自定义层的区别？**
前面六个是「seedvr2 项目自己的自定义层」（内嵌 shader），这个是「修改 ncnn 上游库」——通过 ncnn 的自动注册机制加进去，属于对第三方库的扩展。难点在于理解 ncnn 的 bin 格式（flag）、4D Mat 的 cstep 布局、以及构建期注册机制。
