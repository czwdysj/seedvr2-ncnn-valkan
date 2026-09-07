# SeedVR2DiTOutput 的 Vulkan 实现（面试文档）

## 1. 功能

`SeedVR2DiTOutput` 是 SeedVR2 3B DiT 的**输出层**，位于 32 层 transformer block 之后，负责把 DiT 输出的视频 token 还原成 VAE 可解码的 latent。它是 `SeedVR2DiTInput` 的**对称逆操作**。

完整数据流：

```
输入视频 token [T*H*W, dim=2560]  ─┐
timestep embedding [dim*6=15360] ──┤
patch 形状 [T, H, W] ──────────────┘
        │
        ├─① affine RMSNorm（逐 token，非逐通道）
        ├─② output Ada 调制（用 embedding 的 scale/shift 调制）
        ├─③ 输出投影 dim → oc*4（复用原生 InnerProduct）
        ├─④ 2x2 unpatchify（把 4 通道拆回 2x2 空间位置）
        │
        ▼
输出 latent [oc=16, T*2H*2W] + 形状 [T, 2H, 2W]
```

**逐 token 的 affine RMSNorm** 是这层的第一个自定义点：DiTInput 的 patchify 把视频切成 token，这里先对每个 token 独立做 RMSNorm（ncnn 原生 RMSNorm 是按 batch 或按通道的，没有逐 token 语义）。

## 2. 输入输出定义

| 项 | 定义 |
|---|---|
| 输入 0 `video` | `[tokens, dim]` 2D，tokens = T·H·W，dim=2560（DiT 隐藏维）|
| 输入 1 `embedding` | `[dim*6]` 1D，即 `[dim, 6]` 扁平，最后一层 block 的 adaLN 调制参数 |
| 输入 2 `shape` | `[3]` 1D int 位模式，存 `[T, H, W]`（patch 网格尺寸）|
| 输出 0 `vid_out` | `[oc, T·2H·2W]` 2D，oc=16，每个空间位置 16 通道连续（channel-major）|
| 输出 1 `output_shape` | `[3]` 1D int，存 `[T, 2H, 2W]` |

权重（param/bin 顺序）：`norm_weight[dim]`、`output_shift[dim]`、`output_scale[dim]`、`projection.weight[dim, oc*4]`、`projection.bias[oc*4]`。

## 3. 实现方法

### 3.1 总体结构：三 shader + 一 InnerProduct 复用

```
norm_reduce（逐 token 平方和归约）
    │  square_sum[tokens]（fp32 workspace）
    ▼
norm_apply（RMSNorm + Ada，pack1→pack1）
    │  normalized [tokens, dim]
    ▼
InnerProduct（复用原生，pack1→pack1，gemm 分支）
    │  projected [tokens, oc*4]
    ▼
unpatchify（2x2 空间反重排，pack1→pack1）
    │  output [oc, spatial]
    ▼
output_shape（record_upload 上传 int 形状）
```

### 3.2 norm_reduce shader（核心：逐 token 归约）

与 GroupNorm 的 reduce 同构，但更简单——每个 workgroup 处理一个 token，无分组：

```glsl
// 每个 workgroup 处理一个 token（gid = token），shared-memory 树形归约求平方和。
void main()
{
    uint token = gl_WorkGroupID.x;
    uint tid = gl_LocalInvocationID.x;
    uint nthreads = gl_WorkGroupSize.x;
    if (token >= p.tokens) return;

    float local_sqsum = 0.0;
    for (uint c = tid; c < p.dim; c += nthreads)   // strided 累加，覆盖 dim>线程数
    {
        float v = buffer_ld1(video_blob, token * p.dim + c);
        local_sqsum += v * v;
    }
    sqsum_shared[tid] = local_sqsum;
    barrier();
    for (uint s = nthreads / 2u; s > 0u; s >>= 1u) // 树形归约
    {
        if (tid < s) sqsum_shared[tid] += sqsum_shared[tid + s];
        barrier();
    }
    if (tid == 0u) sqsum_blob[token] = sqsum_shared[0];
}
```

统计量用 **fp32**（`float sqsum_blob[]`）而非 sfp，避免半精度累积误差。

### 3.3 norm_apply shader（核心：RMSNorm + Ada 合并成一次 FMA）

```glsl
void main()
{
    uint gi = gl_GlobalInvocationID.x;
    if (gi >= p.total) return;
    uint c = gi % p.dim;
    uint token = gi / p.dim;

    float inv_rms = 1.0 / sqrt(sqsum_blob[token] / float(p.dim) + p.eps);

    afp v  = buffer_ld1(video_blob, token * p.dim + c);
    afp n  = buffer_ld1(norm_blob, c);
    afp s  = buffer_ld1(scale_blob, c);
    afp sh = buffer_ld1(shift_blob, c);
    afp e0 = buffer_ld1(emb_blob, c * 6u);        // emb[c*6]
    afp e1 = buffer_ld1(emb_blob, c * 6u + 1u);   // emb[c*6+1]

    // 合并：result = v * (inv_rms * n * (e1 + s)) + (e0 + sh)
    afp result = v * (inv_rms * n * (e1 + s)) + (e0 + sh);
    buffer_st1(normalized_blob, token * p.dim + c, result);
}
```

公式与 CPU 逐行等价（CPU 是两行，这里合并成一次 FMA）：

```
CPU: value = data[c] * inverse_rms * norm[c]
     result = value * (emb[c*6+1] + scale[c]) + emb[c*6] + shift[c]
GPU: result = data[c] * (inverse_rms * norm[c] * (emb[c*6+1] + scale[c])) + (emb[c*6] + shift[c])
```

注意 `embedding` 的访问步长是 6（`emb[c*6+k]`），非 4 对齐，**必须标量 `buffer_ld1` 读**，无法用 vec4 load。

### 3.4 unpatchify shader（核心：2x2 空间反重排）

```glsl
void main()
{
    uint gi = gl_GlobalInvocationID.x;
    if (gi >= p.total) return;
    uint channel = gi % p.output_channels;
    uint output_index = gi / p.output_channels;

    uint x_out = output_index % p.output_width;
    uint y_out = (output_index / p.output_width) % p.output_height;
    uint t = output_index / (p.output_width * p.output_height);

    // 反推 patch 网格坐标与 2x2 子像素偏移（patch 是 GLSL 关键字，用 pidx）
    uint x = x_out / 2u, dx = x_out % 2u;
    uint y = y_out / 2u, dy = y_out % 2u;
    uint pidx = t * (p.height * p.width) + y * p.width + x;

    uint src = pidx * (p.output_channels * 4u)
        + dy * (p.output_channels * 2u) + dx * p.output_channels + channel;
    afp v = buffer_ld1(projected_blob, src);
    buffer_st1(output_blob, output_index * p.output_channels + channel, v);
}
```

这是 DiTInput patchify 的精确逆操作，`(dy, dx, channel)` 拼接顺序一一对应。

## 4. 关键问题与解决（面试最有价值）

### 4.1 最大的坑：ncnn 2D Mat 的 pack4 是「行打包」，不是「列内连续」

**这是我这次踩的最隐蔽的坑，值得重点讲。**

直觉上，`[w, h]` 的 2D 矩阵 pack4 后，每个 float4 存「连续的 4 个 w 元素」。**但这个直觉是错的**。

ncnn 的 `mat.h` 里对 `elempack` 的注释：

```
// packed count inside element
// c/4-d-h-w-4  c/4-h-w-4  h/4-w-4  w/4-4  sse/neon
```

`h/4-w-4` 意味着 2D Mat pack4 时，**h 维度（行）被打包**，每个 float4 存「4 个连续行的同一列」，而不是「同一行的 4 个连续列」。

验证：`packing_pack4to1.comp` 的映射

```glsl
const uint gi = gy * psc(n) + gx;                          // pack4 索引
const uvec4 gi4 = (gy * 4 + uvec4(0,1,2,3)) * psc(stride) + gx;  // pack1 索引
```

`gy*4 + {0,1,2,3}` 明确表示：pack4 的一个 float4 的 4 个分量，落到 pack1 的 **4 个连续行**。

**后果**：我一开始让 `norm_apply` 输出 `[dim/4, tokens]` 的 pack4 normalized，直接喂给 InnerProduct。结果 InnerProduct 走了 flatten 分支（因为 gemm 分支条件 `bottom.w == num_input` 只对 pack1 成立），把 2D pack4 按「行打包」语义展平，和我 shader 里「float4 存 4 个连续通道」的写入语义冲突，输出错位（实测 GPU[1] = CPU[4]，步长 4 错位）。

**解决**：全 pack1。`norm_apply` 输出 pack1 normalized（w=dim），让 InnerProduct 命中 gemm 分支（条件 `w == num_input`），自然输出 pack1 projected，`unpatchify` 也读 pack1。彻底绕开 2D pack4 的布局陷阱，代码反而更简单。

**教训**：ncnn 的 elempack 是「通道打包」（打包最外层维度 c/h），不是「向量化连续元素」。逐元素算子（RMSNorm/Ada/unpatchify）用 pack1 最安全；只有 GEMM 类算子才需要在 pack4 上做文章，而 InnerProduct 内部会自己处理。

### 4.2 GLSL 关键字 `patch`

`patch` 是 GLSL 的 tessellation 关键字，不能作变量名（报错 `unexpected PATCH`）。改用 `pidx`。这与 DiTInput 踩过的坑完全一致，说明「用 GLSL 保留字当变量名」是高频错误。

### 4.3 标量输入 shape 的同步读取

`shape`（`[T,H,W]` int）是极小标量数据，但 `record_upload` 的 `convert_packing` 是异步的，`mapped_ptr` 直接读会拿到旧数据。需 `record_download + submit_and_wait + reset` 同步读取（同 DiTInput 的 timestep 处理）。

### 4.4 复用 InnerProduct 必须手动传 vkdev

Net 只给顶层自定义层设 `vkdev`，`projection` 子层的 `layer_vulkan` 会在 `load_param` 时因 vkdev 为空被删除（fallback CPU）。必须在 `load_inner_product` 里 `load_param` 之前 `layer->vkdev = vkdev`（同 DiTInput 的教训）。

## 5. 性能考量

- **全 pack1**：逐元素算子（RMSNorm/Ada/unpatchify）本质是 memory-bound，pack1 不损失访存效率；矩阵乘（projection）交给 InnerProduct 的 gemm 分支，它内部会 convert 到 pack4 再算，GEMM 效率不损失。
- **两阶段归约**：norm_reduce 用树形归约，误差 O(log n)，且统计量 fp32，比 CPU 的 double 串行累加略差但远在容差内。
- **中间张量**：normalized/projected 都在 GPU workspace，层间不落回 CPU。

## 6. 测试验证

`tests/dit_output_vulkan_runner.cpp` 4 组参数：

```
[case 0] dim=32 oc=4  T=1 H=2 W=2          : max|diff| = 0.000000 PASS
[case 1] dim=64 oc=8  T=2 H=2 W=3(非对齐)   : max|diff| = 0.000000 PASS
[case 2] dim=32 oc=5(非4倍数) T=1 单patch   : max|diff| = 0.000000 PASS
[case 3] dim=64 oc=8  T=2 H=3 W=3(非方阵)   : max|diff| = 0.000000 PASS
```

覆盖：pack1/pack4 边界（dim 是 4 倍数）、多帧/单帧、`output_channels` 非 4 倍数（unpatchify 的 channel 除法边界）、非方阵 patch 网格、W 非 4 对齐。

## 7. 面试问答

**Q1：为什么全 pack1 而不是 pack4？**
ncnn 的 2D pack4 是「行打包」（打包 h 维度），逐元素算子用 pack4 会导致「float4 存 4 个连续行」的语义和「逐通道」语义冲突。而且 InnerProduct 的 gemm 分支（最省搬运的路径）要求输入 `w == num_input`，仅 pack1 满足。全 pack1 既简单又正确，矩阵乘的 pack4 由 InnerProduct 内部自行处理。

**Q2：RMSNorm 和 GroupNorm 的 reduce 有什么区别？**
GroupNorm 是「逐 (帧,组)」统计（count = cpg×W×H），RMSNorm 是「逐 token」统计（count = dim）。前者 workgroup 索引解码 `(frame, group)`，后者 workgroup 索引直接就是 token。两者都是 shared-memory 树形归约，但 RMSNorm 无需除法通道分组。

**Q3：为什么统计量用 fp32 不用 sfp？**
统计量（square_sum）若用 fp16 存，半精度会丢失累加精度，导致 inverse_rms 偏差。ncnn 内置 reduce_mean 也是用 float 存统计量，这是约定。

**Q4：embedding 为什么不能 vec4 读？**
embedding 的访问是 `emb[c*6]` / `emb[c*6+1]`，步长 6 不是 4 的倍数，无法对齐到 float4 边界，只能标量 `buffer_ld1`。
