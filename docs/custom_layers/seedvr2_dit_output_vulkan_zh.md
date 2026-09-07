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

**为什么说这个 kernel 最重要**：它是六个自定义层里**唯一同时包含四种算子形态**的——归约（norm_reduce）、逐元素+权重调制（norm_apply）、矩阵乘（projection）、数据重排（unpatchify）。DiTBlock 里的 adaLN、attention、MLP 全都是这四种形态的组合，所以这个算子本质是 DiTBlock 的「预演」，把它吃透了，DiTBlock 就只剩工程量。

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

## 4. 关键问题与解决（正确性视角的坑）

### 4.1 最大的坑：ncnn 2D Mat 的 pack4 是「行打包」，不是「列内连续」

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

`patch` 是 GLSL 的 tessellation 关键字，不能作变量名（报错 `unexpected PATCH`）。改用 `pidx`。这与 DiTInput 踩过的坑完全一致。

### 4.3 标量输入 shape 的同步读取

`shape`（`[T,H,W]` int）是极小标量数据，但 `record_upload` 的 `convert_packing` 是异步的，`mapped_ptr` 直接读会拿到旧数据。需 `record_download + submit_and_wait + reset` 同步读取（同 DiTInput 的 timestep 处理）。

### 4.4 复用 InnerProduct 必须手动传 vkdev

Net 只给顶层自定义层设 `vkdev`，`projection` 子层的 `layer_vulkan` 会在 `load_param` 时因 vkdev 为空被删除（fallback CPU）。必须在 `load_inner_product` 里 `load_param` 之前 `layer->vkdev = vkdev`（同 DiTInput 的教训）。

## 5. 这个算子难在哪（性能视角的难点剖析）

正确性只是及格线，真正体现功力的是**性能**。这个算子的难点不在「算得对不对」，而在「它是 memory-bound 的，且当前结构有大量冗余访存」。

### 5.1 先算一笔账：这个算子到底搬了多少字节

真实尺寸 T=32、H=8、W=8、dim=2560、oc=16 → tokens=2048：

| 阶段 | 内存流量 |
|---|---|
| norm_reduce | 读 video **20 MB** + 写 sqsum 8 KB |
| norm_apply | 读 video **20 MB** + 写 normalized **20 MB** |
| projection | 读 normalized **20 MB** + 写 projected 0.5 MB |
| unpatchify | 读 projected 0.5 MB + 写 output 0.5 MB |
| **合计** | **≈ 81.5 MB** |

两个惊人的事实：

1. **video 被读了两遍（40 MB，占 49%）**——norm_reduce 读一遍算平方和，norm_apply 又读一遍做归一化。
2. **normalized 中间张量写 20 MB 再读 20 MB（40 MB，占 49%）**——norm_apply 写出来，projection 再读进去。

这 81.5 MB 里，真正「有用」的计算输出只有 0.5 MB（output），**其余 99% 都是中间搬运**。

### 5.2 为什么这是 memory-bound

算一下算术强度（arithmetic intensity）：

- norm_apply 每读一个 float（4 字节），做约 6 次浮点运算（3 乘 + 2 加 + 1 除/sqrt）。算术强度 ≈ 6 FLOP / 4 B ≈ 1.5 FLOP/B。
- 5090 的内存带宽 ~1.7 TB/s，算力 ~100+ TFLOP/s。**平衡点 ≈ 60 FLOP/B**（算力/带宽）。
- 1.5 FLOP/B 远低于 60 FLOP/B 的平衡点 → **纯 memory-bound**，瓶颈在访存带宽，不在算力。

结论：**这个算子的性能天花板由「搬了多少字节」决定，而不是「算了多少浮点」**。所以优化的唯一正确方向是**减少访存量**，而不是「用 tensor core」「用 fp16 加速计算」——那些都是 compute 优化，对这个算子无效。

### 5.3 三个真正的难点

**难点 1：归约和归一化之间需要全局同步，天然被拆成两个 kernel，导致 video 读两遍。**

RMSNorm 必须先算出 square_sum（全局归约），才能做归一化。Vulkan 的 compute shader **跨 workgroup 没有全局 barrier**，所以要么拆成两个 kernel（当前方案，video 读两遍），要么把整个 token 装进一个 workgroup 用 shared memory 归约（fuse 方案，见 §6.1）。

**难点 2：embedding 的访问步长是 6，破坏了一切向量化的可能。**

`emb[c*6]` / `emb[c*6+1]` 的步长是 6，不是 2/4/8 的幂。这意味着：
- 无法用 vec2/vec4 load（对齐不了）；
- 无法用 `buffer_ld4` 一次读 4 个通道的 embedding；
- 每个通道必须 2 次标量 load。

这是 Ada（adaLN）结构固有的——embedding 是 6 组调制参数（scale/shift × 3），channel 维度是 6 的倍数，和 GPU 的 4/8 对齐天然冲突。

**难点 3：2D pack4 的「行打包」语义，让「逐通道 + 向量化」两难。**

想向量化就要 pack4，但 2D pack4 打包的是行（§4.1），和「逐通道」语义冲突；想逐通道就用 pack1，但丢掉了向量化的访存效率。**这个矛盾的唯一出路是改用 1D 布局**（把 video 看成 `[tokens*dim]` 一维，1D pack4 就是「连续 4 通道打包」），详见 §6.2。

## 6. 如何优化这个 Vulkan 算子（面试必问，含方案与代码）

### 6.1 优化一：fuse norm_reduce + norm_apply（消除 video 的第二次读，收益最大）

**现状**：norm_reduce 读一遍 video，norm_apply 再读一遍，video 读两次。

**思路**：既然「一个 token 的 dim=2560 个 float = 10 KB」能装进 shared memory（GPU 通常 48~96 KB shared），那就可以让**一个 workgroup 处理一个 token**：第一遍把 token 的 dim 个元素读进 shared（同时累加平方和），barrier 归约出 square_sum 后，第二遍直接从 shared 读元素做归一化。这样 **video 只从 global 读一次**。

```glsl
// fused norm kernel：一个 workgroup 处理一个 token，video 只读一次
shared afp token_data[2560];   // 10 KB，缓存整个 token
shared float sqsum_shared[256];

void main()
{
    uint token = gl_WorkGroupID.x;
    uint tid = gl_LocalInvocationID.x;
    uint nthreads = gl_WorkGroupSize.x;

    // 阶段一：读 video 到 shared，同时累加平方和（video 唯一一次 global 读）
    float local_sqsum = 0.0;
    for (uint c = tid; c < p.dim; c += nthreads)
    {
        afp v = buffer_ld1(video_blob, token * p.dim + c);
        token_data[c] = v;              // 存 shared，供阶段二复用
        local_sqsum += float(v) * float(v);
    }
    sqsum_shared[tid] = local_sqsum;
    barrier();
    for (uint s = nthreads / 2u; s > 0u; s >>= 1u)
    {
        if (tid < s) sqsum_shared[tid] += sqsum_shared[tid + s];
        barrier();
    }

    float inv_rms = 1.0 / sqrt(sqsum_shared[0] / float(p.dim) + p.eps);

    // 阶段二：从 shared 读元素做归一化（不再碰 video global buffer）
    for (uint c = tid; c < p.dim; c += nthreads)
    {
        afp v = token_data[c];
        afp n = buffer_ld1(norm_blob, c);
        afp s = buffer_ld1(scale_blob, c);
        afp sh = buffer_ld1(shift_blob, c);
        afp e0 = buffer_ld1(emb_blob, c * 6u);
        afp e1 = buffer_ld1(emb_blob, c * 6u + 1u);
        buffer_st1(normalized_blob, token * p.dim + c, v * (inv_rms * n * (e1 + s)) + (e0 + sh));
    }
}
```

**收益**：
- video 读从 2 遍降到 1 遍，**省 20 MB（总流量 81.5 → 61.5 MB，约 -25%）**；
- 少一个 pipeline dispatch（少一次 kernel 启动 + barrier）；
- 不需要 square_sum workspace。

**代价与边界**：
- shared memory 10 KB/workgroup，occupancy 受 shared 限制（64 KB shared 的 GPU 最多 6 个 workgroup resident，但 256 线程 × 6 = 1536 线程，仍在可接受范围）；
- **前提是 dim 能装进 shared**。dim=2560 时 10 KB 可行；若 dim 更大（如 5120）就装不下，必须退回两阶段方案。所以这是「有条件优化」，需要运行时判断 `dim * sizeof(fp) <= shared_limit`。

### 6.2 优化二：1D pack4 向量化（减少指令数与内存事务）

**现状**：全 pack1，norm_apply 每个 work item 处理 1 个标量，做 6 次标量 load + 1 次标量 store。

**思路**：把 video/normalized/norm/scale/shift 看成 **1D 布局**（`[tokens*dim]` 连续），1D pack4 就是「连续 4 通道打包」，正好和逐通道操作对齐。每个 work item 处理一个 float4（4 个连续通道），video/norm/scale/shift 用 `buffer_ld4/st4` 一次 16 字节。

```glsl
// 1D pack4 版本：每个 work item 处理 4 个连续通道
void main()
{
    uint g = gl_GlobalInvocationID.x;   // float4 索引 [0, tokens*dim/4)
    uint c4 = g % (p.dim / 4);
    uint token = g / (p.dim / 4);
    uint c = c4 * 4;

    float inv_rms = 1.0 / sqrt(sqsum_blob[token] / float(p.dim) + p.eps);

    // video/norm/scale/shift 用 vec4，一次 load 16 字节
    afpvec4 v  = buffer_ld4(video_blob, g);
    afpvec4 n  = buffer_ld4(norm_blob, c4);
    afpvec4 s  = buffer_ld4(scale_blob, c4);
    afpvec4 sh = buffer_ld4(shift_blob, c4);

    // embedding 步长 6 无法 vec4，只能标量 gather 8 次
    afp e0 = afpvec4(buffer_ld1(emb_blob, c*6),     buffer_ld1(emb_blob, (c+1)*6),
                     buffer_ld1(emb_blob, (c+2)*6), buffer_ld1(emb_blob, (c+3)*6));
    afp e1 = afpvec4(buffer_ld1(emb_blob, c*6+1),     buffer_ld1(emb_blob, (c+1)*6+1),
                     buffer_ld1(emb_blob, (c+2)*6+1), buffer_ld1(emb_blob, (c+3)*6+1));

    afpvec4 result = v * (inv_rms * n * (e1 + s)) + (e0 + sh);
    buffer_st4(normalized_blob, g, result);
}
```

**收益**：video/norm/scale/shift 的访存从「4 次标量」变成「1 次 vec4」，指令数下降，内存事务更规整（16 字节对齐）。

**代价与权衡**：这是「有取舍」的优化，值得讲清楚——
- 输出变成 **1D pack4**，不再是 2D pack1。InnerProduct 的 gemm 分支要求 2D `w == num_input`，所以会走 flatten 分支（而不是 gemm 分支），GEMM 路径可能不如 gemm 分支高效。
- 所以这是一个「norm_apply 向量化收益 vs InnerProduct 走非 gemm 分支代价」的 trade-off，**必须在真 GPU 上 benchmark 才能下结论**。正确性优先阶段选全 pack1（方案 A）是合理的，性能调优阶段再测这个 trade-off。

### 6.3 优化三：subgroup 归约替代 shared-memory 树形归约

norm_reduce 的归约用 shared memory + 多轮 barrier。Vulkan 的 **subgroup 操作**（`VK_KHR_shader_subgroup_arithmetic`）提供硬件级的 `subgroupAdd` 归约，跨 32/64 个线程的归约不需要 shared memory 和 barrier：

```glsl
// 用 subgroupAdd 做归约，替代 shared 树形归约
float local = 0.0;
for (uint c = tid; c < p.dim; c += nthreads) { ... }

float subgroup_sum = subgroupAdd(local);   // 一次硬件归约 32 个线程
// 剩下每个 subgroup 一个代表，用 shared 归约 subgroup 之间（仅 nthreads/32 个元素）
```

**收益**：减少 barrier 次数和 shared 访问，归约更快。**前提**：真 GPU 支持 subgroup（5090 支持；llvmpipe 可能不支持，所以正确性验证阶段用 shared 树形归约更稳妥）。

### 6.4 优化四：unpatchify 与 projection 的访存融合（收益最小，优先级最低）

unpatchify 只占 1 MB（总流量 1.2%），即使把它和 projection 融合（让 projection 直接输出 unpatchify 后的布局），收益也微乎其微。**优化要抓大放小，priority 是：fuse norm > 1D pack4 > subgroup > unpatchify 融合**。

### 6.5 优化优先级总结

| 优先级 | 优化 | 收益 | 代价 |
|---|---|---|---|
| 1 | fuse norm_reduce+norm_apply | -25% 总流量 | 10KB shared/workgroup，dim 受 shared 限制 |
| 2 | 1D pack4 向量化 | 指令数↓、事务规整 | InnerProduct 走 flatten 分支，需 benchmark |
| 3 | subgroup 归约 | barrier↓ | 依赖真 GPU subgroup 扩展 |
| 4 | unpatchify 融合 | <1.5% | 侵入 InnerProduct，不值 |

## 7. 测试验证

`tests/dit_output_vulkan_runner.cpp` 4 组参数：

```
[case 0] dim=32 oc=4  T=1 H=2 W=2          : max|diff| = 0.000000 PASS
[case 1] dim=64 oc=8  T=2 H=2 W=3(非对齐)   : max|diff| = 0.000000 PASS
[case 2] dim=32 oc=5(非4倍数) T=1 单patch   : max|diff| = 0.000000 PASS
[case 3] dim=64 oc=8  T=2 H=3 W=3(非方阵)   : max|diff| = 0.000000 PASS
```

覆盖：pack1/pack4 边界（dim 是 4 倍数）、多帧/单帧、`output_channels` 非 4 倍数（unpatchify 的 channel 除法边界）、非方阵 patch 网格、W 非 4 对齐。

## 8. 面试问答

### 正确性类

**Q1：为什么全 pack1 而不是 pack4？**
ncnn 的 2D pack4 是「行打包」（打包 h 维度），逐元素算子用 pack4 会导致「float4 存 4 个连续行」的语义和「逐通道」语义冲突。而且 InnerProduct 的 gemm 分支（最省搬运的路径）要求输入 `w == num_input`，仅 pack1 满足。全 pack1 既简单又正确，矩阵乘的 pack4 由 InnerProduct 内部自行处理。

**Q2：RMSNorm 和 GroupNorm 的 reduce 有什么区别？**
GroupNorm 是「逐 (帧,组)」统计（count = cpg×W×H），RMSNorm 是「逐 token」统计（count = dim）。前者 workgroup 索引解码 `(frame, group)`，后者 workgroup 索引直接就是 token。两者都是 shared-memory 树形归约，但 RMSNorm 无需除法通道分组。

**Q3：为什么统计量用 fp32 不用 sfp？**
统计量（square_sum）若用 fp16 存，半精度会丢失累加精度，导致 inverse_rms 偏差。ncnn 内置 reduce_mean 也是用 float 存统计量，这是约定。

**Q4：embedding 为什么不能 vec4 读？**
embedding 的访问是 `emb[c*6]` / `emb[c*6+1]`，步长 6 不是 2/4/8 的幂，无法对齐到 float4 边界，只能标量 `buffer_ld1`。

### 优化类（面试官最可能追问）

**Q5：这个算子怎么优化？（必问）**
先判断它的性质：memory-bound（算术强度 ~1.5 FLOP/B，远低于 5090 的平衡点 ~60 FLOP/B）。所以优化方向是**减少访存量**，不是加速计算。按收益排序：① fuse norm_reduce+norm_apply（video 读 2 遍变 1 遍，省 25% 总流量）；② 1D pack4 向量化（vec4 访存，减少指令）；③ subgroup 归约（减 barrier）；④ unpatchify 融合（收益 <1.5%，不值得）。

**Q6：为什么 fuse 是收益最大的优化？**
因为 RMSNorm 的「先归约再归一化」结构导致 video 被读两遍（40 MB，占 49%），这是最大的冗余。fuse 后一个 workgroup 用 shared memory 缓存整个 token，video 只读一次，直接省 20 MB。代价是 10 KB shared/workgroup 限制 occupancy，且要求 dim 能装进 shared（dim=2560 可行）。

**Q7：这个算子能用 tensor core 加速吗？**
不能（对 norm/reduce/unpatchify 部分无效）。tensor core 只做矩阵乘，而这个算子的逐元素部分是 memory-bound，瓶颈在带宽不在算力。唯一沾边的是 projection（矩阵乘），但它复用 InnerProduct，ncnn 的 InnerProduct Vulkan 实现本身不用 coopmat，走的是通用 FMA。用 tensor core 的方向应该是「重写 projection 的 GEMM 用 cooperative matrix」，但那是另一个量级的工程，且对这个 memory-bound 为主的算子整体收益有限。

**Q8：如果 dim 太大装不进 shared，fuse 还成立吗？**
不成立，fuse 的前提是「一个 token 能装进 shared memory」。dim=2560 → 10 KB 可行；若 dim 到 5120（20 KB）或更大，就要退回两阶段方案，或者在 fuse 内部做「分块归约」（先部分归约写回 global，再二次归约），用两次 global 读换一次 shared 缓存——收益递减。所以正确做法是运行时判断 dim，自适应选择 fuse 或两阶段。

**Q9：为什么 1D pack4 能向量化而 2D pack4 不能？**
ncnn 的 pack4 打包的是「最外层维度」：1D `[w]` 打包 w（连续元素），所以 1D pack4 = 连续 4 通道打包，正好适合逐通道操作；2D `[w,h]` 打包 h（行），float4 存 4 个连续行的同一列，和逐通道语义错位。所以逐通道算子的向量化，必须先把张量看成 1D 布局。

**Q10：如果让你在真 GPU 上验证这些优化，你会怎么做？**
用 ncnn 的 benchmark（`benchncnn`）或自己写计时，跑真实尺寸（T=32、H=8、W=8、dim=2560），分别测四个变体（当前 pack1 两阶段 / fuse / 1D pack4 / fuse+pack4），比较延迟和显存带宽利用率。重点看「有效带宽 = 总流量/延迟」是否接近理论峰值（5090 ~1.7 TB/s），以及 occupancy 是否被 shared/寄存器限制。用 Nsight Graphics 抓 counter 确认瓶颈确实是 memory（而不是 latency-bound 或 occupancy-bound）。
