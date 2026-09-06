# DynamicFramewiseGroupNorm 的 Vulkan 实现

> 这是六个 SeedVR2 自定义层中**第二个**完成 Vulkan 化的算子。本文档按面试口径记录：
> 实现了什么、用了什么方法、遇到什么问题、如何解决、性能如何考量，并附真实代码。

## 1. 这个算子实现了什么

`DynamicFramewiseGroupNorm` 是 VAE 编码器/解码器**残差块内**的归一化层，出现约 **50 次**。
输入输出均为 `Mat(w=W, h=H, d=T, c=C)` 的视频 latent，逻辑形状 `[C, T, H, W]`。

对每个**帧** t、每个**组** g，统计该帧内该组的 `channels_per_group × W × H` 个元素的均值方差，
再做仿射归一化：

```
count = channels_per_group × W × H          // 单帧内单组的元素数
mean  = Σx / count
var   = Σx² / count − mean²
inv   = 1 / √(max(var, 0) + eps)
out   = (x − mean) · inv · weight[c] + bias[c]
      = x · (weight[c]·inv) + (bias[c] − mean·weight[c]·inv)   // 重排成一次 FMA
```

参数：`channels`（从 param 里 bias shape 恢复）、`groups=32`、`eps=1e-6`、仿射 `weight/bias`。

## 2. 为什么要自定义：逐帧语义

这是本层存在的**根本原因**。ncnn 原生 `GroupNorm` 在 `dims==4` 时源码写死
`size = w * h * d`，统计范围**包含全部帧**（跨帧）；而 PyTorch VAE 的 GroupNorm 是
**逐帧**统计——每个帧独立归一化，不把不同时间帧混在一起。

后果：直接替换原生算子，在视频时长 T 变化时统计口径错误，输出与 reference 系统性偏离。
（此前 `test_demo/framewise_groupnorm_probe` 已实测：原生 GroupNorm 吃 4D 视频 Mat，
逐帧参考 diff=0.344、跨帧参考 diff=0，证明原生是跨帧语义。）

## 3. 用了什么方法

两条 forward 路径共享 `load_param/load_model`，保证权重与动态语义一致：

- **CPU 路径**（`forward(Mat)`）：`omp parallel for` 逐 (帧, 组) 用 `double` 累加，作为数值基线；
- **Vulkan 路径**（`forward(VkMat)`）：GroupNorm 是「先归约、再归一化」的两段式算子，
  拆成**两个 compute shader**。

### 核心：reduce shader（统计 sum 与 sq_sum）

每个 workgroup 处理一个 (帧, 组)，workgroup 内先 strided 累加、再 shared-memory 树形归约：

```glsl
layout(binding = 0, std430) readonly buffer bottom_data { sfp bottom_blob[]; };
layout(binding = 1, std430) writeonly buffer sum_data { float sum_blob[]; };   // 统计量用 fp32
layout(binding = 2, std430) writeonly buffer sqsum_data { float sqsum_blob[]; };

layout(push_constant) uniform parameter {
    uint spatial_size;
    uint channels_per_group;
    uint groups;
    uint cstep;
    uint count;
} p;

shared float sum_shared[256];
shared float sqsum_shared[256];

void main()
{
    uint gid = gl_WorkGroupID.x;        // 一个 (帧, 组)
    uint tid = gl_LocalInvocationID.x;
    uint nthreads = gl_WorkGroupSize.x;

    uint frame = gid / p.groups;        // 由 workgroup 索引解出 (帧, 组)
    uint group = gid % p.groups;
    uint begin_channel = group * p.channels_per_group;

    // 阶段一：每个线程 strided 累加自己负责的元素（覆盖 count 远大于线程数）
    float local_sum = 0.0;
    float local_sqsum = 0.0;
    for (uint i = tid; i < p.count; i += nthreads)
    {
        uint local_channel = i / p.spatial_size;
        uint spatial_idx = i % p.spatial_size;
        uint addr = (begin_channel + local_channel) * p.cstep + frame * p.spatial_size + spatial_idx;
        float v = buffer_ld1(bottom_blob, addr);
        local_sum += v;
        local_sqsum += v * v;
    }
    sum_shared[tid] = local_sum;
    sqsum_shared[tid] = local_sqsum;
    barrier();

    // 阶段二：树形归约（O(log n) 误差，优于单线程串行的 O(n)）
    for (uint s = nthreads / 2u; s > 0u; s >>= 1u)
    {
        if (tid < s)
        {
            sum_shared[tid] += sum_shared[tid + s];
            sqsum_shared[tid] += sqsum_shared[tid + s];
        }
        barrier();
    }

    if (tid == 0u)
    {
        sum_blob[gid] = sum_shared[0];
        sqsum_blob[gid] = sqsum_shared[0];
    }
}
```

### 核心：normalize shader（逐元素仿射归一化）

```glsl
layout(constant_id = 0) const uint cpg = 1u;   // channels_per_group 烤进 SPIR-V

layout(binding = 0, std430) readonly buffer bottom_data { sfp bottom_blob[]; };
layout(binding = 1, std430) writeonly buffer top_data { sfp top_blob[]; };
layout(binding = 2, std430) readonly buffer weight_data { sfp weight_blob[]; };
layout(binding = 3, std430) readonly buffer bias_data { sfp bias_blob[]; };
layout(binding = 4, std430) readonly buffer sum_data { float sum_blob[]; };
layout(binding = 5, std430) readonly buffer sqsum_data { float sqsum_blob[]; };

layout(push_constant) uniform parameter {
    uint w; uint h; uint t; uint cstep;
    uint spatial_size; uint groups; uint count;
    float eps; uint total;
} p;

void main()
{
    uint gi = gl_GlobalInvocationID.x;
    if (gi >= p.total) return;

    // 稠密索引解码：x 最内 → y → t → c 最外（ncnn 4D 布局）
    uint x = gi % p.w;
    uint y = (gi / p.w) % p.h;
    uint t_frame = (gi / (p.w * p.h)) % p.t;
    uint c = gi / (p.w * p.h * p.t);

    uint group = c / cpg;                       // 由通道反推所属组
    uint fg = t_frame * p.groups + group;
    float mean = sum_blob[fg] / float(p.count);
    float var = sqsum_blob[fg] / float(p.count) - mean * mean;
    float inv = 1.0 / sqrt(max(var, 0.0) + p.eps);

    float gamma = buffer_ld1(weight_blob, c);   // 仿射系数重排成一次 FMA
    float beta  = buffer_ld1(bias_blob, c);
    float scale = gamma * inv;
    float offset = beta - mean * scale;

    uint addr = c * p.cstep + t_frame * p.spatial_size + y * p.w + x;
    afp v = buffer_ld1(bottom_blob, addr);
    buffer_st1(top_blob, addr, v * scale + offset);
}
```

## 4. 遇到了什么问题（及其解决）

### 4.1 【最隐蔽】dispatcher.w 语义：workgroup 数只启动了 1 个

**现象**：所有 case 误差 ~500（输出爆炸）。排查发现统计量 sum/sq_sum 几乎全是 0，
导致 `var=0`、`inv=1/√(0+eps)≈1000`，`scale=weight×1000` 把输出放大到数百量级。

**根因**：ncnn 的 `record_pipeline` 里 `group_count_x = ceil(dispatcher.w / local_size_x)`。
我的 reduce shader 用 `gl_WorkGroupID.x` 作为 (帧, 组) 索引，期望 workgroup 数 = `stat_count`，
但 `dispatcher.w` 设成了 `stat_count`——`ceil(stat_count / 256) = 1`，只启动了 1 个 workgroup，
其余 63 个 (帧, 组) 的统计量从未被写入。

**解决**：`dispatcher.w = stat_count * pipeline_reduce->local_size_x()`，让
`ceil(stat_count·256 / 256) = stat_count`，恰好启动 stat_count 个 workgroup。

```cpp
// ncnn 的 group_count_x = ceil(dispatcher.w / local_size_x)，故 dispatcher.w
// 必须等于 stat_count × local_size_x，才能启动恰好 stat_count 个 workgroup。
dispatcher.w = stat_count * pipeline_reduce->local_size_x();
```

这与 ncnn 内置层的习惯不同：内置 `reduce_mean` 用 `gl_GlobalInvocationID.x`（一个线程
算一个 group，串行累加），所以 `dispatcher.w = group` 即可。我用 `gl_WorkGroupID.x`
（一个 workgroup 算一个 (帧,组)，内部树形归约），必须显式乘上 local_size。

### 4.2 统计量必须 fp32，不能用 sfp

mean/var 是后续所有元素归一化的分母来源，半精度（fp16）存储会把统计误差放大到每个
输出元素。因此 workspace 显式用 `4u`（fp32 elemsize）创建，shader 里 sum/sq_sum buffer
用 `float`（而非 `sfp`）声明——这是 ncnn 内置 GroupNorm 同一套约定（`reduce_mean.comp`
里 `mean_blob` 也是 `float`）。

### 4.3 float 累加 vs CPU double 的精度差异

CPU 基线用 `double` 累加 sum/sq_sum，GPU 用 `float`（`afp`）。小 count（<256）时两者
完全一致（diff=0）；大 count（4096）时 float 累加引入约 `5e-6` 的误差——这是**预期内**
的，因为树形归约把误差从 O(n) 降到 O(log n)，最终归一化输出误差仍在 `1e-4` 容差内。
（如需更高精度，可改「两遍归约」：先算 mean，再算 Σ(x−mean)²，避免 `E[x²]−E[x]²` 的
灾难性抵消——ncnn 的 `sub_mean_square` 正是这个思路，目前 latent 数值范围下不需要。）

## 5. 性能考量（当前实现 vs 后续优化）

**当前版本**是「正确性优先」的两 shader 实现：

- reduce：`stat_count = T×groups` 个 workgroup，每个 workgroup 256 线程做树形归约，
  读一次输入、写一对 (sum, sq_sum)；
- normalize：`T×C×H×W` 个 work item，每个读一次输入、一次统计量、一次权重，写一次输出；
- 两次 shader 之间靠 ncnn 的 pipeline barrier 保证 reduce 写完 normalize 再读。

**可优化点**（记录在案，等真 GPU 再调）：

1. **合并 reduce 的 sum/sq_sum 为一个 buffer**：当前两个 workspace，可合成
   `float2` 或交错存储，减少一次 binding 与写事务；
2. **subgroup 归约**：树形归约的前几层用 `subgroupAdd`（llvmpipe subgroup=8，NVIDIA
   真 GPU 通常 32），比 shared-memory 往返更快；
3. **融合进后续卷积**：GroupNorm 与紧随的 Conv 常可融合（scale/offset 吸收进卷积权重），
   但会破坏层的模块化，需在 5090 上实测再取舍。

## 6. 面试问答（附真实代码）

**Q1：为什么这层必须自定义，不能用 ncnn 原生 GroupNorm？**
原生 GroupNorm 在 dims==4 时 `size = w*h*d`，统计范围**跨帧**；VAE 要求**逐帧**统计。
直接替换会在动态视频时长下统计口径错误，输出系统性偏离（实测 diff=0.344）。

**Q2：逐帧语义在 shader 里怎么体现？**
reduce 阶段 `gid = gl_WorkGroupID.x` 解码成 `frame = gid / groups`，每个 (帧,组) 独立
归约、独立写一个统计量；normalize 阶段 `fg = t_frame * groups + group` 按帧读对应统计量。
没有任何跨帧的累加或共享。

**Q3：为什么 reduce 和 normalize 要拆两个 shader，不能一个？**
GroupNorm 本质上要求「先知道全组统计量，才能归一化任意元素」，存在全局数据依赖。
Vulkan 的 compute 模型里跨 workgroup 无全局 barrier，必须用两个 dispatch + 中间
workspace，靠 ncnn 的 pipeline barrier 串起来（这正是 GPU 上 reduction 的标准做法）。

**Q4：统计量为什么用 fp32 而不是 fp16？**
mean/var 是归一化的分母来源，fp16 的统计误差会放大到每个输出元素；ncnn 内置
GroupNorm 的 `mean_blob` 同样用 fp32。数据 buffer 用 sfp（受 fp16 storage 控制），
统计量 buffer 强制 float。

## 7. 验证结果

`tests/groupnorm_vulkan_runner.cpp` 对 5 组参数分别跑 CPU 与 Vulkan，断言
`max|diff| < 1e-4`。实测（WSL llvmpipe）：

```
[case 0] C=32  cpg=1  T=2 W=5  (非对齐)      : max|diff| = 0.000000 PASS
[case 1] C=64  cpg=2  T=3 W=4  (对齐)        : max|diff| = 0.000000 PASS
[case 2] C=32  cpg=1  T=1 (单帧)             : max|diff| = 0.000000 PASS
[case 3] C=96  cpg=3  T=4 W=7 (cpg非2幂)     : max|diff| = 0.000000 PASS
[case 4] C=512 cpg=16 T=2 W=16 (count=4096)  : max|diff| = 0.000005 PASS
======== 结果: 5 pass, 0 fail ========
```

用例覆盖：cpg=1/2/3/16（不同 group 划分，含非 2 次幂）、T=1 单帧边界、W=5/7 非 4
对齐（cstep 地址）、count=4096>256（strided loop + 树形归约）。全部逐元素一致。
