# DynamicFramewiseSpatialAttention 的 Vulkan 实现

> 这是六个 SeedVR2 自定义层中**第三个**完成 Vulkan 化的算子，也是第一个包含
> **softmax 归约**的注意力形态算子。本文档按面试口径记录：实现了什么、用了什么
> 方法、遇到什么问题、如何解决、性能如何考量，并附真实代码。

## 1. 这个算子实现了什么

`DynamicFramewiseSpatialAttention` 是 VAE **bottleneck（mid_block）** 的单头空间自注意力
层，输入输出均为 `Mat(w=W, h=H, d=T, c=C)`，`channels=512`。每个帧独立执行以下链路：

```
GroupNorm(逐帧, 32组, eps=1e-6)          → normalized [tokens, C]
Q/K/V 投影（三个 [C,C] 全连接）           → Q, K, V  [tokens, C]
softmax(QKᵀ / √C) @ V                     → attended [tokens, C]   （单头，head_dim=C）
输出投影（[C,C] 全连接）+ 残差连接        → output = proj(attended) + input
```

其中 `tokens = W×H`（空间位置数），`attention_scale = 1/√C`。这是**单头**注意力
（head_dim 就是全通道 C），且**逐帧**——不同时间帧的 attention 完全不串扰。

## 2. 为什么要自定义

三点都超出 ncnn 原生算子的表达能力：

1. **逐帧 + 动态 tokens**：`tokens = W×H` 随运行时输入变化，且 attention 只在单帧内
   token 之间做，不能跨帧；
2. **内嵌 GroupNorm**：attention 前的归一化是逐帧 32 组 GroupNorm（和第二个算子同语义），
   原生的 MultiHeadAttention 没有这个前置；
3. **残差 + 投影的完整 attention block**：这是 PyTorch 里一个完整的 `AttentionBlock`，
   而非孤立的多头注意力，整块封装才能保证权重加载顺序与 reference 一致。

## 3. 用了什么方法

注意力是「先归约 → 再归一化 → 再矩阵乘 → 再 softmax → 再输出投影」的多段式算子，
拆成 **5 个 compute shader**（比前两个算子的 1~2 个多，因为链路更长）：

| shader | 作用 | 并行策略 |
|---|---|---|
| `norm_reduce` | 统计每个 (帧,组) 的 sum/sq_sum | 每 workgroup 一个 (帧,组)，树形归约 |
| `norm_apply` | 逐元素 GroupNorm 归一化 | 每 work item 一个 (帧,token,c) |
| `qkv_projection` | Q/K/V 三个线性投影 | 每 work item 一个 (帧,token,out_c)，**三次 dispatch 复用同一 pipeline** |
| `attention` | softmax(QKᵀ/√C)@V | 每 work item 一个 (帧,query_token) |
| `output_projection` | 输出投影 + 残差 | 每 work item 一个 (帧,token,out_c) |

**关键设计决策**：中间激活（normalized/Q/K/V/attended）统一用 **fp32 workspace** 暂存，
权重/输入/输出用 `sfp` + `buffer_ld1/st1`（精度自适应）——与前两个算子约定一致。

### 核心：attention shader（softmax 的三遍 QKᵀ）

这是本算子的核心。每个 work item 处理一个 query_token，为避免物化 `[tokens, tokens]`
的 scores 矩阵（O(tokens²) 显存），采用**三遍 QKᵀ**：

```glsl
void main()
{
    uint gi = gl_GlobalInvocationID.x;
    if (gi >= p.total) return;                       // total = frames * tokens
    uint frame = gi / p.tokens;
    uint q_token = gi % p.tokens;
    uint q_base = frame * p.tokens * p.channels + q_token * p.channels;

    // 第一遍：找 max（数值稳定，防止 exp 溢出）
    float max_score = -3.402823e+38f;
    for (uint k = 0u; k < p.tokens; k++) {
        float dot = 0.0;
        uint k_base = frame * p.tokens * p.channels + k * p.channels;
        for (uint c = 0u; c < p.channels; c++)
            dot += q_blob[q_base + c] * k_blob[k_base + c];
        dot *= p.scale;
        if (dot > max_score) max_score = dot;
    }

    // 第二遍：算 softmax 分母 Σ exp(score - max)
    float sum = 0.0;
    for (uint k = 0u; k < p.tokens; k++) {
        float dot = 0.0;
        uint k_base = frame * p.tokens * p.channels + k * p.channels;
        for (uint c = 0u; c < p.channels; c++)
            dot += q_blob[q_base + c] * k_blob[k_base + c];
        sum += exp(dot * p.scale - max_score);
    }

    // 第三遍：PV 加权累加 attended[q] = Σ_k prob_k · V[k]
    uint out_base = frame * p.tokens * p.channels + q_token * p.channels;
    for (uint c = 0u; c < p.channels; c++) attended_blob[out_base + c] = 0.0;
    for (uint k = 0u; k < p.tokens; k++) {
        float dot = 0.0;
        uint k_base = frame * p.tokens * p.channels + k * p.channels;
        for (uint c = 0u; c < p.channels; c++)
            dot += q_blob[q_base + c] * k_blob[k_base + c];
        float prob = exp(dot * p.scale - max_score) / sum;
        uint v_base = frame * p.tokens * p.channels + k * p.channels;
        for (uint c = 0u; c < p.channels; c++)
            attended_blob[out_base + c] += prob * v_blob[v_base + c];
    }
}
```

### 核心：qkv_projection 的三次 dispatch 复用

Q/K/V 是三个结构相同的线性投影，用**同一个 shader + 同一个 pipeline**，靠三次
dispatch 换不同的权重/偏置/输出 buffer：

```cpp
// Q 投影
bindings = {normalized, q_weight_gpu, q_bias_gpu, q_workspace};
cmd.record_pipeline(pipeline_qkv_projection, bindings, constants, dispatcher);
// K 投影
bindings = {normalized, k_weight_gpu, k_bias_gpu, k_workspace};
cmd.record_pipeline(pipeline_qkv_projection, bindings, constants, dispatcher);
// V 投影
bindings = {normalized, v_weight_gpu, v_bias_gpu, v_workspace};
cmd.record_pipeline(pipeline_qkv_projection, bindings, constants, dispatcher);
```

这比写三个几乎一样的 shader 更省代码，且 pipeline 编译只发生一次。

## 4. 遇到了什么问题（及其解决）

### 4.1 softmax 数值稳定（max 减法）

QKᵀ 的 score 若直接 `exp(score)`，当 score 较大时（如 >88，因为 `exp(88)≈1.7e38` 已
接近 float 上界）会溢出成 `inf`。**解决**：先找该行的 max，`exp(score - max)` 保证
指数项 ≤ 1，同时 `max` 在分子分母抵消，数学结果不变。这与 CPU 基线、PyTorch 的
`F.softmax` 内部做法完全一致。

### 4.2 多段式算子靠 ncnn 的 pipeline barrier 串起来

5 个 shader 之间有严格的数据依赖（reduce 写完 norm_apply 才能读，等等）。Vulkan 的
compute 模型里跨 workgroup 无全局 barrier，但**跨 dispatch 的依赖由 ncnn 的
`record_pipeline` 自动插入 pipeline barrier 保证**——这是 ncnn 封装层的关键价值：我只需
按顺序 `record_pipeline` 5 次，barrier 全自动。不需要手动 `vkCmdPipelineBarrier`。

### 4.3 vkdev 是 const 指针

自定义层的 `Layer::vkdev` 成员是 `const VulkanDevice*`，写辅助函数 `compile_pipeline`
时参数类型必须也是 const，否则编译报 `invalid conversion`。这是与前两个算子（直接把
`new Pipeline(vkdev)` 内联在 create_pipeline 里）不同的地方——第三个算子因为 pipeline
多，抽了辅助函数，才第一次显式暴露这个 const 约束。

### 4.4 workspace 布局：token-major vs channel-major

输入/输出是 ncnn 4D **channel-major**（c 最外层、cstep 对齐），但 attention 的操作
（QKᵀ、softmax、PV）都在 **token 维度**上。因此中间激活统一用 **token-major** 的 1D
workspace（`[frame][token][c]` 连续），在 `norm_apply`（channel-major 读 → token-major 写）
和 `output_projection`（token-major 读 → channel-major 写）两处做布局转换。这是最容易
写错索引的地方，测试用非 4 对齐的 `tokens=15` 专门踩这个边界。

## 5. 性能考量（当前实现 vs 后续优化）

**当前版本**是「正确性优先」的朴素实现，attention shader 的时间复杂度是
`O(frames × tokens² × C)`，且 QKᵀ 算了 3 遍（找 max、算 sum、PV 各一遍）。

**可优化点**（记录在案，等真 GPU 再调）：

1. **两遍替代三遍**：第二遍算 sum 时同时缓存 `exp(score-max)`（用 shared memory 或
   subgroup），第三遍省一次 QKᵀ——但 tokens 运行时可变，shared 大小固定，需分块；
2. **flash-attention 式 online softmax**：单遍流式计算 running max + running sum，
   把 QKᵀ 从 3 遍降到 1 遍，同时用 tiling 把 O(tokens²) 的中间量控制在 shared memory
   内，避免外存往返；
3. **shared memory 分块矩阵乘**：QKᵀ 和 PV 都是 GEMM，可以像标准 attention kernel 那样
   tile 进 shared memory，提升数据复用。

**为什么当前朴素实现可接受**：bottleneck 的 `tokens` 是 VAE 最小分辨率处（mid_block），
规模很小；且本层在 VAE 中只出现 1~2 处，不是性能热点。真正的性能瓶颈在 DiTBlock 的
32 层注意力，届时再上 tiling/online-softmax。

## 6. 面试问答（附真实代码）

**Q1：为什么拆成 5 个 shader，而不是一个？**
注意力天然是多段式：先归约（GroupNorm 统计）、再逐元素（归一化）、再矩阵乘（QKV）、
再 softmax（含归约）、再矩阵乘 + 残差。Vulkan 里跨 workgroup 无全局 barrier，每段之间
有数据依赖，只能拆成多个 dispatch，靠 ncnn 的 pipeline barrier 串起来。这也是 GPU 上
实现 attention 的标准做法。

**Q2：softmax 为什么用三遍 QKᵀ？**
因为 softmax 需要先知道整行的 max 才能稳定地算 exp，再需要整行的 sum 才能归一化。
不物化 `[tokens,tokens]` 的 scores 矩阵（省显存）就要么重算 QKᵀ、要么用 online-softmax。
当前正确性优先版本用三遍重算，性能优化方向是 flash-attention 式单遍流式计算。

**Q3：Q/K/V 投影为什么三次 dispatch 复用同一个 shader？**
三个投影只是权重不同、结构完全相同。一个 shader + 一个 pipeline 编译一次，三次
dispatch 换 binding，比写三个重复 shader 更省代码和编译时间。

**Q4：中间激活为什么用 fp32 workspace 而不是 fp16？**
和前两个算子一致：正确性优先阶段用 fp32 保证精度对齐 CPU 基线；且 softmax 的累加、
attention 的点积对精度敏感。数据 buffer 用 sfp（受 fp16 storage 控制），统计量和中间
激活强制 float。

## 7. 验证结果

`tests/attention_vulkan_runner.cpp` 对 4 组参数分别跑 CPU 与 Vulkan，断言
`max|diff| < 1e-4`。实测（WSL llvmpipe）：

```
[case 0] C=32 T=2 tokens=16               : max|diff| = 0.000000 PASS
[case 1] C=64 T=3 tokens=25               : max|diff| = 0.000000 PASS
[case 2] C=32 T=1 tokens=64 (单帧)        : max|diff| = 0.000000 PASS
[case 3] C=64 T=2 tokens=15 (非方阵/非4对齐) : max|diff| = 0.000000 PASS
======== 结果: 4 pass, 0 fail ========
```

用例覆盖：单帧/多帧（逐帧独立性）、非方阵 tokens（15）、非 4 对齐（cstep 与布局转换
边界）、channels 32/64（不同 head_dim）。全部逐元素一致。
