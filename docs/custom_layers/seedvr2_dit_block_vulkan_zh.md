# SeedVR2DiTBlock 的 Vulkan 实现（面试文档）

## 1. 功能

`SeedVR2DiTBlock` 是 SeedVR2 3B DiT 的 **Transformer block**，是 32 层堆叠的核心计算单元，也是六个自定义层里**最难、最重要**的一个。它把视频/文本 token 送入一个完整的标准 transformer 子层：

```
输入 vid[L_v,2560] / txt[L_t,2560] / embedding[15360] / shape[T,H,W]
        │
        ├─① 视频 attention 支路：
        │     RMSNorm → adaLN调制 → QKV投影 → 窗口attention(MM-RoPE) → 输出投影 → 门控残差
        ├─② 文本 attention 支路：同①（last_layer 时 vid_only 跳过调制）
        ├─③ 视频 MLP 支路：RMSNorm → adaLN调制 → SwiGLU MLP → 门控残差
        ├─④ 文本 MLP 支路：同③
        │
        ▼
输出 vid_out[L_v,2560] + txt_out[L_t,2560]
```

**为什么它最重要**：它把前面五个算子的所有算子形态一次性组合——归约（RMSNorm/softmax）、逐元素+权重调制（adaLN/SwiGLU/残差）、矩阵乘（5 个投影）、数据重排（窗口切分）+ **两个新难点**（MM-RoPE 旋转、动态窗口 attention）。吃透它 = 吃透整个 DiT 的推理加速。

## 2. 输入输出定义

| 项 | 定义 |
|---|---|
| 输入 0 `vid` | `[L_v, dim]` 2D，L_v = T·H·W，dim=2560 |
| 输入 1 `txt` | `[L_t, dim]` 2D，L_t 文本 token 数（~154）|
| 输入 2 `embedding` | `[dim*6]` 1D，即 `[dim, 6]` 扁平，本层 adaLN 调制参数 |
| 输入 3 `vid_shape` | `[3]` 1D int，存 `[T, H, W]`（patch 网格）|
| 输出 0 `vid_out` | `[L_v, dim]` |
| 输出 1 `txt_out` | `[L_t, dim]` |

参数：`block_index`、`shared_weights`（视频/文本是否共享权重）、`last_layer`（末层 vid_only 特殊语义）、`shifted_window`、`dim=2560`、`heads=20`、`head_dim=128`、`mlp_hidden=6912`、`norm_eps`。

权重（每个分支）：6 个调制向量 + 5 个投影（qkv/proj_out/mlp_gate_proj/mlp_in_proj/mlp_out_proj）+ norm_q/norm_k + 全局 rope_freqs[21]。

## 3. 实现方法

### 3.1 总体结构：8 个 shader + 5 个 InnerProduct 复用

```
rmsnorm_reduce（逐 token 平方和归约）      ┐
rmsnorm_apply（RMSNorm + adaLN 调制）      ├─ 复用 DiTOutput 的归约思路
qkv_prepare（MM-RoPE 旋转 + bf16 舍入）    │
attention（softmax 三遍 QK^T @ V）         ├─ 核心难点
ada_residual（门控/普通/两倍残差）         │
silu（SwiGLU 门控）                        │
zero_buffer（窗口稀疏 workspace 清零）      │
window_sum（跨窗口归约）                   ┘
+ qkv/proj_out/mlp_gate_proj/mlp_in_proj/mlp_out_proj（复用 InnerProduct）
```

### 3.2 rmsnorm_apply：RMSNorm + adaLN 融合（核心代码块）

与 DiTOutput 的 norm_apply 同构，但多了 `slot` 参数（attention 用 slot=0，mlp 用 slot=3）和 `has_ada` 开关（last_layer 文本只做 RMSNorm）：

```glsl
void main()
{
    uint gi = gl_GlobalInvocationID.x;
    if (gi >= p.total) return;
    uint c = gi % p.dim;
    uint token = gi / p.dim;

    float inv_rms = 1.0 / sqrt(sqsum_blob[token] / float(p.dim) + p.eps);
    afp v = buffer_ld1(bottom_blob, token * p.dim + c);

    afp result;
    if (p.has_ada == 0u)
        result = v * inv_rms;                       // 纯 RMSNorm
    else {
        afp s  = buffer_ld1(scale_blob, c);
        afp sh = buffer_ld1(shift_blob, c);
        afp e0 = buffer_ld1(emb_blob, c * 6u + p.slot);
        afp e1 = buffer_ld1(emb_blob, c * 6u + p.slot + 1u);
        result = v * (inv_rms * (e1 + s)) + (e0 + sh);  // 合并成一次 FMA
    }
    buffer_st1(top_blob, token * p.dim + c, result);
}
```

### 3.3 qkv_prepare：MM-RoPE + bf16（核心代码块）

每个 work item 处理一个 (head, token)，串行遍历 head_dim 维，做三件事：**RMSNorm → 乘 gamma → MM-RoPE 旋转 → bf16 舍入**：

```glsl
void main()
{
    uint head = gi / p.seq;
    uint token = gi % p.seq;

    // 解码源 token 索引与 RoPE 位置（视频=窗口内相对坐标，文本=(text,text,text)）
    bool is_text = token >= p.video_length;
    uint src_token, pos_t, pos_h, pos_w;
    if (is_text) {
        src_token = token - p.video_length;
        pos_t = pos_h = pos_w = src_token;
    } else {
        uint local_wh = p.local_h * p.local_w;
        uint t_local = token / local_wh;
        uint yx = token % local_wh;
        uint y_local = yx / p.local_w, x_local = yx % p.local_w;
        src_token = ((p.t0 + t_local) * p.height + (p.h0 + y_local)) * p.width + (p.w0 + x_local);
        pos_t = p.text_length + t_local; pos_h = y_local; pos_w = x_local;
    }

    // 阶段一：head_dim 维平方和（RMSNorm reduce）
    float q_sqsum = 0.0, k_sqsum = 0.0;
    for (uint d = 0; d < p.head_dim; d++) { ... q_sqsum += qv*qv; k_sqsum += kv*kv; }
    float q_inv = 1.0 / sqrt(q_sqsum / p.head_dim + p.eps);
    float k_inv = 1.0 / sqrt(k_sqsum / p.head_dim + p.eps);

    // 阶段二：归一化 + gamma，暂存局部数组（RoPE 需全部 head_dim 维）
    float q_local[128], k_local[128];
    for (uint d = 0; d < p.head_dim; d++) {
        q_local[d] = qv * q_inv * norm_q[d];
        k_local[d] = kv * k_inv * norm_k[d];
    }

    // 阶段三：MM-RoPE 旋转前 3*42=126 维（3 轴 × 21 对 × 2）
    for (uint axis = 0; axis < 3; axis++) {
        float pos = (axis==0) ? pos_t : ((axis==1) ? pos_h : pos_w);
        for (uint pair = 0; pair < 21; pair++) {
            uint d = axis * 42 + pair * 2;
            if (d + 1 >= p.head_dim) break;
            float c = cos(pos * freqs[pair]), s = sin(pos * freqs[pair]);
            float f0 = q_local[d], f1 = q_local[d+1];
            q_local[d]   = f0 * c - f1 * s;
            q_local[d+1] = f1 * c + f0 * s;
            f0 = k_local[d]; f1 = k_local[d+1];
            k_local[d]   = f0 * c - f1 * s;
            k_local[d+1] = f1 * c + f0 * s;
        }
    }

    // 阶段四：bf16 舍入后写出
    for (uint d = 0; d < p.head_dim; d++) {
        buffer_st1(q_blob, out + d, bf16(q_local[d]));
        buffer_st1(k_blob, out + d, bf16(k_local[d]));
        buffer_st1(v_blob, out + d, bf16(vv));
    }
}

// PyTorch .bfloat16() 的 round-to-nearest ties-to-even，位运算复现
float bf16(float v) {
    uint bits = floatBitsToUint(v);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    bits &= 0xffff0000u;
    return uintBitsToFloat(bits);
}
```

### 3.4 attention：softmax 三遍 QKᵀ（核心代码块）

每个 work item 处理一个 (head, query)，**三遍遍历 key 但不物化 scores 矩阵**：

```glsl
void main()
{
    uint head = gi / p.seq;
    uint query = gi % p.seq;
    uint q_base = (head * p.seq + query) * p.head_dim;

    // 第一遍：找 max（softmax 数值稳定）
    float maximum = -1e30;
    for (uint key = 0; key < p.seq; key++) {
        float dot = 0.0;
        for (uint d = 0; d < p.head_dim; d++)
            dot += q[q_base+d] * k[k_base+d];
        maximum = max(maximum, dot * p.scale);
    }

    // 第二遍：算 sum(exp)
    float denom = 0.0;
    for (uint key = 0; key < p.seq; key++) { ... denom += exp(dot*scale - maximum); }

    // 第三遍：加权累加 v
    float acc[128] = {0};
    for (uint key = 0; key < p.seq; key++) {
        float w = exp(dot*scale - maximum) / denom;
        for (uint d = 0; d < p.head_dim; d++)
            acc[d] += w * v[k_base+d];
    }

    // 视频 query → vid_out_ws[窗口段]，文本 query → txt_out_ws[窗口段]
    ...
}
```

## 4. 这个算子难在哪（核心难点剖析）

### 4.1 难点 1：MM-RoPE 的「位置」是窗口相关的，不能预计算

标准 RoPE 的位置编码只依赖 token 的绝对位置。但这里的 **MM-RoPE（Multi-Modal RoPE）位置是「窗口内相对坐标」**：

```
视频 token 的 position = (text_length + (t - t0), y - h0, x - w0)
```

`t0/h0/w0` 是窗口起点。这意味着**同一个 token 在不同窗口里（shifted window 时）有不同 RoPE 位置**，所以 q/k 无法预计算成全局 workspace，必须在每个窗口内重新计算。这是「动态窗口」和「旋转位置编码」叠加出来的硬约束。

### 4.2 难点 2：head_dim 的 3 轴旋转布局

head_dim=128 被切成 `3 轴 × 21 对 × 2 = 126` 维做旋转（t/h/w 三个轴的频率一样，但位置不同），剩 2 维不旋转。这个「42 步长 × 21 对」的访存模式不是连续的，无法向量化，只能标量旋转。

### 4.3 难点 3：bf16 舍入的精确复现

PyTorch 在 attention 前显式 `.bfloat16()`，用的是 **round-to-nearest ties-to-even**。GPU 上不能简单截断（truncate），必须用位运算精确复现舍入模式，否则数值和 CPU 基线对不齐（实测不处理会有 1e-2 量级偏差）。

### 4.4 难点 4：文本跨窗口平均 + 视频跨窗口累加（两种不同的归约语义）

- **文本**：text token 在每个窗口都参与 attention，结果要**平均**（÷ 窗口数）；
- **视频**：非 shifted 时每个 token 只在一个窗口；shifted 时一个 token 跨多个窗口，结果要**累加**（不平均）。

两种不同的跨窗口归约语义，需要分开处理（window_sum 的 has_divide flag），且 shifted 的视频累加不能简单用「直接写」（会覆盖而非累加），必须用独立窗口 buffer + 最后 sum。

每个窗口只写 `vid_out_ws` 中属于自己的 token，其他位置也会被 `window_sum`
读取，因此不能假设新分配的 GPU 内存天然为零。DiT 改为跨 block 共享 allocator
后，旧 workspace 会被稳定复用，未初始化值会直接污染 attention。当前实现先用
`zero_buffer` shader 清空 `vid_out_ws`，再执行逐窗口 attention 和归约；这既是
正确性要求，也是零拷贝调度能够成立的前提。

### 4.5 难点 5：动态窗口的变长序列

窗口数量、每个窗口的 video token 数都是运行时 T/H/W 决定的（`make_windows` 用 sqrt(3600/HW) 缩放 + 3 分片 + 时间 4 分片）。GPU 上「变长序列」无法用固定 workgroup 尺寸，只能逐窗口 dispatch + 复用 workspace。

### 4.6 难点 6：内存膨胀的权衡

「逐窗口 dispatch + 独立窗口 buffer」的正确性方案，在真实尺寸下 vid_out_ws/txt_out_ws 会膨胀到几十 MB（窗口数 ×）。这是「正确性优先」的代价，性能优化（flash attention 分块）才能解决。

## 5. 如何优化这个 Vulkan 算子（面试必问）

### 5.1 优化一：flash attention 式分块（收益最大）

**现状**：attention 三遍 QKᵀ，每遍都从 global 重读所有 key 的 k/v，Q/K/V workspace 也是 seq×head_dim 的完整物化。

**思路**：把 Q 按块切分，每块用 **online softmax**（增量维护 running max 和 running sum），一趟遍历 K/V 就出结果，把显存从 O(seq²)（scores 矩阵）或 O(seq)（q/k/v 物化）降到 O(block)。同时 K/V 用 shared memory 分块缓存，减少 global 重读。

**收益**：attention 从「三遍读 k/v」降到「一遍读 k/v」，且 q/k/v 不用完整物化，显存和带宽都大幅下降。这是 attention 优化的标准答案。

### 5.2 优化二：RoPE 的 cos/sin 预计算

**现状**：每个 (head, token) 在 qkv_prepare 里重复算 63 对 cos/sin（3 轴 × 21 对）。

**思路**：RoPE 的 angle = position × freq，其中 position 是有限的窗口内相对坐标（0..local 范围），freq 是固定 21 个。可以**预计算 position×freq 的 cos/sin 表**，避免每个 token 重复算（cos/sin 是 SFU 指令，较慢）。

### 5.3 优化三：1D pack4 向量化

head_dim=128 是 4 的倍数，q/k/v 的逐元素操作（RMSNorm、gamma、bf16）可以用 1D pack4 向量化（连续 4 维打包）。但 MM-RoPE 的「42 步长 × 2」访存无法向量化，所以 RoPE 部分仍是标量。收益中等，且要小心 2D pack4 的「行打包」陷阱（见 DiTOutput 文档）。

### 5.4 优化四：subgroup 归约

rmsnorm_reduce 和 attention 的 max/sum 归约用 shared-memory 树形归约，可以换 subgroup 操作（`subgroupAdd`/`subgroupMax`），减少 barrier 和 shared 访问。

### 5.5 优化优先级总结

| 优先级 | 优化 | 收益 | 代价 |
|---|---|---|---|
| 1 | flash attention 分块 | attention 访存降 ~3× | 复杂，需 online softmax + 分块 |
| 2 | RoPE cos/sin 预计算 | 减少 SFU 计算 | 需要小查找表 |
| 3 | 1D pack4 | 指令数↓ | RoPE 部分无法向量化 |
| 4 | subgroup 归约 | barrier↓ | 依赖 subgroup 扩展 |

## 6. 测试验证

`tests/dit_block_vulkan_runner.cpp` 4 组参数（head_dim=128 与真实一致）：

```
[case 0] dim=128 heads=1 T=2 基本           : max|diff| = 0.000005 PASS
[case 1] shifted(视频跨窗口累加)            : max|diff| = 0.000005 PASS
[case 2] last_layer+shared 单patch          : max|diff| = 0.000018 PASS
[case 3] dim=256 heads=2 T=3                : max|diff| = 0.000007 PASS
```

覆盖：MM-RoPE 多轴旋转、bf16 舍入、动态窗口（含 shifted 的视频跨窗口累加）、文本跨窗口平均、last_layer 的 vid_only 语义、shared_weights、多 head、多帧。误差 1e-5 量级（bf16 舍入 + softmax 累积的预期精度，远小于 1e-3 阈值）。

## 7. 面试问答

### 正确性类

**Q1：为什么 q/k/v 不能在全局预计算？**
因为 MM-RoPE 的位置是「窗口内相对坐标」（`t-t0`、`y-h0`、`x-w0`），同一个 token 在 shifted window 下属于多个窗口、有不同的 RoPE 位置。所以 q/k 必须在每个窗口内重新计算，不能预计算成全局 workspace。

**Q2：MM-RoPE 和标准 RoPE 有什么区别？**
标准 RoPE 只有一维位置（序列位置），MM-RoPE 有三维（t/h/w），把 head_dim=128 切成 `3 轴 × 21 对 × 2 = 126` 维，三个轴用相同的频率表但不同的位置。视频 token 用空间坐标，文本 token 用 `(text, text, text)` 三轴同位置。

**Q3：为什么 bf16 舍入要位运算而不是截断？**
PyTorch 的 `.bfloat16()` 是 round-to-nearest ties-to-even，不是截断。截断会有系统性偏差（实测 1e-2），和 CPU 基线对不齐。位运算 `bits += 0x7fff + ((bits>>16)&1); bits &= 0xffff0000` 精确复现这个舍入模式。

**Q4：文本和视频的跨窗口归约为什么不一样？**
文本 token 在每个窗口都作为 query 参与 attention，所以结果是「各窗口的平均」；视频 token 在 shifted 窗口下跨窗口重叠，结果是「各窗口的累加」（对应 CPU 的 `+=` 语义）。这是两个不同的归约，用 window_sum 的 has_divide flag 区分。

### 优化类（面试官最可能追问）

**Q5：这个算子怎么优化？（必问）**
它是 transformer block，瓶颈在 attention。核心优化是 flash attention 式分块：把三遍 QKᵀ（每遍都重读 k/v）改成分块 + online softmax 的一趟遍历，显存从 O(seq²) 降到 O(block)，带宽降 ~3×。其次是 RoPE cos/sin 预计算（避免重复 SFU）、1D pack4 向量化、subgroup 归约。

**Q6：为什么三遍 QKᵀ 而不是物化 scores 矩阵？**
scores 矩阵是 O(seq²) 显存（seq=666 时 666²×4B=1.7MB/head/窗口，20 head × 多窗口会爆显存）。三遍 QKᵀ 用「重算」换「不物化」，省 O(seq²) 显存，代价是多算两遍 dot。这是「时间换空间」的经典 trade-off，flash attention 进一步把「重算」也优化掉。

**Q7：这个算子能用 tensor core 吗？**
能，但只在矩阵乘部分（5 个投影）。attention 的 softmax(QKᵀ)@V 是「变长序列的注意力」，ncnn 的 InnerProduct 走通用 FMA。要用 tensor core 需要把 attention 的 QKᵀ 和 PV 写成 cooperative matrix 形态 + fp16，这是另一个量级的工程。当前逐元素部分（RMSNorm/RoPE/残差）是 memory-bound，tensor core 帮不上。

**Q8：动态窗口为什么不用固定窗口？**
窗口大小由运行时 T/H/W 决定（make_windows 用 sqrt(3600/HW) 缩放，保证窗口面积接近 3600 像素，再 3 分片 + 时间 4 分片），这是模型对任意分辨率视频的适配。GPU 上变长序列只能逐窗口 dispatch + 复用 workspace。

**Q9：如果真 GPU 上验证，你关注什么指标？**
有效带宽（总流量/延迟）是否接近 5090 的 1.7TB/s 峰值；attention 部分是否被 global 访存绑死（flash attention 前后对比）；qkv_prepare 的 RoPE 是否成为 SFU 瓶颈；occupancy 是否被 q/k/v workspace 的寄存器压力限制。用 Nsight Graphics 抓 counter 定位。

**Q10：这个算子和前面 5 个算子的关系？**
它是「集大成者」——RMSNorm 归约复用 DiTOutput/GroupNorm 的经验，softmax 三遍 QKᵀ 复用 SpatialAttention 的经验，复用 InnerProduct 的 vkdev/elempack 复用 DiTInput 的经验，全 pack1 复用 DiTOutput 的教训。新增的 MM-RoPE 和动态窗口是它独有的难点。吃透它 = 掌握 GPU 上 transformer 推理加速的全部核心问题。
