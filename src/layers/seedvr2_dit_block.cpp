// 本文件实现 SeedVR2 3B 动态多模态 Transformer block 的 FP32 CPU 路径。
// 大矩阵乘法由 NCNN 原生 InnerProduct 完成；这里实现 batch=1 下按运行时
// T/H/W 生成普通或 shifted 窗口、视频/文本联合 MM-RoPE attention、AdaSingle
// 调制、残差和 SwiGLU 调度。该实现首先用于逐层数值基线，Vulkan 路径会复用
// 完全相同的窗口索引和权重布局，避免 CPU/GPU 两套语义发生偏差。
//
// Vulkan 路径：5 个矩阵乘投影复用 InnerProduct（Layer_final 委托），自定义
// 代码负责 8 个 compute shader——rmsnorm_reduce/rmsnorm_apply（逐 token RMSNorm
// + adaLN 调制，reduce+apply 两段）、qkv_prepare（MM-RoPE 旋转 + bf16 舍入）、
// attention（softmax 三遍 QK^T，不物化 scores）、ada_residual（门控残差）、
// silu（SwiGLU 门控）、zero_buffer（窗口稀疏区清零）和 window_sum（跨窗口归约）。
#include "layers/seedvr2_dit_block.h"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

#if NCNN_VULKAN
#include <gpu.h>
#include <vector>
#endif

namespace
{
    ncnn::Layer *load_inner_product(
        const ncnn::ModelBin &mb,
        int input_size,
        int output_size,
        bool bias,
        const ncnn::VulkanDevice *vkdev)
    {
        ncnn::Layer *layer = ncnn::create_layer("InnerProduct");
        if (!layer)
            return nullptr;
#if NCNN_VULKAN
        // 关键：必须在 load_param 之前设置 vkdev。Layer_final::load_param 里若
        // vkdev 为空会把 layer_vulkan 删除（fallback 到 CPU），之后无法走 Vulkan。
        if (vkdev)
            layer->vkdev = vkdev;
#endif
        ncnn::ParamDict pd;
        pd.set(0, output_size);
        pd.set(1, bias ? 1 : 0);
        pd.set(2, input_size * output_size);
        if (layer->load_param(pd) != 0 || layer->load_model(mb) != 0)
        {
            delete layer;
            return nullptr;
        }
        return layer;
    }

    inline float silu(float value)
    {
        return value / (1.f + std::exp(-value));
    }

    inline int ceil_div(int value, int divisor)
    {
        return (value + divisor - 1) / divisor;
    }

    inline float round_to_bfloat16(float value)
    {
        // PyTorch 在调用 SDPA 前显式执行 .bfloat16()。这里采用 round-to-nearest
        // ties-to-even 后再以 FP32 保存，既复现数值又保持 NCNN Mat 的 FP32 ABI。
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        const uint32_t least_significant = (bits >> 16) & 1u;
        bits += 0x7fffu + least_significant;
        bits &= 0xffff0000u;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }
} // namespace

#if NCNN_VULKAN

// ============================================================================
// shader 1：rmsnorm_reduce —— 逐 token 平方和归约
// ============================================================================
// 与 DiTOutput 的 norm_reduce 同构：每个 workgroup 处理一个 token，strided
// 累加 + shared-memory 树形归约，统计量用 fp32 避免半精度累积误差。
static const char* block_rmsnorm_reduce_shader_source = R"VKGLSL(
#version 450

layout(binding = 0, std430) readonly buffer bottom_data { sfp bottom_blob[]; };
layout(binding = 1, std430) writeonly buffer sqsum_data { float sqsum_blob[]; };

layout(push_constant) uniform parameter {
    uint dim;
    uint tokens;
} p;

shared float sqsum_shared[256];

void main()
{
    uint token = gl_WorkGroupID.x;
    uint tid = gl_LocalInvocationID.x;
    uint nthreads = gl_WorkGroupSize.x;
    if (token >= p.tokens)
        return;

    float local_sqsum = 0.0;
    for (uint c = tid; c < p.dim; c += nthreads)
    {
        float v = buffer_ld1(bottom_blob, token * p.dim + c);
        local_sqsum += v * v;
    }
    sqsum_shared[tid] = local_sqsum;
    barrier();
    for (uint s = nthreads / 2u; s > 0u; s >>= 1u)
    {
        if (tid < s)
            sqsum_shared[tid] += sqsum_shared[tid + s];
        barrier();
    }
    if (tid == 0u)
        sqsum_blob[token] = sqsum_shared[0];
}
)VKGLSL";

// ============================================================================
// shader 2：rmsnorm_apply —— 逐 token RMSNorm + adaLN 输入调制
// ============================================================================
// 融合 apply_rmsnorm（无 gamma）与 apply_ada_input：
//   result = data * (inverse_rms * (emb[c*6+slot+1] + scale[c])) + (emb[c*6+slot] + shift[c])
// has_ada=0 时退化为纯 RMSNorm（last_layer 的文本分支只做 RMSNorm）。
static const char* block_rmsnorm_apply_shader_source = R"VKGLSL(
#version 450

layout(binding = 0, std430) readonly buffer bottom_data { sfp bottom_blob[]; };
layout(binding = 1, std430) writeonly buffer top_data { sfp top_blob[]; };
layout(binding = 2, std430) readonly buffer scale_data { sfp scale_blob[]; };
layout(binding = 3, std430) readonly buffer shift_data { sfp shift_blob[]; };
layout(binding = 4, std430) readonly buffer emb_data { sfp emb_blob[]; };
layout(binding = 5, std430) readonly buffer sqsum_data { float sqsum_blob[]; };

layout(push_constant) uniform parameter {
    uint dim;
    float eps;
    uint slot;
    uint has_ada;
    uint total;
} p;

void main()
{
    uint gi = gl_GlobalInvocationID.x;
    if (gi >= p.total)
        return;
    uint c = gi % p.dim;
    uint token = gi / p.dim;

    float inv_rms = 1.0 / sqrt(sqsum_blob[token] / float(p.dim) + p.eps);
    afp v = buffer_ld1(bottom_blob, token * p.dim + c);

    afp result;
    if (p.has_ada == 0u)
    {
        result = v * inv_rms;
    }
    else
    {
        afp s = buffer_ld1(scale_blob, c);
        afp sh = buffer_ld1(shift_blob, c);
        afp e0 = buffer_ld1(emb_blob, c * 6u + p.slot);
        afp e1 = buffer_ld1(emb_blob, c * 6u + p.slot + 1u);
        result = v * (inv_rms * (e1 + s)) + (e0 + sh);
    }
    buffer_st1(top_blob, token * p.dim + c, result);
}
)VKGLSL";

// ============================================================================
// shader 3：qkv_prepare —— Q/K/V 的 RMSNorm + MM-RoPE 旋转 + bf16 舍入
// ============================================================================
// 每个 work item 处理一个 (head, token)，串行遍历 head_dim 维。token 分两类：
//   - 视频 token（token < video_length）：使用视频分支 QK Norm，MM-RoPE 坐标为
//     (text_length + t_local, y_local, x_local)；
//   - 文本 token（token >= video_length）：使用文本分支 QK Norm，MM-RoPE 坐标为
//     (text,text,text)。这对应当前 3B 权重使用的 dit_v2 mmrope3d，而非 dit/v1。
//
// q/k 流程：head_dim 维 RMSNorm → 乘 norm_q/norm_k 的 gamma → MM-RoPE 旋转前
// 126 维（3 axis × 21 pair × 2）→ bf16 舍入。v 只做 bf16 舍入。
static const char* block_qkv_prepare_shader_source = R"VKGLSL(
#version 450

layout(binding = 0, std430) readonly buffer vid_qkv_data { sfp vid_qkv_blob[]; };
layout(binding = 1, std430) readonly buffer txt_qkv_data { sfp txt_qkv_blob[]; };
layout(binding = 2, std430) writeonly buffer q_data { sfp q_blob[]; };
layout(binding = 3, std430) writeonly buffer k_data { sfp k_blob[]; };
layout(binding = 4, std430) writeonly buffer v_data { sfp v_blob[]; };
layout(binding = 5, std430) readonly buffer norm_q_data { sfp norm_q_blob[]; };
layout(binding = 6, std430) readonly buffer norm_k_data { sfp norm_k_blob[]; };
layout(binding = 7, std430) readonly buffer freqs_data { float freqs_blob[]; };
layout(binding = 8, std430) readonly buffer txt_norm_q_data { sfp txt_norm_q_blob[]; };
layout(binding = 9, std430) readonly buffer txt_norm_k_data { sfp txt_norm_k_blob[]; };

layout(push_constant) uniform parameter {
    uint heads;
    uint head_dim;
    uint seq;
    uint video_length;
    uint text_length;
    uint local_h;
    uint local_w;
    uint h0;
    uint w0;
    uint t0;
    uint height;
    uint width;
    uint dim;
    float eps;
    uint total;
} p;

float bf16(float value)
{
    uint bits = floatBitsToUint(value);
    uint ls = (bits >> 16u) & 1u;
    bits += 0x7fffu + ls;
    bits &= 0xffff0000u;
    return uintBitsToFloat(bits);
}

void main()
{
    uint gi = gl_GlobalInvocationID.x;
    if (gi >= p.total)
        return;
    uint head = gi / p.seq;
    uint token = gi % p.seq;

    // 解码源 token 索引与 RoPE 位置。
    uint src_token;
    uint pos_t, pos_h, pos_w;
    bool is_text = token >= p.video_length;
    if (is_text)
    {
        uint text = token - p.video_length;
        src_token = text;
        pos_t = text; pos_h = text; pos_w = text;
    }
    else
    {
        uint local_wh = p.local_h * p.local_w;
        uint t_local = token / local_wh;
        uint yx = token % local_wh;
        uint y_local = yx / p.local_w;
        uint x_local = yx % p.local_w;
        uint t = p.t0 + t_local;
        uint y = p.h0 + y_local;
        uint x = p.w0 + x_local;
        src_token = (t * p.height + y) * p.width + x;
        pos_t = p.text_length + t_local;
        pos_h = y_local;
        pos_w = x_local;
    }

    // 源基地址（vid_qkv/txt_qkv 均为 [*, dim*3]，q/k/v 各占 dim 段）。
    uint src_base = src_token * (p.dim * 3u) + head * p.head_dim;
    uint q_src = src_base;
    uint k_src = src_base + p.dim;
    uint v_src = src_base + p.dim * 2u;

    // 阶段一：q/k 的 head_dim 维平方和（RMSNorm 的 reduce）。
    float q_sqsum = 0.0;
    float k_sqsum = 0.0;
    for (uint d = 0u; d < p.head_dim; d++)
    {
        float qv = is_text ? buffer_ld1(txt_qkv_blob, q_src + d) : buffer_ld1(vid_qkv_blob, q_src + d);
        float kv = is_text ? buffer_ld1(txt_qkv_blob, k_src + d) : buffer_ld1(vid_qkv_blob, k_src + d);
        q_sqsum += qv * qv;
        k_sqsum += kv * kv;
    }
    float q_inv = 1.0 / sqrt(q_sqsum / float(p.head_dim) + p.eps);
    float k_inv = 1.0 / sqrt(k_sqsum / float(p.head_dim) + p.eps);

    // 阶段二：归一化 + gamma，暂存到局部数组（RoPE 需要全部 head_dim 维）。
    float q_local[128];
    float k_local[128];
    for (uint d = 0u; d < p.head_dim; d++)
    {
        float qv = is_text ? buffer_ld1(txt_qkv_blob, q_src + d) : buffer_ld1(vid_qkv_blob, q_src + d);
        float kv = is_text ? buffer_ld1(txt_qkv_blob, k_src + d) : buffer_ld1(vid_qkv_blob, k_src + d);
        float q_gamma = is_text ? buffer_ld1(txt_norm_q_blob, d) : buffer_ld1(norm_q_blob, d);
        float k_gamma = is_text ? buffer_ld1(txt_norm_k_blob, d) : buffer_ld1(norm_k_blob, d);
        q_local[d] = qv * q_inv * q_gamma;
        k_local[d] = kv * k_inv * k_gamma;
    }

    // dit_v2 NaMMRotaryEmbedding3d 同时旋转视频和文本，二者使用不同坐标。
    for (uint axis = 0u; axis < 3u; axis++)
    {
        float pos = (axis == 0u) ? float(pos_t) : ((axis == 1u) ? float(pos_h) : float(pos_w));
        for (uint pair = 0u; pair < 21u; pair++)
        {
            uint d = axis * 42u + pair * 2u;
            if (d + 1u >= p.head_dim)
                break;
            float angle = pos * freqs_blob[pair];
            float cosine = cos(angle);
            float sine = sin(angle);
            float f0 = q_local[d], f1 = q_local[d + 1u];
            q_local[d] = f0 * cosine - f1 * sine;
            q_local[d + 1u] = f1 * cosine + f0 * sine;
            f0 = k_local[d]; f1 = k_local[d + 1u];
            k_local[d] = f0 * cosine - f1 * sine;
            k_local[d + 1u] = f1 * cosine + f0 * sine;
        }
    }

    // 阶段四：bf16 舍入后写出 q/k/v。
    uint out_base = (head * p.seq + token) * p.head_dim;
    for (uint d = 0u; d < p.head_dim; d++)
    {
        float vv = is_text ? buffer_ld1(txt_qkv_blob, v_src + d) : buffer_ld1(vid_qkv_blob, v_src + d);
        buffer_st1(q_blob, out_base + d, bf16(q_local[d]));
        buffer_st1(k_blob, out_base + d, bf16(k_local[d]));
        buffer_st1(v_blob, out_base + d, bf16(vv));
    }
}
)VKGLSL";

// ============================================================================
// shader 4：attention —— softmax(QK^T * scale) @ V，三遍 QK^T 不物化 scores
// ============================================================================
// 每个 work item 处理一个 (head, query)。三遍遍历 key：
//   ① 找 max（softmax 数值稳定）；② 算 sum(exp)；③ 加权累加 v。
// 视频 query 结果写 vid_out_ws 的当前窗口段（shifted 窗口时跨窗口累加由
// window_sum 统一完成）；文本 query 结果写 txt_out_ws 的当前窗口段。
static const char* block_attention_shader_source = R"VKGLSL(
#version 450

layout(binding = 0, std430) readonly buffer q_data { sfp q_blob[]; };
layout(binding = 1, std430) readonly buffer k_data { sfp k_blob[]; };
layout(binding = 2, std430) readonly buffer v_data { sfp v_blob[]; };
layout(binding = 3, std430) writeonly buffer vid_out_data { sfp vid_out_blob[]; };
layout(binding = 4, std430) writeonly buffer txt_out_data { sfp txt_out_blob[]; };

layout(push_constant) uniform parameter {
    uint head_dim;
    uint seq;
    uint video_length;
    uint text_length;
    uint video_tokens;
    uint local_h;
    uint local_w;
    uint h0;
    uint w0;
    uint t0;
    uint height;
    uint width;
    uint dim;
    uint window_index;
    float scale;
    uint total;
} p;

void main()
{
    uint gi = gl_GlobalInvocationID.x;
    if (gi >= p.total)
        return;
    uint head = gi / p.seq;
    uint query = gi % p.seq;

    uint q_base = (head * p.seq + query) * p.head_dim;

    // 第一遍：找 max。
    float maximum = -1.0e30;
    for (uint key = 0u; key < p.seq; key++)
    {
        uint k_base = (head * p.seq + key) * p.head_dim;
        float dot = 0.0;
        for (uint d = 0u; d < p.head_dim; d++)
            dot += buffer_ld1(q_blob, q_base + d) * buffer_ld1(k_blob, k_base + d);
        float s = dot * p.scale;
        if (s > maximum)
            maximum = s;
    }

    // 第二遍：算 sum(exp)。
    float denominator = 0.0;
    for (uint key = 0u; key < p.seq; key++)
    {
        uint k_base = (head * p.seq + key) * p.head_dim;
        float dot = 0.0;
        for (uint d = 0u; d < p.head_dim; d++)
            dot += buffer_ld1(q_blob, q_base + d) * buffer_ld1(k_blob, k_base + d);
        denominator += exp(dot * p.scale - maximum);
    }

    // 第三遍：加权累加 v。
    float acc[128];
    for (uint d = 0u; d < p.head_dim; d++)
        acc[d] = 0.0;
    for (uint key = 0u; key < p.seq; key++)
    {
        uint k_base = (head * p.seq + key) * p.head_dim;
        float dot = 0.0;
        for (uint d = 0u; d < p.head_dim; d++)
            dot += buffer_ld1(q_blob, q_base + d) * buffer_ld1(k_blob, k_base + d);
        float w = exp(dot * p.scale - maximum) / denominator;
        for (uint d = 0u; d < p.head_dim; d++)
            acc[d] += w * buffer_ld1(v_blob, k_base + d);
    }

    // 写出结果到当前窗口段。
    if (query < p.video_length)
    {
        // 解码窗口内视频 token 的全局 flat 索引。
        uint local_wh = p.local_h * p.local_w;
        uint t_local = query / local_wh;
        uint yx = query % local_wh;
        uint y_local = yx / p.local_w;
        uint x_local = yx % p.local_w;
        uint global_index = ((p.t0 + t_local) * p.height + (p.h0 + y_local)) * p.width + (p.w0 + x_local);
        uint out_base = p.window_index * (p.video_tokens * p.dim) + global_index * p.dim + head * p.head_dim;
        for (uint d = 0u; d < p.head_dim; d++)
            buffer_st1(vid_out_blob, out_base + d, acc[d]);
    }
    else
    {
        uint text = query - p.video_length;
        uint out_base = p.window_index * (p.text_length * p.dim) + text * p.dim + head * p.head_dim;
        for (uint d = 0u; d < p.head_dim; d++)
            buffer_st1(txt_out_blob, out_base + d, acc[d]);
    }
}
)VKGLSL";

// attention 每个窗口只写 vid_out_ws 中属于该窗口的 token。共享 allocator 会
// 复用旧显存，未覆盖区域不能假定为零，因此归约前必须显式清空整个稀疏缓冲。
static const char* block_zero_buffer_shader_source = R"VKGLSL(
#version 450

layout(binding = 0, std430) writeonly buffer output_data { sfp output_blob[]; };

layout(push_constant) uniform parameter {
    uint total;
} p;

void main()
{
    uint i = gl_GlobalInvocationID.x;
    if (i < p.total)
        buffer_st1(output_blob, i, 0.0);
}
)VKGLSL";

// ============================================================================
// shader 5：ada_residual —— 输出调制（门控残差 / 普通残差 / 两倍，三模式）
// ============================================================================
//   mode=0（门控残差）：result = data * (emb[c*6 + slot] + gate[c]) + residual[c]
//   mode=1（普通残差）：result = data + residual[c]（last_layer 文本跳过 gate）
//   mode=2（两倍）    ：result = data * 2（last_layer 文本的 2*txt_attn 语义）
static const char* block_ada_residual_shader_source = R"VKGLSL(
#version 450

layout(binding = 0, std430) buffer data { sfp data_blob[]; };
layout(binding = 1, std430) readonly buffer residual_data { sfp residual_blob[]; };
layout(binding = 2, std430) readonly buffer gate_data { sfp gate_blob[]; };
layout(binding = 3, std430) readonly buffer emb_data { sfp emb_blob[]; };

layout(push_constant) uniform parameter {
    uint dim;
    uint slot;
    uint mode;
    uint total;
} p;

void main()
{
    uint gi = gl_GlobalInvocationID.x;
    if (gi >= p.total)
        return;
    uint c = gi % p.dim;
    uint token = gi / p.dim;

    afp d = buffer_ld1(data_blob, token * p.dim + c);
    afp result;
    if (p.mode == 0u)
    {
        afp r = buffer_ld1(residual_blob, token * p.dim + c);
        afp g = buffer_ld1(gate_blob, c);
        afp e = buffer_ld1(emb_blob, c * 6u + p.slot);
        result = d * (e + g) + r;
    }
    else if (p.mode == 1u)
    {
        afp r = buffer_ld1(residual_blob, token * p.dim + c);
        result = d + r;
    }
    else
    {
        result = d * 2.0;
    }
    buffer_st1(data_blob, token * p.dim + c, result);
}
)VKGLSL";

// ============================================================================
// shader 6：silu —— SwiGLU 门控（in-place）
// ============================================================================
// gate[i] = silu(gate[i]) * value[i]
static const char* block_silu_shader_source = R"VKGLSL(
#version 450

layout(binding = 0, std430) buffer gate_data { sfp gate_blob[]; };
layout(binding = 1, std430) readonly buffer value_data { sfp value_blob[]; };

layout(push_constant) uniform parameter {
    uint total;
} p;

void main()
{
    uint i = gl_GlobalInvocationID.x;
    if (i >= p.total)
        return;
    afp g = buffer_ld1(gate_blob, i);
    afp v = buffer_ld1(value_blob, i);
    buffer_st1(gate_blob, i, (g / (1.0 + exp(-g))) * v);
}
)VKGLSL";

// ============================================================================
// shader 7：window_sum —— 跨窗口归约（视频累加 / 文本平均）
// ============================================================================
// src 布局 [windows, tokens, dim]，每个 work item 处理一个 (token, channel)，
// 累加 windows 个窗口段；has_divide=1 时除以 windows（文本平均），否则直接累加
// （视频 shifted 窗口的跨窗口累加）。
static const char* block_window_sum_shader_source = R"VKGLSL(
#version 450

layout(binding = 0, std430) readonly buffer src_data { sfp src_blob[]; };
layout(binding = 1, std430) writeonly buffer dst_data { sfp dst_blob[]; };

layout(push_constant) uniform parameter {
    uint dim;
    uint tokens;
    uint windows;
    uint has_divide;
    uint total;
} p;

void main()
{
    uint gi = gl_GlobalInvocationID.x;
    if (gi >= p.total)
        return;
    uint channel = gi % p.dim;
    uint token = gi / p.dim;

    float sum = 0.0;
    for (uint w = 0u; w < p.windows; w++)
    {
        uint idx = w * (p.tokens * p.dim) + token * p.dim + channel;
        sum += buffer_ld1(src_blob, idx);
    }
    afp result = (p.has_divide == 1u) ? afp(sum / float(p.windows)) : afp(sum);
    buffer_st1(dst_blob, token * p.dim + channel, result);
}
)VKGLSL";

#endif // NCNN_VULKAN

SeedVR2DiTBlock::SeedVR2DiTBlock()
    : block_index(0),
      shared_weights(false),
      last_layer(false),
      shifted_window(false),
      dim(2560),
      heads(20),
      head_dim(128),
      mlp_hidden(6912),
      norm_eps(1e-5f),
      runtime_frames(0),
      runtime_height(0),
      runtime_width(0),
      runtime_shape_valid(false),
      debug_tensors(nullptr)
{
    one_blob_only = false;
    support_inplace = false;
    support_packing = false;
#if NCNN_VULKAN
    support_vulkan = true;
    support_vulkan_packing = false;
    pipeline_rmsnorm_reduce = 0;
    pipeline_rmsnorm_apply = 0;
    pipeline_qkv_prepare = 0;
    pipeline_attention = 0;
    pipeline_ada_residual = 0;
    pipeline_silu = 0;
    pipeline_zero_buffer = 0;
    pipeline_text_normalize = 0;
#endif
}

void SeedVR2DiTBlock::set_runtime_shape(int frames, int height, int width)
{
    runtime_frames = frames;
    runtime_height = height;
    runtime_width = width;
    runtime_shape_valid = frames > 0 && height > 0 && width > 0;
}

void SeedVR2DiTBlock::set_debug_tensors(SeedVR2DiTBlockDebugTensors* tensors)
{
    debug_tensors = tensors;
}

SeedVR2DiTBlock::~SeedVR2DiTBlock()
{
    destroy_branch(vid_weights);
    if (!shared_weights)
        destroy_branch(txt_weights);
}

void SeedVR2DiTBlock::destroy_branch(BranchWeights &branch)
{
    delete branch.qkv;
    delete branch.proj_out;
    delete branch.mlp_gate_proj;
    delete branch.mlp_in_proj;
    delete branch.mlp_out_proj;
    branch.qkv = nullptr;
    branch.proj_out = nullptr;
    branch.mlp_gate_proj = nullptr;
    branch.mlp_in_proj = nullptr;
    branch.mlp_out_proj = nullptr;
}

int SeedVR2DiTBlock::load_param(const ncnn::ParamDict &pd)
{
    block_index = pd.get(0, 0);
    shared_weights = pd.get(1, 0) != 0;
    last_layer = pd.get(2, 0) != 0;
    shifted_window = pd.get(3, 0) != 0;
    dim = pd.get(4, 2560);
    heads = pd.get(5, 20);
    head_dim = pd.get(6, 128);
    mlp_hidden = pd.get(7, 6912);
    norm_eps = pd.get(8, 1e-5f);
    return dim == heads * head_dim ? 0 : -1;
}

int SeedVR2DiTBlock::load_branch(const ncnn::ModelBin &mb, BranchWeights &branch)
{
    branch.attn_shift = mb.load(dim, 1);
    branch.attn_scale = mb.load(dim, 1);
    branch.attn_gate = mb.load(dim, 1);
    branch.mlp_shift = mb.load(dim, 1);
    branch.mlp_scale = mb.load(dim, 1);
    branch.mlp_gate = mb.load(dim, 1);
    if (branch.attn_shift.empty() || branch.attn_scale.empty() || branch.attn_gate.empty() || branch.mlp_shift.empty() || branch.mlp_scale.empty() || branch.mlp_gate.empty())
        return -100;

    branch.qkv = load_inner_product(mb, dim, dim * 3, false, vkdev);
    branch.proj_out = load_inner_product(mb, dim, dim, true, vkdev);
    branch.norm_q = mb.load(head_dim, 1);
    branch.norm_k = mb.load(head_dim, 1);
    branch.mlp_gate_proj = load_inner_product(mb, dim, mlp_hidden, false, vkdev);
    branch.mlp_in_proj = load_inner_product(mb, dim, mlp_hidden, false, vkdev);
    branch.mlp_out_proj = load_inner_product(mb, mlp_hidden, dim, false, vkdev);
    if (!branch.qkv || !branch.proj_out || branch.norm_q.empty() || branch.norm_k.empty() || !branch.mlp_gate_proj || !branch.mlp_in_proj || !branch.mlp_out_proj)
        return -100;
    return 0;
}

int SeedVR2DiTBlock::load_model(const ncnn::ModelBin &mb)
{
    if (load_branch(mb, vid_weights) != 0)
        return -100;
    if (!shared_weights && load_branch(mb, txt_weights) != 0)
        return -100;
    rope_freqs = mb.load(21, 1);
    return rope_freqs.empty() ? -100 : 0;
}

int SeedVR2DiTBlock::create_pipeline(const ncnn::Option &opt)
{
    auto create_branch = [&](BranchWeights &branch)
    {
        for (ncnn::Layer *layer : {branch.qkv, branch.proj_out, branch.mlp_gate_proj,
                                   branch.mlp_in_proj, branch.mlp_out_proj})
        {
            if (!layer)
                return -1;
#if NCNN_VULKAN
            if (vkdev)
                layer->vkdev = vkdev;
#endif
            if (layer->create_pipeline(opt) != 0)
                return -1;
        }
        return 0;
    };
    if (create_branch(vid_weights) != 0)
        return -1;
    if (!shared_weights && create_branch(txt_weights) != 0)
        return -1;

#if NCNN_VULKAN
    if (!vkdev)
        return 0;

    auto compile = [&](const char* source, ncnn::Pipeline*& pipeline, int local_x, const char* name) {
        std::vector<uint32_t> spirv;
        const int ret = ncnn::compile_spirv_module(source, opt, spirv);
        if (ret != 0)
        {
            std::fprintf(stderr, "SeedVR2DiTBlock[%d]: %s compile failed %d\n", block_index, name, ret);
            return ret;
        }
        pipeline = new ncnn::Pipeline(vkdev);
        pipeline->set_optimal_local_size_xyz(local_x, 1, 1);
        pipeline->create(spirv.data(), spirv.size() * sizeof(uint32_t), std::vector<ncnn::vk_specialization_type>());
        return 0;
    };

    int ret;
    if ((ret = compile(block_rmsnorm_reduce_shader_source, pipeline_rmsnorm_reduce, 256, "rmsnorm_reduce")) != 0) return ret;
    if ((ret = compile(block_rmsnorm_apply_shader_source, pipeline_rmsnorm_apply, 64, "rmsnorm_apply")) != 0) return ret;
    if ((ret = compile(block_qkv_prepare_shader_source, pipeline_qkv_prepare, 64, "qkv_prepare")) != 0) return ret;
    if ((ret = compile(block_attention_shader_source, pipeline_attention, 64, "attention")) != 0) return ret;
    if ((ret = compile(block_ada_residual_shader_source, pipeline_ada_residual, 64, "ada_residual")) != 0) return ret;
    if ((ret = compile(block_silu_shader_source, pipeline_silu, 64, "silu")) != 0) return ret;
    if ((ret = compile(block_zero_buffer_shader_source, pipeline_zero_buffer, 64, "zero_buffer")) != 0) return ret;
    if ((ret = compile(block_window_sum_shader_source, pipeline_text_normalize, 64, "window_sum")) != 0) return ret;
#endif
    return 0;
}

int SeedVR2DiTBlock::destroy_pipeline(const ncnn::Option &opt)
{
    auto destroy = [&](BranchWeights &branch)
    {
        for (ncnn::Layer *layer : {branch.qkv, branch.proj_out, branch.mlp_gate_proj,
                                   branch.mlp_in_proj, branch.mlp_out_proj})
            if (layer)
                layer->destroy_pipeline(opt);
    };
    destroy(vid_weights);
    if (!shared_weights)
        destroy(txt_weights);

#if NCNN_VULKAN
    delete pipeline_rmsnorm_reduce; pipeline_rmsnorm_reduce = 0;
    delete pipeline_rmsnorm_apply; pipeline_rmsnorm_apply = 0;
    delete pipeline_qkv_prepare; pipeline_qkv_prepare = 0;
    delete pipeline_attention; pipeline_attention = 0;
    delete pipeline_ada_residual; pipeline_ada_residual = 0;
    delete pipeline_silu; pipeline_silu = 0;
    delete pipeline_zero_buffer; pipeline_zero_buffer = 0;
    delete pipeline_text_normalize; pipeline_text_normalize = 0;
#endif
    return 0;
}

void SeedVR2DiTBlock::apply_rmsnorm(ncnn::Mat &value) const
{
    for (int row = 0; row < value.h; row++)
    {
        float *data = value.row(row);
        double square_sum = 0.0;
        for (int channel = 0; channel < dim; channel++)
            square_sum += static_cast<double>(data[channel]) * data[channel];
        const float inverse_rms = 1.f / std::sqrt(static_cast<float>(square_sum / dim) + norm_eps);
        for (int channel = 0; channel < dim; channel++)
            data[channel] *= inverse_rms;
    }
}

void SeedVR2DiTBlock::apply_ada_input(
    ncnn::Mat &value,
    const ncnn::Mat &embedding,
    const ncnn::Mat &shift,
    const ncnn::Mat &scale,
    int layer_index) const
{
    const float *emb = embedding;
    const float *shift_b = shift;
    const float *scale_b = scale;
    const int slot = layer_index * 3;
#pragma omp parallel for
    for (int row = 0; row < value.h; row++)
    {
        float *data = value.row(row);
        for (int channel = 0; channel < dim; channel++)
        {
            const int offset = channel * 6 + slot;
            data[channel] = data[channel] * (emb[offset + 1] + scale_b[channel]) + emb[offset] + shift_b[channel];
        }
    }
}

void SeedVR2DiTBlock::apply_ada_output_and_residual(
    ncnn::Mat &value,
    const ncnn::Mat &residual,
    const ncnn::Mat &embedding,
    const ncnn::Mat &gate,
    int layer_index) const
{
    const float *emb = embedding;
    const float *gate_b = gate;
    const int slot = layer_index * 3 + 2;
#pragma omp parallel for
    for (int row = 0; row < value.h; row++)
    {
        float *data = value.row(row);
        const float *skip = residual.row(row);
        for (int channel = 0; channel < dim; channel++)
            data[channel] = data[channel] * (emb[channel * 6 + slot] + gate_b[channel]) + skip[channel];
    }
}

std::vector<SeedVR2DiTBlock::Window> SeedVR2DiTBlock::make_windows(
    int frames,
    int height,
    int width) const
{
    const double scale = std::sqrt(3600.0 / static_cast<double>(height * width));
    // Python round 使用 ties-to-even；nearbyint 在默认舍入模式下保持相同规则。
    const int resized_height = static_cast<int>(std::nearbyint(height * scale));
    const int resized_width = static_cast<int>(std::nearbyint(width * scale));
    const int window_height = ceil_div(resized_height, 3);
    const int window_width = ceil_div(resized_width, 3);
    const int window_frames = ceil_div(std::min(frames, 30), 4);

    const double shift_t = shifted_window && window_frames < frames ? 0.5 : 0.0;
    const double shift_h = shifted_window && window_height < height ? 0.5 : 0.0;
    const double shift_w = shifted_window && window_width < width ? 0.5 : 0.0;
    int count_t = shifted_window
                      ? (shift_t > 0.0 ? static_cast<int>(std::ceil((frames - shift_t) / window_frames)) + 1 : 1)
                      : ceil_div(frames, window_frames);
    int count_h = shifted_window
                      ? (shift_h > 0.0 ? static_cast<int>(std::ceil((height - shift_h) / window_height)) + 1 : 1)
                      : ceil_div(height, window_height);
    int count_w = shifted_window
                      ? (shift_w > 0.0 ? static_cast<int>(std::ceil((width - shift_w) / window_width)) + 1 : 1)
                      : ceil_div(width, window_width);

    std::vector<Window> windows;
    for (int iw = 0; iw < count_w; iw++)
        for (int ih = 0; ih < count_h; ih++)
            for (int it = 0; it < count_t; it++)
            {
                const int t0 = std::max(static_cast<int>((it - shift_t) * window_frames), 0);
                const int t1 = std::min(static_cast<int>((it - shift_t + 1.0) * window_frames), frames);
                const int h0 = std::max(static_cast<int>((ih - shift_h) * window_height), 0);
                const int h1 = std::min(static_cast<int>((ih - shift_h + 1.0) * window_height), height);
                const int w0 = std::max(static_cast<int>((iw - shift_w) * window_width), 0);
                const int w1 = std::min(static_cast<int>((iw - shift_w + 1.0) * window_width), width);
                if (t1 > t0 && h1 > h0 && w1 > w0)
                    windows.push_back({t0, t1, h0, h1, w0, w1});
            }
    return windows;
}

int SeedVR2DiTBlock::attention(
    const ncnn::Mat &vid_qkv,
    const ncnn::Mat &txt_qkv,
    const ncnn::Mat &vid_shape,
    const BranchWeights &vid_branch,
    const BranchWeights &txt_branch,
    ncnn::Mat &vid_output,
    ncnn::Mat &txt_output,
    const ncnn::Option &opt) const
{
    const int frames = static_cast<const int *>(vid_shape)[0];
    const int height = static_cast<const int *>(vid_shape)[1];
    const int width = static_cast<const int *>(vid_shape)[2];
    if (frames * height * width != vid_qkv.h)
        return -1;
    const int text_length = txt_qkv.h;
    const std::vector<Window> windows = make_windows(frames, height, width);
    if (windows.empty())
        return -1;

    vid_output.create(dim, vid_qkv.h, 4u, opt.blob_allocator);
    txt_output.create(dim, text_length, 4u, opt.blob_allocator);
    if (vid_output.empty() || txt_output.empty())
        return -100;
    vid_output.fill(0.f);
    txt_output.fill(0.f);
    const float attention_scale = 1.f / std::sqrt(static_cast<float>(head_dim));
    const float *frequencies = rope_freqs;

    std::vector<int> sequence_offsets(windows.size());
    int captured_sequence_length = 0;
    for (size_t index = 0; index < windows.size(); ++index)
    {
        sequence_offsets[index] = captured_sequence_length;
        const Window& window = windows[index];
        captured_sequence_length += (window.t1 - window.t0) * (window.h1 - window.h0)
                                    * (window.w1 - window.w0) + text_length;
    }
    if (debug_tensors)
    {
        debug_tensors->norm_q_vid.create(dim, vid_qkv.h);
        debug_tensors->norm_k_vid.create(dim, vid_qkv.h);
        debug_tensors->norm_q_txt.create(dim, text_length);
        debug_tensors->norm_k_txt.create(dim, text_length);
        debug_tensors->attention_q.create(dim, captured_sequence_length);
        debug_tensors->attention_k.create(dim, captured_sequence_length);
        debug_tensors->attention_v.create(dim, captured_sequence_length);
        debug_tensors->attention_output.create(dim, captured_sequence_length);
    }

    // 每个 head 独占自己的输出通道，窗口在同一 head 内顺序执行，避免文本
    // 汇聚产生写冲突；视频窗口本身是分区，每个 token 只写一次。
#pragma omp parallel for num_threads(opt.num_threads)
    for (int head = 0; head < heads; head++)
    {
        std::vector<float> text_accumulator(static_cast<size_t>(text_length) * head_dim, 0.f);
        for (size_t window_index = 0; window_index < windows.size(); ++window_index)
        {
            const Window& window = windows[window_index];
            const int sequence_offset = sequence_offsets[window_index];
            const int local_t = window.t1 - window.t0;
            const int local_h = window.h1 - window.h0;
            const int local_w = window.w1 - window.w0;
            const int video_length = local_t * local_h * local_w;
            const int sequence_length = video_length + text_length;
            std::vector<int> video_indices;
            video_indices.reserve(video_length);
            std::vector<float> q(static_cast<size_t>(sequence_length) * head_dim);
            std::vector<float> k(static_cast<size_t>(sequence_length) * head_dim);
            std::vector<float> v(static_cast<size_t>(sequence_length) * head_dim);
            std::vector<float> scores(sequence_length);

            auto normalize_and_rotate = [&](const float *source, const float *gamma,
                                            float *destination, bool apply_rope,
                                            int position_t, int position_h, int position_w,
                                            float* normalized_debug)
            {
                double square_sum = 0.0;
                for (int d = 0; d < head_dim; d++)
                    square_sum += static_cast<double>(source[d]) * source[d];
                const float inverse_rms = 1.f / std::sqrt(static_cast<float>(square_sum / head_dim) + norm_eps);
                for (int d = 0; d < head_dim; d++)
                    destination[d] = source[d] * inverse_rms * gamma[d];
                if (normalized_debug)
                    std::memcpy(normalized_debug, destination,
                                static_cast<size_t>(head_dim) * sizeof(float));
                if (apply_rope)
                {
                    const int positions[3] = {position_t, position_h, position_w};
                    for (int axis = 0; axis < 3; axis++)
                        for (int pair = 0; pair < 21; pair++)
                        {
                            const int d = axis * 42 + pair * 2;
                            const float angle = positions[axis] * frequencies[pair];
                            const float cosine = std::cos(angle);
                            const float sine = std::sin(angle);
                            const float first = destination[d];
                            const float second = destination[d + 1];
                            destination[d] = first * cosine - second * sine;
                            destination[d + 1] = second * cosine + first * sine;
                        }
                }
                for (int d = 0; d < head_dim; d++)
                    destination[d] = round_to_bfloat16(destination[d]);
            };

            int token = 0;
            for (int t = window.t0; t < window.t1; t++)
                for (int y = window.h0; y < window.h1; y++)
                    for (int x = window.w0; x < window.w1; x++, token++)
                    {
                        const int index = (t * height + y) * width + x;
                        video_indices.push_back(index);
                        const float *row = vid_qkv.row(index);
                        normalize_and_rotate(
                            row + head * head_dim,
                            vid_branch.norm_q,
                            q.data() + static_cast<size_t>(token) * head_dim,
                            true, text_length + (t - window.t0), y - window.h0, x - window.w0,
                            debug_tensors ? debug_tensors->norm_q_vid.row(index) + head * head_dim : nullptr);
                        normalize_and_rotate(
                            row + dim + head * head_dim,
                            vid_branch.norm_k,
                            k.data() + static_cast<size_t>(token) * head_dim,
                            true, text_length + (t - window.t0), y - window.h0, x - window.w0,
                            debug_tensors ? debug_tensors->norm_k_vid.row(index) + head * head_dim : nullptr);
                        std::memcpy(
                            v.data() + static_cast<size_t>(token) * head_dim,
                            row + dim * 2 + head * head_dim,
                            static_cast<size_t>(head_dim) * sizeof(float));
                        float *value = v.data() + static_cast<size_t>(token) * head_dim;
                        for (int d = 0; d < head_dim; d++)
                            value[d] = round_to_bfloat16(value[d]);
                    }
            for (int text = 0; text < text_length; text++, token++)
            {
                const float *row = txt_qkv.row(text);
                normalize_and_rotate(
                    row + head * head_dim,
                    txt_branch.norm_q,
                    q.data() + static_cast<size_t>(token) * head_dim,
                    true, text, text, text,
                    debug_tensors && window_index == 0
                        ? debug_tensors->norm_q_txt.row(text) + head * head_dim : nullptr);
                normalize_and_rotate(
                    row + dim + head * head_dim,
                    txt_branch.norm_k,
                    k.data() + static_cast<size_t>(token) * head_dim,
                    true, text, text, text,
                    debug_tensors && window_index == 0
                        ? debug_tensors->norm_k_txt.row(text) + head * head_dim : nullptr);
                std::memcpy(
                    v.data() + static_cast<size_t>(token) * head_dim,
                    row + dim * 2 + head * head_dim,
                    static_cast<size_t>(head_dim) * sizeof(float));
                float *value = v.data() + static_cast<size_t>(token) * head_dim;
                for (int d = 0; d < head_dim; d++)
                    value[d] = round_to_bfloat16(value[d]);
            }

            if (debug_tensors)
            {
                for (int sequence = 0; sequence < sequence_length; ++sequence)
                {
                    const size_t source_offset = static_cast<size_t>(sequence) * head_dim;
                    const int capture_row = sequence_offset + sequence;
                    std::memcpy(debug_tensors->attention_q.row(capture_row) + head * head_dim,
                                q.data() + source_offset, static_cast<size_t>(head_dim) * sizeof(float));
                    std::memcpy(debug_tensors->attention_k.row(capture_row) + head * head_dim,
                                k.data() + source_offset, static_cast<size_t>(head_dim) * sizeof(float));
                    std::memcpy(debug_tensors->attention_v.row(capture_row) + head * head_dim,
                                v.data() + source_offset, static_cast<size_t>(head_dim) * sizeof(float));
                }
            }

            for (int query = 0; query < sequence_length; query++)
            {
                const float *query_data = q.data() + static_cast<size_t>(query) * head_dim;
                float maximum = -std::numeric_limits<float>::infinity();
                for (int key = 0; key < sequence_length; key++)
                {
                    const float *key_data = k.data() + static_cast<size_t>(key) * head_dim;
                    double dot = 0.0;
                    for (int d = 0; d < head_dim; d++)
                        dot += static_cast<double>(query_data[d]) * key_data[d];
                    scores[key] = static_cast<float>(dot) * attention_scale;
                    maximum = std::max(maximum, scores[key]);
                }
                float denominator = 0.f;
                for (float &score : scores)
                {
                    score = std::exp(score - maximum);
                    denominator += score;
                }
                float *destination = nullptr;
                if (query < video_length)
                    destination = vid_output.row(video_indices[query]) + head * head_dim;
                else
                    destination = text_accumulator.data() + static_cast<size_t>(query - video_length) * head_dim;
                for (int d = 0; d < head_dim; d++)
                {
                    double sum = 0.0;
                    for (int key = 0; key < sequence_length; key++)
                        sum += static_cast<double>(scores[key] / denominator) * v[static_cast<size_t>(key) * head_dim + d];
                    const float rounded = round_to_bfloat16(static_cast<float>(sum));
                    destination[d] += rounded;
                    if (debug_tensors)
                        debug_tensors->attention_output.row(sequence_offset + query)[head * head_dim + d] = rounded;
                }
            }
        }
        const float inverse_window_count = 1.f / windows.size();
        for (int text = 0; text < text_length; text++)
        {
            float *destination = txt_output.row(text) + head * head_dim;
            const float *source = text_accumulator.data() + static_cast<size_t>(text) * head_dim;
            for (int d = 0; d < head_dim; d++)
                destination[d] = source[d] * inverse_window_count;
        }
    }
    return 0;
}

int SeedVR2DiTBlock::apply_mlp(
    const ncnn::Mat &input,
    const BranchWeights &branch,
    ncnn::Mat &output,
    const ncnn::Option &opt) const
{
    ncnn::Mat gate;
    ncnn::Mat value;
    if (branch.mlp_gate_proj->forward(input, gate, opt) != 0 || branch.mlp_in_proj->forward(input, value, opt) != 0)
        return -100;
#pragma omp parallel for num_threads(opt.num_threads)
    for (int row = 0; row < gate.h; row++)
    {
        float *gate_data = gate.row(row);
        const float *value_data = value.row(row);
        for (int channel = 0; channel < mlp_hidden; channel++)
            gate_data[channel] = silu(gate_data[channel]) * value_data[channel];
    }
    return branch.mlp_out_proj->forward(gate, output, opt);
}

int SeedVR2DiTBlock::forward(
    const std::vector<ncnn::Mat> &bottom_blobs,
    std::vector<ncnn::Mat> &top_blobs,
    const ncnn::Option &opt) const
{
    if (bottom_blobs.size() != 4 || top_blobs.size() != 2)
    {
        std::fprintf(stderr, "SeedVR2DiTBlock[%d]: blob count mismatch (bottom=%zu top=%zu)\n",
                     block_index, bottom_blobs.size(), top_blobs.size());
        return -1;
    }
    const ncnn::Mat &vid = bottom_blobs[0];
    const ncnn::Mat &txt = bottom_blobs[1];
    const ncnn::Mat &embedding = bottom_blobs[2];
    const ncnn::Mat &vid_shape = bottom_blobs[3];
    if (vid.dims != 2 || txt.dims != 2 || vid.w != dim || txt.w != dim || embedding.w != dim * 6 || vid_shape.dims != 1 || vid_shape.w != 3)
    {
        std::fprintf(
            stderr,
            "SeedVR2DiTBlock[%d]: invalid input vid=(dims=%d w=%d h=%d) "
            "txt=(dims=%d w=%d h=%d) emb=%zu shape=%zu expected_dim=%d\n",
            block_index, vid.dims, vid.w, vid.h, txt.dims, txt.w, txt.h,
            embedding.total(), vid_shape.total(), dim);
        return -1;
    }
    const BranchWeights &text_branch = shared_weights ? vid_weights : txt_weights;

    ncnn::Mat vid_attention_input = vid.clone(opt.blob_allocator);
    ncnn::Mat txt_attention_input = txt.clone(opt.blob_allocator);
    if (vid_attention_input.empty() || txt_attention_input.empty())
        return -100;
    apply_rmsnorm(vid_attention_input);
    apply_rmsnorm(txt_attention_input);
    apply_ada_input(vid_attention_input, embedding, vid_weights.attn_shift, vid_weights.attn_scale, 0);
    // 最后一个 PyTorch block 的 Ada MMModule 配置为 vid_only，文本仅经过
    // RMSNorm，不应用 attention shift/scale。
    if (!last_layer)
        apply_ada_input(txt_attention_input, embedding, text_branch.attn_shift, text_branch.attn_scale, 0);

    ncnn::Mat vid_qkv;
    ncnn::Mat txt_qkv;
    const int vid_qkv_result = vid_weights.qkv->forward(vid_attention_input, vid_qkv, opt);
    const int txt_qkv_result = text_branch.qkv->forward(txt_attention_input, txt_qkv, opt);
    if (vid_qkv_result != 0 || txt_qkv_result != 0)
    {
        std::fprintf(stderr, "SeedVR2DiTBlock[%d]: qkv failed (vid=%d txt=%d)\n",
                     block_index, vid_qkv_result, txt_qkv_result);
        return -100;
    }
    if (debug_tensors)
    {
        debug_tensors->qkv_vid = vid_qkv.clone();
        debug_tensors->qkv_txt = txt_qkv.clone();
    }
    ncnn::Mat vid_attention;
    ncnn::Mat txt_attention;
    const int attention_result = attention(vid_qkv, txt_qkv, vid_shape, vid_weights,
                                           text_branch, vid_attention, txt_attention, opt);
    if (attention_result != 0)
    {
        std::fprintf(stderr, "SeedVR2DiTBlock[%d]: attention failed (%d)\n",
                     block_index, attention_result);
        return -100;
    }
    ncnn::Mat vid_projected;
    ncnn::Mat txt_projected;
    const int vid_proj_result = vid_weights.proj_out->forward(vid_attention, vid_projected, opt);
    const int txt_proj_result = text_branch.proj_out->forward(txt_attention, txt_projected, opt);
    if (vid_proj_result != 0 || txt_proj_result != 0)
    {
        std::fprintf(stderr, "SeedVR2DiTBlock[%d]: attention projection failed (vid=%d txt=%d)\n",
                     block_index, vid_proj_result, txt_proj_result);
        return -100;
    }
    if (debug_tensors)
    {
        debug_tensors->projected_vid = vid_projected.clone();
        debug_tensors->projected_txt = txt_projected.clone();
    }
    apply_ada_output_and_residual(vid_projected, vid, embedding, vid_weights.attn_gate, 0);
    if (!last_layer)
    {
        apply_ada_output_and_residual(txt_projected, txt, embedding, text_branch.attn_gate, 0);
    }
    else
    {
        // vid_only 同样跳过文本 attention gate，但 block 外层普通残差仍存在。
#pragma omp parallel for num_threads(opt.num_threads)
        for (int row = 0; row < txt_projected.h; row++)
        {
            float *data = txt_projected.row(row);
            const float *skip = txt.row(row);
            for (int channel = 0; channel < dim; channel++)
                data[channel] += skip[channel];
        }
    }

    ncnn::Mat vid_mlp_input = vid_projected.clone(opt.blob_allocator);
    apply_rmsnorm(vid_mlp_input);
    apply_ada_input(vid_mlp_input, embedding, vid_weights.mlp_shift, vid_weights.mlp_scale, 1);
    ncnn::Mat vid_mlp;
    const int vid_mlp_result = apply_mlp(vid_mlp_input, vid_weights, vid_mlp, opt);
    if (vid_mlp_result != 0)
    {
        std::fprintf(stderr, "SeedVR2DiTBlock[%d]: video MLP failed (%d)\n",
                     block_index, vid_mlp_result);
        return -100;
    }
    apply_ada_output_and_residual(vid_mlp, vid_projected, embedding, vid_weights.mlp_gate, 1);
    top_blobs[0] = vid_mlp;

    if (last_layer)
    {
        // PyTorch 的 MMModule(vid_only=True) 会原样返回 txt，随后 block 仍然
        // 执行 txt_mlp + txt_attn，因此最后一个 block 的文本结果是两倍
        // txt_attn。这里保留该看似多余但对模型数值兼容必需的行为。
        ncnn::Mat txt_last = txt_projected.clone(opt.blob_allocator);
        if (txt_last.empty())
            return -100;
#pragma omp parallel for num_threads(opt.num_threads)
        for (int row = 0; row < txt_last.h; row++)
        {
            float *data = txt_last.row(row);
            for (int channel = 0; channel < dim; channel++)
                data[channel] *= 2.f;
        }
        top_blobs[1] = txt_last;
        if (debug_tensors)
        {
            debug_tensors->output_vid = top_blobs[0].clone();
            debug_tensors->output_txt = top_blobs[1].clone();
        }
        return 0;
    }
    ncnn::Mat txt_mlp_input = txt_projected.clone(opt.blob_allocator);
    apply_rmsnorm(txt_mlp_input);
    apply_ada_input(txt_mlp_input, embedding, text_branch.mlp_shift, text_branch.mlp_scale, 1);
    ncnn::Mat txt_mlp;
    const int txt_mlp_result = apply_mlp(txt_mlp_input, text_branch, txt_mlp, opt);
    if (txt_mlp_result != 0)
    {
        std::fprintf(stderr, "SeedVR2DiTBlock[%d]: text MLP failed (%d)\n",
                     block_index, txt_mlp_result);
        return -100;
    }
    apply_ada_output_and_residual(txt_mlp, txt_projected, embedding, text_branch.mlp_gate, 1);
    top_blobs[1] = txt_mlp;
    if (debug_tensors)
    {
        debug_tensors->output_vid = top_blobs[0].clone();
        debug_tensors->output_txt = top_blobs[1].clone();
    }
    return 0;
}

#if NCNN_VULKAN

int SeedVR2DiTBlock::upload_model(ncnn::VkTransfer& cmd, const ncnn::Option& opt)
{
    // 上传一个分支的 8 个调制权重 + 5 个投影层的权重（InnerProduct 各自 upload）。
    auto upload_branch = [&](BranchWeights& branch) {
        cmd.record_upload(branch.attn_shift, branch.attn_shift_gpu, opt);
        cmd.record_upload(branch.attn_scale, branch.attn_scale_gpu, opt);
        cmd.record_upload(branch.attn_gate, branch.attn_gate_gpu, opt);
        cmd.record_upload(branch.mlp_shift, branch.mlp_shift_gpu, opt);
        cmd.record_upload(branch.mlp_scale, branch.mlp_scale_gpu, opt);
        cmd.record_upload(branch.mlp_gate, branch.mlp_gate_gpu, opt);
        cmd.record_upload(branch.norm_q, branch.norm_q_gpu, opt);
        cmd.record_upload(branch.norm_k, branch.norm_k_gpu, opt);
        for (ncnn::Layer* layer : {branch.qkv, branch.proj_out, branch.mlp_gate_proj,
                                   branch.mlp_in_proj, branch.mlp_out_proj})
        {
            if (layer && layer->upload_model(cmd, opt) != 0)
            {
                std::fprintf(stderr, "SeedVR2DiTBlock[%d]: sublayer upload_model failed\n", block_index);
                return -1;
            }
        }
        return 0;
    };
    if (upload_branch(vid_weights) != 0)
        return -1;
    if (!shared_weights && upload_branch(txt_weights) != 0)
        return -1;
    cmd.record_upload(rope_freqs, rope_freqs_gpu, opt);
    return 0;
}

int SeedVR2DiTBlock::forward(
    const std::vector<ncnn::VkMat>& bottom_blobs,
    std::vector<ncnn::VkMat>& top_blobs,
    ncnn::VkCompute& cmd,
    const ncnn::Option& opt) const
{
    if (bottom_blobs.size() != 4 || top_blobs.size() != 2)
        return -1;
    const ncnn::VkMat& vid = bottom_blobs[0];
    const ncnn::VkMat& txt = bottom_blobs[1];
    const ncnn::VkMat& embedding = bottom_blobs[2];
    const ncnn::VkMat& vid_shape = bottom_blobs[3];
    if (vid.dims != 2 || vid.w != dim || txt.dims != 2 || txt.w != dim
        || embedding.w != dim * 6 || vid_shape.w != 3)
        return -1;

    // 窗口边界由 CPU 生成，但形状在进入 DiT 前已经确定。直接使用调度器注入值，
    // 避免 32 个 block 各自下载 vid_shape 并打断连续 Vulkan 命令流。
    if (!runtime_shape_valid)
        return -1;
    const int frames = runtime_frames;
    const int height = runtime_height;
    const int width = runtime_width;
    if (frames <= 0 || height <= 0 || width <= 0 || vid.h != frames * height * width)
        return -1;
    const int video_tokens = frames * height * width;
    const int text_length = txt.h;
    const std::vector<Window> windows = make_windows(frames, height, width);
    if (windows.empty())
        return -1;
    const int window_count = static_cast<int>(windows.size());
    const BranchWeights& text_branch = shared_weights ? vid_weights : txt_weights;

    // 最大窗口 seq（video + text），用于 workspace 分配。
    int max_seq = 0;
    for (const Window& w : windows)
        max_seq = std::max(max_seq, (w.t1 - w.t0) * (w.h1 - w.h0) * (w.w1 - w.w0) + text_length);

    // ========================================================================
    // 辅助 lambda：rmsnorm_ada —— 逐 token RMSNorm + adaLN 输入调制（reduce+apply）
    // ========================================================================
    auto rmsnorm_ada = [&](const ncnn::VkMat& input, const ncnn::VkMat& scale_gpu,
                           const ncnn::VkMat& shift_gpu, uint slot, uint has_ada,
                           ncnn::VkMat& output) -> int
    {
        const int tokens = input.h;
        ncnn::VkMat sqsum(tokens, 4u, 1, opt.workspace_vkallocator);
        if (sqsum.empty())
            return -100;
        {
            std::vector<ncnn::VkMat> bindings(2);
            bindings[0] = input;
            bindings[1] = sqsum;
            std::vector<ncnn::vk_constant_type> constants(2);
            constants[0].u32 = static_cast<uint32_t>(dim);
            constants[1].u32 = static_cast<uint32_t>(tokens);
            ncnn::VkMat dispatcher;
            dispatcher.w = tokens * pipeline_rmsnorm_reduce->local_size_x();
            dispatcher.h = 1;
            dispatcher.c = 1;
            cmd.record_pipeline(pipeline_rmsnorm_reduce, bindings, constants, dispatcher);
        }
        output.create(dim, tokens, 4u, 1, opt.workspace_vkallocator);
        if (output.empty())
            return -100;
        {
            std::vector<ncnn::VkMat> bindings(6);
            bindings[0] = input;
            bindings[1] = output;
            bindings[2] = scale_gpu;
            bindings[3] = shift_gpu;
            bindings[4] = embedding;
            bindings[5] = sqsum;
            const uint32_t total = static_cast<uint32_t>(tokens) * dim;
            std::vector<ncnn::vk_constant_type> constants(5);
            constants[0].u32 = static_cast<uint32_t>(dim);
            constants[1].f = norm_eps;
            constants[2].u32 = slot;
            constants[3].u32 = has_ada;
            constants[4].u32 = total;
            ncnn::VkMat dispatcher;
            dispatcher.w = total;
            dispatcher.h = 1;
            dispatcher.c = 1;
            cmd.record_pipeline(pipeline_rmsnorm_apply, bindings, constants, dispatcher);
        }
        return 0;
    };

    // ========================================================================
    // 辅助 lambda：ada_residual —— 门控残差 / 普通残差 / 两倍
    // ========================================================================
    auto ada_residual = [&](ncnn::VkMat& data, const ncnn::VkMat& residual,
                            const ncnn::VkMat& gate_gpu, uint slot, uint mode) -> int
    {
        const int tokens = data.h;
        std::vector<ncnn::VkMat> bindings(4);
        bindings[0] = data;
        bindings[1] = residual;
        bindings[2] = gate_gpu;
        bindings[3] = embedding;
        const uint32_t total = static_cast<uint32_t>(tokens) * dim;
        std::vector<ncnn::vk_constant_type> constants(4);
        constants[0].u32 = static_cast<uint32_t>(dim);
        constants[1].u32 = slot;
        constants[2].u32 = mode;
        constants[3].u32 = total;
        ncnn::VkMat dispatcher;
        dispatcher.w = total;
        dispatcher.h = 1;
        dispatcher.c = 1;
        cmd.record_pipeline(pipeline_ada_residual, bindings, constants, dispatcher);
        return 0;
    };

    // ========================================================================
    // 辅助 lambda：apply_mlp —— SwiGLU MLP（gate_proj + in_proj + silu + out_proj）
    // ========================================================================
    auto apply_mlp_gpu = [&](const ncnn::VkMat& input, const BranchWeights& branch,
                             ncnn::VkMat& output) -> int
    {
        ncnn::VkMat gate;
        ncnn::VkMat value;
        if (branch.mlp_gate_proj->forward(input, gate, cmd, opt) != 0
            || branch.mlp_in_proj->forward(input, value, cmd, opt) != 0)
            return -100;
        {
            std::vector<ncnn::VkMat> bindings(2);
            bindings[0] = gate;
            bindings[1] = value;
            const uint32_t total = static_cast<uint32_t>(gate.h) * mlp_hidden;
            std::vector<ncnn::vk_constant_type> constants(1);
            constants[0].u32 = total;
            ncnn::VkMat dispatcher;
            dispatcher.w = total;
            dispatcher.h = 1;
            dispatcher.c = 1;
            cmd.record_pipeline(pipeline_silu, bindings, constants, dispatcher);
        }
        return branch.mlp_out_proj->forward(gate, output, cmd, opt);
    };

    // ========================================================================
    // 阶段一：attention 输入准备（rmsnorm + adaLN 调制）
    // ========================================================================
    ncnn::VkMat vid_attn_in;
    ncnn::VkMat txt_attn_in;
    if (rmsnorm_ada(vid, vid_weights.attn_scale_gpu, vid_weights.attn_shift_gpu, 0, 1, vid_attn_in) != 0)
        return -100;
    if (rmsnorm_ada(txt, text_branch.attn_scale_gpu, text_branch.attn_shift_gpu, 0, last_layer ? 0 : 1, txt_attn_in) != 0)
        return -100;

    // ========================================================================
    // 阶段二：QKV 投影（复用 InnerProduct）
    // ========================================================================
    ncnn::VkMat vid_qkv;
    ncnn::VkMat txt_qkv;
    if (vid_weights.qkv->forward(vid_attn_in, vid_qkv, cmd, opt) != 0
        || text_branch.qkv->forward(txt_attn_in, txt_qkv, cmd, opt) != 0)
        return -100;

    // ========================================================================
    // 阶段三：attention（逐窗口 dispatch qkv_prepare + attention）
    // ========================================================================
    // 跨窗口归约的 workspace：q/k/v 复用（每窗口覆盖），vid/txt 输出按窗口分段。
    ncnn::VkMat q_ws(heads * max_seq * head_dim, 4u, 1, opt.workspace_vkallocator);
    ncnn::VkMat k_ws(heads * max_seq * head_dim, 4u, 1, opt.workspace_vkallocator);
    ncnn::VkMat v_ws(heads * max_seq * head_dim, 4u, 1, opt.workspace_vkallocator);
    ncnn::VkMat vid_out_ws(window_count * video_tokens * dim, 4u, 1, opt.workspace_vkallocator);
    ncnn::VkMat txt_out_ws(window_count * text_length * dim, 4u, 1, opt.workspace_vkallocator);
    if (q_ws.empty() || k_ws.empty() || v_ws.empty() || vid_out_ws.empty() || txt_out_ws.empty())
        return -100;

    // 每个窗口段只覆盖自己的视频 token；其余位置参与 window_sum 前必须为零。
    // 跨 Net 共享 allocator 后显存会稳定复用，因此不能依赖新分配内存恰好为零。
    {
        std::vector<ncnn::VkMat> bindings(1);
        bindings[0] = vid_out_ws;
        const uint32_t total = static_cast<uint32_t>(window_count) * video_tokens * dim;
        std::vector<ncnn::vk_constant_type> constants(1);
        constants[0].u32 = total;
        ncnn::VkMat dispatcher;
        dispatcher.w = total;
        dispatcher.h = 1;
        dispatcher.c = 1;
        cmd.record_pipeline(pipeline_zero_buffer, bindings, constants, dispatcher);
    }

    const float attention_scale = 1.f / std::sqrt(static_cast<float>(head_dim));
    for (int wi = 0; wi < window_count; wi++)
    {
        const Window& w = windows[wi];
        const int local_h = w.h1 - w.h0;
        const int local_w = w.w1 - w.w0;
        const int video_length = (w.t1 - w.t0) * local_h * local_w;
        const int seq = video_length + text_length;

        // qkv_prepare：算窗口内 seq 个 token 的 q/k/v（MM-RoPE + bf16）。
        {
            std::vector<ncnn::VkMat> bindings(10);
            bindings[0] = vid_qkv;
            bindings[1] = txt_qkv;
            bindings[2] = q_ws;
            bindings[3] = k_ws;
            bindings[4] = v_ws;
            bindings[5] = vid_weights.norm_q_gpu;
            bindings[6] = vid_weights.norm_k_gpu;
            bindings[7] = rope_freqs_gpu;
            bindings[8] = text_branch.norm_q_gpu;
            bindings[9] = text_branch.norm_k_gpu;

            const uint32_t total = static_cast<uint32_t>(heads) * seq;
            std::vector<ncnn::vk_constant_type> constants(15);
            constants[0].u32 = static_cast<uint32_t>(heads);
            constants[1].u32 = static_cast<uint32_t>(head_dim);
            constants[2].u32 = static_cast<uint32_t>(seq);
            constants[3].u32 = static_cast<uint32_t>(video_length);
            constants[4].u32 = static_cast<uint32_t>(text_length);
            constants[5].u32 = static_cast<uint32_t>(local_h);
            constants[6].u32 = static_cast<uint32_t>(local_w);
            constants[7].u32 = static_cast<uint32_t>(w.h0);
            constants[8].u32 = static_cast<uint32_t>(w.w0);
            constants[9].u32 = static_cast<uint32_t>(w.t0);
            constants[10].u32 = static_cast<uint32_t>(height);
            constants[11].u32 = static_cast<uint32_t>(width);
            constants[12].u32 = static_cast<uint32_t>(dim);
            constants[13].f = norm_eps;
            constants[14].u32 = total;

            ncnn::VkMat dispatcher;
            dispatcher.w = total;
            dispatcher.h = 1;
            dispatcher.c = 1;
            cmd.record_pipeline(pipeline_qkv_prepare, bindings, constants, dispatcher);
        }

        // attention：softmax(QK^T * scale) @ V，写当前窗口段。
        {
            std::vector<ncnn::VkMat> bindings(5);
            bindings[0] = q_ws;
            bindings[1] = k_ws;
            bindings[2] = v_ws;
            bindings[3] = vid_out_ws;
            bindings[4] = txt_out_ws;

            const uint32_t total = static_cast<uint32_t>(heads) * seq;
            std::vector<ncnn::vk_constant_type> constants(16);
            constants[0].u32 = static_cast<uint32_t>(head_dim);
            constants[1].u32 = static_cast<uint32_t>(seq);
            constants[2].u32 = static_cast<uint32_t>(video_length);
            constants[3].u32 = static_cast<uint32_t>(text_length);
            constants[4].u32 = static_cast<uint32_t>(video_tokens);
            constants[5].u32 = static_cast<uint32_t>(local_h);
            constants[6].u32 = static_cast<uint32_t>(local_w);
            constants[7].u32 = static_cast<uint32_t>(w.h0);
            constants[8].u32 = static_cast<uint32_t>(w.w0);
            constants[9].u32 = static_cast<uint32_t>(w.t0);
            constants[10].u32 = static_cast<uint32_t>(height);
            constants[11].u32 = static_cast<uint32_t>(width);
            constants[12].u32 = static_cast<uint32_t>(dim);
            constants[13].u32 = static_cast<uint32_t>(wi);
            constants[14].f = attention_scale;
            constants[15].u32 = total;

            ncnn::VkMat dispatcher;
            dispatcher.w = total;
            dispatcher.h = 1;
            dispatcher.c = 1;
            cmd.record_pipeline(pipeline_attention, bindings, constants, dispatcher);
        }
    }

    // 跨窗口归约：视频累加（不平均），文本平均。
    ncnn::VkMat vid_attention;
    ncnn::VkMat txt_attention;
    vid_attention.create(dim, video_tokens, 4u, 1, opt.workspace_vkallocator);
    txt_attention.create(dim, text_length, 4u, 1, opt.workspace_vkallocator);
    if (vid_attention.empty() || txt_attention.empty())
        return -100;
    auto window_sum = [&](const ncnn::VkMat& src, ncnn::VkMat& dst, int tokens, uint has_divide) {
        std::vector<ncnn::VkMat> bindings(2);
        bindings[0] = src;
        bindings[1] = dst;
        const uint32_t total = static_cast<uint32_t>(tokens) * dim;
        std::vector<ncnn::vk_constant_type> constants(5);
        constants[0].u32 = static_cast<uint32_t>(dim);
        constants[1].u32 = static_cast<uint32_t>(tokens);
        constants[2].u32 = static_cast<uint32_t>(window_count);
        constants[3].u32 = has_divide;
        constants[4].u32 = total;
        ncnn::VkMat dispatcher;
        dispatcher.w = total;
        dispatcher.h = 1;
        dispatcher.c = 1;
        cmd.record_pipeline(pipeline_text_normalize, bindings, constants, dispatcher);
    };
    window_sum(vid_out_ws, vid_attention, video_tokens, 0);
    window_sum(txt_out_ws, txt_attention, text_length, 1);

    // ========================================================================
    // 阶段四：输出投影 + 门控残差
    // ========================================================================
    ncnn::VkMat vid_proj;
    ncnn::VkMat txt_proj;
    if (vid_weights.proj_out->forward(vid_attention, vid_proj, cmd, opt) != 0
        || text_branch.proj_out->forward(txt_attention, txt_proj, cmd, opt) != 0)
        return -100;
    if (ada_residual(vid_proj, vid, vid_weights.attn_gate_gpu, 2, 0) != 0)
        return -100;
    if (last_layer)
    {
        if (ada_residual(txt_proj, txt, text_branch.attn_gate_gpu, 2, 1) != 0)
            return -100;
    }
    else
    {
        if (ada_residual(txt_proj, txt, text_branch.attn_gate_gpu, 2, 0) != 0)
            return -100;
    }

    // ========================================================================
    // 阶段五：视频 MLP
    // ========================================================================
    ncnn::VkMat vid_mlp_in;
    if (rmsnorm_ada(vid_proj, vid_weights.mlp_scale_gpu, vid_weights.mlp_shift_gpu, 3, 1, vid_mlp_in) != 0)
        return -100;
    ncnn::VkMat vid_mlp;
    if (apply_mlp_gpu(vid_mlp_in, vid_weights, vid_mlp) != 0)
        return -100;
    if (ada_residual(vid_mlp, vid_proj, vid_weights.mlp_gate_gpu, 5, 0) != 0)
        return -100;
    top_blobs[0] = vid_mlp;

    // ========================================================================
    // 阶段六：文本 MLP（last_layer 时 txt_out = 2 * txt_proj）
    // ========================================================================
    if (last_layer)
    {
        ncnn::VkMat txt_last;
        txt_last.create(dim, text_length, 4u, 1, opt.blob_vkallocator);
        if (txt_last.empty())
            return -100;
        // 复用 window_sum 无法表达「两倍」，用 ada_residual mode=2（不读 residual/gate/emb）。
        {
            std::vector<ncnn::VkMat> bindings(4);
            bindings[0] = txt_proj;
            bindings[1] = txt_proj;   // 占位（mode=2 不读）
            bindings[2] = text_branch.mlp_gate_gpu;  // 占位
            bindings[3] = embedding;  // 占位
            const uint32_t total = static_cast<uint32_t>(text_length) * dim;
            std::vector<ncnn::vk_constant_type> constants(4);
            constants[0].u32 = static_cast<uint32_t>(dim);
            constants[1].u32 = 0;
            constants[2].u32 = 2;   // mode=2：两倍
            constants[3].u32 = total;
            ncnn::VkMat dispatcher;
            dispatcher.w = total;
            dispatcher.h = 1;
            dispatcher.c = 1;
            cmd.record_pipeline(pipeline_ada_residual, bindings, constants, dispatcher);
        }
        top_blobs[1] = txt_proj;
        return 0;
    }

    ncnn::VkMat txt_mlp_in;
    if (rmsnorm_ada(txt_proj, text_branch.mlp_scale_gpu, text_branch.mlp_shift_gpu, 3, 1, txt_mlp_in) != 0)
        return -100;
    ncnn::VkMat txt_mlp;
    if (apply_mlp_gpu(txt_mlp_in, text_branch, txt_mlp) != 0)
        return -100;
    if (ada_residual(txt_mlp, txt_proj, text_branch.mlp_gate_gpu, 5, 0) != 0)
        return -100;
    top_blobs[1] = txt_mlp;
    return 0;
}

#endif // NCNN_VULKAN

DEFINE_LAYER_CREATOR(SeedVR2DiTBlock)
