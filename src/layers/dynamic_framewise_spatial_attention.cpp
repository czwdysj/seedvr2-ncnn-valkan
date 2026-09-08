// 本文件实现 SeedVR2 VAE bottleneck 的动态逐帧空间自注意力层。
// 它是正确性优先的 CPU 版本：全部使用 FP32，并在点积和 softmax 分母处用 double
// 累加，目的是先和 PyTorch reference 对齐；后续 Vulkan 版本复用同一套输入输出测试。
//
// 除 CPU forward 外，本文件还提供 Vulkan 计算路径（forward 的 VkMat 重载）。
// 注意力是「先归约(GroupNorm 统计) → 再归一化 → 再矩阵乘(Q/K/V 投影) →
// 再 softmax(QK^T)@V → 再输出投影 + 残差」的多段式算子，拆成五个 compute shader：
//   norm_reduce      —— 树形归约每个 (帧, 组) 的 sum/sq_sum
//   norm_apply       —— 逐元素 GroupNorm 归一化，产出 normalized workspace
//   qkv_projection   —— Q/K/V 三个线性投影（同一 shader 三次 dispatch，复用 pipeline）
//   attention        —— softmax(QK^T / sqrt(C)) @ V，每个 work item 一个 query token
//   output_projection —— 输出投影 + 残差连接，写回 top_blob
//
// 中间激活（normalized/q/k/v/attended）与统计量统一用 fp32 workspace 暂存，
// 保证「正确性优先」阶段的精度；权重/输入/输出用 ncnn 注入的 sfp 类型 + buffer_ld1/st1
// 宏读写（精度自适应）。
#include "layers/dynamic_framewise_spatial_attention.h"

#include <algorithm>
#include <cmath>
#include <vector>

#if NCNN_VULKAN
#include <gpu.h>
#endif

DynamicFramewiseSpatialAttention::DynamicFramewiseSpatialAttention()
    : channels(0)
{
    one_blob_only = true;
    support_inplace = false;
    support_packing = false;
#if NCNN_VULKAN
    // 标量布局（elempack=1），只声明支持 Vulkan、不支持 Vulkan packing。
    support_vulkan = true;
    support_vulkan_packing = false;
    pipeline_norm_reduce = 0;
    pipeline_norm_apply = 0;
    pipeline_qkv_projection = 0;
    pipeline_attention = 0;
    pipeline_output_projection = 0;
#endif
}

DynamicFramewiseSpatialAttention::~DynamicFramewiseSpatialAttention()
{
    // pipeline 由 create_pipeline 分配、destroy_pipeline 释放；析构只兜底。
}

int DynamicFramewiseSpatialAttention::load_param(const ncnn::ParamDict& pd)
{
    // 参数 10 保存 norm bias 的 shape，用来恢复通道数。
    // 这里要求 channels 能被 32 整除，因为原始 VAE GroupNorm 使用 32 组。
    const ncnn::Mat shape = pd.get(10, ncnn::Mat());
    if (shape.empty())
        return -1;
    channels = static_cast<const int*>(shape)[0];
    return channels > 0 && channels % 32 == 0 ? 0 : -1;
}

int DynamicFramewiseSpatialAttention::load_model(const ncnn::ModelBin& mb)
{
    const int matrix_size = channels * channels;
    // 导出脚本按固定顺序写入这些权重；顺序错位会导致数值完全失真，
    // 所以这里保持显式逐项加载，便于和 pnnx bin 检查对应。
    norm_bias = mb.load(channels, 1);
    norm_weight = mb.load(channels, 1);
    k_bias = mb.load(channels, 1);
    k_weight = mb.load(matrix_size, 1);
    out_bias = mb.load(channels, 1);
    out_weight = mb.load(matrix_size, 1);
    q_bias = mb.load(channels, 1);
    q_weight = mb.load(matrix_size, 1);
    v_bias = mb.load(channels, 1);
    v_weight = mb.load(matrix_size, 1);
    return v_weight.empty() ? -100 : 0;
}

static void linear(
    const std::vector<float>& input,
    std::vector<float>& output,
    const ncnn::Mat& weight_mat,
    const ncnn::Mat& bias_mat,
    int tokens,
    int channels)
{
    const float* weight = weight_mat;
    const float* bias = bias_mat;
#pragma omp parallel for
    for (int token = 0; token < tokens; token++)
    {
        // 权重按 [out_channel, in_channel] 存储，对每个空间 token 独立做全连接。
        const float* input_row = input.data() + static_cast<size_t>(token) * channels;
        float* output_row = output.data() + static_cast<size_t>(token) * channels;
        for (int out_channel = 0; out_channel < channels; out_channel++)
        {
            const float* weight_row = weight + static_cast<size_t>(out_channel) * channels;
            double value = bias[out_channel];
            for (int in_channel = 0; in_channel < channels; in_channel++)
                value += static_cast<double>(input_row[in_channel]) * weight_row[in_channel];
            output_row[out_channel] = static_cast<float>(value);
        }
    }
}

int DynamicFramewiseSpatialAttention::forward(
    const ncnn::Mat& bottom_blob,
    ncnn::Mat& top_blob,
    const ncnn::Option& opt) const
{
    if (bottom_blob.dims != 4 || bottom_blob.c != channels || bottom_blob.elempack != 1)
        return -1;

    const int width = bottom_blob.w;
    const int height = bottom_blob.h;
    const int frames = bottom_blob.d;
    const int tokens = width * height;
    const int channels_per_group = channels / 32;
    const float attention_scale = 1.f / std::sqrt(static_cast<float>(channels));
    top_blob.create(width, height, frames, channels, 4u, 1, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    for (int frame = 0; frame < frames; frame++)
    {
        // 每一帧单独展开为 [H*W, C] token；VAE 这里做空间注意力，不跨时间帧。
        std::vector<float> normalized(static_cast<size_t>(tokens) * channels);
        for (int group = 0; group < 32; group++)
        {
            // GroupNorm 的统计范围是当前 frame、当前 group 内的所有 H*W 位置。
            double sum = 0.0;
            double square_sum = 0.0;
            for (int local_channel = 0; local_channel < channels_per_group; local_channel++)
            {
                const int channel = group * channels_per_group + local_channel;
                const float* input = bottom_blob.channel(channel).depth(frame);
                for (int token = 0; token < tokens; token++)
                {
                    sum += input[token];
                    square_sum += static_cast<double>(input[token]) * input[token];
                }
            }
            const int count = channels_per_group * tokens;
            const float mean = static_cast<float>(sum / count);
            const float variance = static_cast<float>(square_sum / count - static_cast<double>(mean) * mean);
            const float inverse_std = 1.f / std::sqrt(std::max(variance, 0.f) + 1e-6f);
            for (int local_channel = 0; local_channel < channels_per_group; local_channel++)
            {
                const int channel = group * channels_per_group + local_channel;
                const float* input = bottom_blob.channel(channel).depth(frame);
                const float scale = static_cast<const float*>(norm_weight)[channel] * inverse_std;
                const float offset = static_cast<const float*>(norm_bias)[channel] - mean * scale;
                for (int token = 0; token < tokens; token++)
                    normalized[static_cast<size_t>(token) * channels + channel] = input[token] * scale + offset;
            }
        }

        std::vector<float> query(normalized.size());
        std::vector<float> key(normalized.size());
        std::vector<float> value(normalized.size());
        linear(normalized, query, q_weight, q_bias, tokens, channels);
        linear(normalized, key, k_weight, k_bias, tokens, channels);
        linear(normalized, value, v_weight, v_bias, tokens, channels);

        std::vector<float> attended(normalized.size(), 0.f);
#pragma omp parallel for num_threads(opt.num_threads)
        for (int query_token = 0; query_token < tokens; query_token++)
        {
            // 标准单头空间注意力：softmax(QK^T / sqrt(C)) @ V。
            // 这里先保留直观实现，方便逐 token 对齐；性能优化留给后续 Vulkan kernel。
            std::vector<float> scores(tokens);
            float maximum = -INFINITY;
            for (int key_token = 0; key_token < tokens; key_token++)
            {
                double score = 0.0;
                for (int channel = 0; channel < channels; channel++)
                    score += static_cast<double>(query[static_cast<size_t>(query_token) * channels + channel])
                             * key[static_cast<size_t>(key_token) * channels + channel];
                scores[key_token] = static_cast<float>(score) * attention_scale;
                maximum = std::max(maximum, scores[key_token]);
            }
            double denominator = 0.0;
            for (float& score : scores)
            {
                score = std::exp(score - maximum);
                denominator += score;
            }
            for (int key_token = 0; key_token < tokens; key_token++)
            {
                const float probability = static_cast<float>(scores[key_token] / denominator);
                const float* value_row = value.data() + static_cast<size_t>(key_token) * channels;
                float* output_row = attended.data() + static_cast<size_t>(query_token) * channels;
                for (int channel = 0; channel < channels; channel++)
                    output_row[channel] += probability * value_row[channel];
            }
        }

        std::vector<float> projected(attended.size());
        linear(attended, projected, out_weight, out_bias, tokens, channels);
        for (int channel = 0; channel < channels; channel++)
        {
            // 原始 PyTorch attention block 带残差连接，投影结果需要加回输入。
            const float* residual = bottom_blob.channel(channel).depth(frame);
            float* output = top_blob.channel(channel).depth(frame);
            for (int token = 0; token < tokens; token++)
                output[token] = projected[static_cast<size_t>(token) * channels + channel] + residual[token];
        }
    }
    return 0;
}

#if NCNN_VULKAN

// ============================================================================
// shader 1：norm_reduce —— 统计每个 (帧, 组) 的 sum/sq_sum
// ============================================================================
// 与 DynamicFramewiseGroupNorm 的 reduce 完全一致：每个 workgroup 处理一个
// (帧, 组)，strided 累加 + shared-memory 树形归约。groups 固定 32。
static const char* attention_norm_reduce_shader_source = R"VKGLSL(
#version 450

layout(binding = 0, std430) readonly buffer bottom_data { sfp bottom_blob[]; };
layout(binding = 1, std430) writeonly buffer sum_data { float sum_blob[]; };
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
    uint gid = gl_WorkGroupID.x;
    uint tid = gl_LocalInvocationID.x;
    uint nthreads = gl_WorkGroupSize.x;

    uint frame = gid / p.groups;
    uint group = gid % p.groups;
    uint begin_channel = group * p.channels_per_group;

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
)VKGLSL";

// ============================================================================
// shader 2：norm_apply —— 逐元素 GroupNorm 归一化
// ============================================================================
// 每个 work item 一个 (帧, token, 通道)，读 sum/sq_sum 算 mean/var/inverse_std，
// 输出 token-major 的 normalized workspace（后续投影按 token 内积）。
static const char* attention_norm_apply_shader_source = R"VKGLSL(
#version 450

layout(constant_id = 0) const uint cpg = 1u;

layout(binding = 0, std430) readonly buffer bottom_data { sfp bottom_blob[]; };
layout(binding = 1, std430) readonly buffer sum_data { float sum_blob[]; };
layout(binding = 2, std430) readonly buffer sqsum_data { float sqsum_blob[]; };
layout(binding = 3, std430) readonly buffer norm_weight_data { sfp norm_weight_blob[]; };
layout(binding = 4, std430) readonly buffer norm_bias_data { sfp norm_bias_blob[]; };
layout(binding = 5, std430) writeonly buffer normalized_data { float normalized_blob[]; };

layout(push_constant) uniform parameter {
    uint frames;
    uint tokens;
    uint cstep;
    uint groups;
    uint count;
    float eps;
    uint total;
} p;

void main()
{
    uint gi = gl_GlobalInvocationID.x;
    if (gi >= p.total) return;

    // 解码：frame 最外，token 次之，c 最内（token-major 输出布局）。
    uint frame = gi / (p.tokens * p.groups * cpg);
    uint token = (gi / (p.groups * cpg)) % p.tokens;
    uint c = gi % (p.groups * cpg);

    uint group = c / cpg;
    uint fg = frame * p.groups + group;
    float mean = sum_blob[fg] / float(p.count);
    float var = sqsum_blob[fg] / float(p.count) - mean * mean;
    float inv = 1.0 / sqrt(max(var, 0.0) + p.eps);

    float scale = buffer_ld1(norm_weight_blob, c) * inv;
    float offset = buffer_ld1(norm_bias_blob, c) - mean * scale;

    // bottom 是 channel-major（c 最外层、cstep 对齐）。
    uint addr = c * p.cstep + frame * p.tokens + token;
    afp v = buffer_ld1(bottom_blob, addr);
    normalized_blob[gi] = v * scale + offset;
}
)VKGLSL";

// ============================================================================
// shader 3：qkv_projection —— Q/K/V 线性投影（同一 shader 三次 dispatch）
// ============================================================================
// 每个 work item 一个 (帧, token, out_channel)，内积 channels 次。三次 dispatch
// 分别绑定 Q/K/V 的权重/偏置/输出 workspace，复用同一个 pipeline。
static const char* attention_qkv_projection_shader_source = R"VKGLSL(
#version 450

layout(binding = 0, std430) readonly buffer normalized_data { float normalized_blob[]; };
layout(binding = 1, std430) readonly buffer proj_weight_data { sfp proj_weight_blob[]; };
layout(binding = 2, std430) readonly buffer proj_bias_data { sfp proj_bias_blob[]; };
layout(binding = 3, std430) writeonly buffer output_data { float output_blob[]; };

layout(push_constant) uniform parameter {
    uint frames;
    uint tokens;
    uint channels;
    uint total;
} p;

void main()
{
    uint gi = gl_GlobalInvocationID.x;
    if (gi >= p.total) return;

    uint frame = gi / (p.tokens * p.channels);
    uint token = (gi / p.channels) % p.tokens;
    uint out_c = gi % p.channels;

    // 权重按 [out_channel, in_channel] 存储，内积 channels 次。
    afp acc = buffer_ld1(proj_bias_blob, out_c);
    uint in_base = frame * p.tokens * p.channels + token * p.channels;
    uint w_base = out_c * p.channels;
    for (uint c = 0u; c < p.channels; c++)
    {
        acc += buffer_ld1(proj_weight_blob, w_base + c) * normalized_blob[in_base + c];
    }
    output_blob[gi] = acc;
}
)VKGLSL";

// ============================================================================
// shader 4：attention —— softmax(QK^T / sqrt(C)) @ V
// ============================================================================
// 每个 work item 一个 (帧, query_token)。为避免物化 [tokens, tokens] 的 scores
// 矩阵，采用「三遍 QK^T」：第一遍找 max，第二遍算 softmax 分母，第三遍做 PV
// 加权累加。每遍重算一次点积（正确性优先，性能优化方向见文档）。
static const char* attention_attention_shader_source = R"VKGLSL(
#version 450

layout(binding = 0, std430) readonly buffer q_data { float q_blob[]; };
layout(binding = 1, std430) readonly buffer k_data { float k_blob[]; };
layout(binding = 2, std430) readonly buffer v_data { float v_blob[]; };
layout(binding = 3, std430) writeonly buffer attended_data { float attended_blob[]; };

layout(push_constant) uniform parameter {
    uint frames;
    uint tokens;
    uint channels;
    float scale;
    uint total;
} p;

void main()
{
    uint gi = gl_GlobalInvocationID.x;
    if (gi >= p.total) return;

    uint frame = gi / p.tokens;
    uint q_token = gi % p.tokens;
    uint q_base = frame * p.tokens * p.channels + q_token * p.channels;

    // 第一遍：找 max（数值稳定，防 exp 溢出）。
    float max_score = -3.402823e+38f;
    for (uint k = 0u; k < p.tokens; k++)
    {
        float dot = 0.0;
        uint k_base = frame * p.tokens * p.channels + k * p.channels;
        for (uint c = 0u; c < p.channels; c++)
            dot += q_blob[q_base + c] * k_blob[k_base + c];
        dot *= p.scale;
        if (dot > max_score) max_score = dot;
    }

    // 第二遍：算 softmax 分母。
    float sum = 0.0;
    for (uint k = 0u; k < p.tokens; k++)
    {
        float dot = 0.0;
        uint k_base = frame * p.tokens * p.channels + k * p.channels;
        for (uint c = 0u; c < p.channels; c++)
            dot += q_blob[q_base + c] * k_blob[k_base + c];
        sum += exp(dot * p.scale - max_score);
    }

    // 第三遍：PV 加权累加。
    uint out_base = frame * p.tokens * p.channels + q_token * p.channels;
    for (uint c = 0u; c < p.channels; c++)
        attended_blob[out_base + c] = 0.0;
    for (uint k = 0u; k < p.tokens; k++)
    {
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
)VKGLSL";

// ============================================================================
// shader 5：output_projection —— 输出投影 + 残差连接
// ============================================================================
// 每个 work item 一个 (帧, token, out_channel)，内积 channels 次后加回残差
// （bottom 原始值），写回 channel-major 的 top_blob。
static const char* attention_output_projection_shader_source = R"VKGLSL(
#version 450

layout(binding = 0, std430) readonly buffer attended_data { float attended_blob[]; };
layout(binding = 1, std430) readonly buffer out_weight_data { sfp out_weight_blob[]; };
layout(binding = 2, std430) readonly buffer out_bias_data { sfp out_bias_blob[]; };
layout(binding = 3, std430) readonly buffer bottom_data { sfp bottom_blob[]; };
layout(binding = 4, std430) writeonly buffer top_data { sfp top_blob[]; };

layout(push_constant) uniform parameter {
    uint frames;
    uint tokens;
    uint channels;
    uint cstep;
    uint total;
} p;

void main()
{
    uint gi = gl_GlobalInvocationID.x;
    if (gi >= p.total) return;

    uint frame = gi / (p.tokens * p.channels);
    uint token = (gi / p.channels) % p.tokens;
    uint out_c = gi % p.channels;

    afp acc = buffer_ld1(out_bias_blob, out_c);
    uint in_base = frame * p.tokens * p.channels + token * p.channels;
    uint w_base = out_c * p.channels;
    for (uint c = 0u; c < p.channels; c++)
    {
        acc += buffer_ld1(out_weight_blob, w_base + c) * attended_blob[in_base + c];
    }

    // 残差连接：加回 bottom 原始输入，写回 channel-major 的 top。
    uint addr = out_c * p.cstep + frame * p.tokens + token;
    afp residual = buffer_ld1(bottom_blob, addr);
    buffer_st1(top_blob, addr, acc + residual);
}
)VKGLSL";

// 编译一个 shader 源码到 pipeline 的辅助函数，保持五个 pipeline 创建逻辑一致。
static int compile_pipeline(
    ncnn::Pipeline*& pipeline,
    const ncnn::VulkanDevice* vkdev,
    const char* source,
    const ncnn::Option& opt,
    const std::vector<ncnn::vk_specialization_type>& specializations,
    int local_size_x,
    const char* name)
{
    std::vector<uint32_t> spirv;
    const int compiled = ncnn::compile_spirv_module(source, opt, spirv);
    if (compiled != 0)
    {
        std::fprintf(stderr, "DynamicFramewiseSpatialAttention: %s compile failed %d\n", name, compiled);
        return compiled;
    }
    pipeline = new ncnn::Pipeline(vkdev);
    pipeline->set_optimal_local_size_xyz(local_size_x, 1, 1);
    return pipeline->create(spirv.data(), spirv.size() * sizeof(uint32_t), specializations);
}

int DynamicFramewiseSpatialAttention::create_pipeline(const ncnn::Option& opt)
{
    // CPU 模式下 Net 同样会调用 create_pipeline，但此时 vkdev 尚未赋值；
    // 没有 Vulkan 设备就直接返回，避免空指针构造 Pipeline。
    if (!vkdev)
        return 0;

    const int channels_per_group = channels / 32;

    // norm_reduce：无 specialization。
    {
        int ret = compile_pipeline(pipeline_norm_reduce, vkdev, attention_norm_reduce_shader_source,
                                   opt, std::vector<ncnn::vk_specialization_type>(), 256, "norm_reduce");
        if (ret != 0)
            return ret;
    }

    // norm_apply：cpg 烤进 SPIR-V，优化 group = c / cpg 的除法。
    {
        std::vector<ncnn::vk_specialization_type> spec(1);
        spec[0].u32 = static_cast<uint32_t>(channels_per_group);
        int ret = compile_pipeline(pipeline_norm_apply, vkdev, attention_norm_apply_shader_source,
                                   opt, spec, 64, "norm_apply");
        if (ret != 0)
            return ret;
    }

    // qkv_projection：无 specialization。
    {
        int ret = compile_pipeline(pipeline_qkv_projection, vkdev, attention_qkv_projection_shader_source,
                                   opt, std::vector<ncnn::vk_specialization_type>(), 64, "qkv_projection");
        if (ret != 0)
            return ret;
    }

    // attention：无 specialization。
    {
        int ret = compile_pipeline(pipeline_attention, vkdev, attention_attention_shader_source,
                                   opt, std::vector<ncnn::vk_specialization_type>(), 64, "attention");
        if (ret != 0)
            return ret;
    }

    // output_projection：无 specialization。
    {
        int ret = compile_pipeline(pipeline_output_projection, vkdev, attention_output_projection_shader_source,
                                   opt, std::vector<ncnn::vk_specialization_type>(), 64, "output_projection");
        if (ret != 0)
            return ret;
    }

    return 0;
}

int DynamicFramewiseSpatialAttention::destroy_pipeline(const ncnn::Option& /*opt*/)
{
    delete pipeline_norm_reduce;
    pipeline_norm_reduce = 0;
    delete pipeline_norm_apply;
    pipeline_norm_apply = 0;
    delete pipeline_qkv_projection;
    pipeline_qkv_projection = 0;
    delete pipeline_attention;
    pipeline_attention = 0;
    delete pipeline_output_projection;
    pipeline_output_projection = 0;
    return 0;
}

int DynamicFramewiseSpatialAttention::upload_model(ncnn::VkTransfer& cmd, const ncnn::Option& opt)
{
    // 全部权重/偏置在推理前一次性上传 GPU，之后 forward 只引用 buffer，零搬运。
    cmd.record_upload(norm_weight, norm_weight_gpu, opt);
    cmd.record_upload(norm_bias, norm_bias_gpu, opt);
    cmd.record_upload(q_weight, q_weight_gpu, opt);
    cmd.record_upload(q_bias, q_bias_gpu, opt);
    cmd.record_upload(k_weight, k_weight_gpu, opt);
    cmd.record_upload(k_bias, k_bias_gpu, opt);
    cmd.record_upload(v_weight, v_weight_gpu, opt);
    cmd.record_upload(v_bias, v_bias_gpu, opt);
    cmd.record_upload(out_weight, out_weight_gpu, opt);
    cmd.record_upload(out_bias, out_bias_gpu, opt);
    return 0;
}

int DynamicFramewiseSpatialAttention::forward(
    const ncnn::VkMat& bottom_blob,
    ncnn::VkMat& top_blob,
    ncnn::VkCompute& cmd,
    const ncnn::Option& opt) const
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int frames = bottom_blob.d;
    const int cstep = static_cast<int>(bottom_blob.cstep);
    const int tokens = w * h;
    const int channels_per_group = channels / 32;
    const int count = channels_per_group * tokens;
    const int stat_count = frames * 32; // 每个 (帧, 组) 一对统计量
    const float attention_scale = 1.f / std::sqrt(static_cast<float>(channels));
    const float eps = 1e-6f;

    top_blob.create(w, h, frames, channels, 4u, 1, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    // 统计量 + 中间激活 workspace，全部 fp32（4u），用 workspace allocator。
    ncnn::VkMat sum_workspace(stat_count, 4u, 1, opt.workspace_vkallocator);
    ncnn::VkMat sqsum_workspace(stat_count, 4u, 1, opt.workspace_vkallocator);
    ncnn::VkMat normalized_workspace(static_cast<size_t>(frames) * tokens * channels, 4u, 1, opt.workspace_vkallocator);
    ncnn::VkMat q_workspace(static_cast<size_t>(frames) * tokens * channels, 4u, 1, opt.workspace_vkallocator);
    ncnn::VkMat k_workspace(static_cast<size_t>(frames) * tokens * channels, 4u, 1, opt.workspace_vkallocator);
    ncnn::VkMat v_workspace(static_cast<size_t>(frames) * tokens * channels, 4u, 1, opt.workspace_vkallocator);
    ncnn::VkMat attended_workspace(static_cast<size_t>(frames) * tokens * channels, 4u, 1, opt.workspace_vkallocator);
    if (sum_workspace.empty() || sqsum_workspace.empty() || normalized_workspace.empty()
        || q_workspace.empty() || k_workspace.empty() || v_workspace.empty() || attended_workspace.empty())
        return -100;

    // 阶段一：norm_reduce，统计每个 (帧, 组) 的 sum/sq_sum。
    {
        std::vector<ncnn::VkMat> bindings(3);
        bindings[0] = bottom_blob;
        bindings[1] = sum_workspace;
        bindings[2] = sqsum_workspace;

        std::vector<ncnn::vk_constant_type> constants(5);
        constants[0].u32 = static_cast<uint32_t>(tokens);
        constants[1].u32 = static_cast<uint32_t>(channels_per_group);
        constants[2].u32 = 32;
        constants[3].u32 = static_cast<uint32_t>(cstep);
        constants[4].u32 = static_cast<uint32_t>(count);

        ncnn::VkMat dispatcher;
        dispatcher.w = stat_count * pipeline_norm_reduce->local_size_x();
        dispatcher.h = 1;
        dispatcher.c = 1;

        cmd.record_pipeline(pipeline_norm_reduce, bindings, constants, dispatcher);
    }

    // 阶段二：norm_apply，逐元素归一化到 normalized workspace。
    {
        const uint32_t total = static_cast<uint32_t>(frames) * tokens * channels;
        std::vector<ncnn::VkMat> bindings(6);
        bindings[0] = bottom_blob;
        bindings[1] = sum_workspace;
        bindings[2] = sqsum_workspace;
        bindings[3] = norm_weight_gpu;
        bindings[4] = norm_bias_gpu;
        bindings[5] = normalized_workspace;

        std::vector<ncnn::vk_constant_type> constants(7);
        constants[0].u32 = static_cast<uint32_t>(frames);
        constants[1].u32 = static_cast<uint32_t>(tokens);
        constants[2].u32 = static_cast<uint32_t>(cstep);
        constants[3].u32 = 32;
        constants[4].u32 = static_cast<uint32_t>(count);
        constants[5].f = eps;
        constants[6].u32 = total;

        ncnn::VkMat dispatcher;
        dispatcher.w = total;
        dispatcher.h = 1;
        dispatcher.c = 1;

        cmd.record_pipeline(pipeline_norm_apply, bindings, constants, dispatcher);
    }

    // 阶段三：qkv_projection，三次 dispatch 分别算 Q/K/V。
    {
        const uint32_t total = static_cast<uint32_t>(frames) * tokens * channels;
        std::vector<ncnn::vk_constant_type> constants(4);
        constants[0].u32 = static_cast<uint32_t>(frames);
        constants[1].u32 = static_cast<uint32_t>(tokens);
        constants[2].u32 = static_cast<uint32_t>(channels);
        constants[3].u32 = total;

        ncnn::VkMat dispatcher;
        dispatcher.w = total;
        dispatcher.h = 1;
        dispatcher.c = 1;

        // Q 投影
        {
            std::vector<ncnn::VkMat> bindings(4);
            bindings[0] = normalized_workspace;
            bindings[1] = q_weight_gpu;
            bindings[2] = q_bias_gpu;
            bindings[3] = q_workspace;
            cmd.record_pipeline(pipeline_qkv_projection, bindings, constants, dispatcher);
        }
        // K 投影
        {
            std::vector<ncnn::VkMat> bindings(4);
            bindings[0] = normalized_workspace;
            bindings[1] = k_weight_gpu;
            bindings[2] = k_bias_gpu;
            bindings[3] = k_workspace;
            cmd.record_pipeline(pipeline_qkv_projection, bindings, constants, dispatcher);
        }
        // V 投影
        {
            std::vector<ncnn::VkMat> bindings(4);
            bindings[0] = normalized_workspace;
            bindings[1] = v_weight_gpu;
            bindings[2] = v_bias_gpu;
            bindings[3] = v_workspace;
            cmd.record_pipeline(pipeline_qkv_projection, bindings, constants, dispatcher);
        }
    }

    // 阶段四：attention，softmax(QK^T/sqrt(C))@V。
    {
        const uint32_t total = static_cast<uint32_t>(frames) * tokens;
        std::vector<ncnn::VkMat> bindings(4);
        bindings[0] = q_workspace;
        bindings[1] = k_workspace;
        bindings[2] = v_workspace;
        bindings[3] = attended_workspace;

        std::vector<ncnn::vk_constant_type> constants(5);
        constants[0].u32 = static_cast<uint32_t>(frames);
        constants[1].u32 = static_cast<uint32_t>(tokens);
        constants[2].u32 = static_cast<uint32_t>(channels);
        constants[3].f = attention_scale;
        constants[4].u32 = total;

        ncnn::VkMat dispatcher;
        dispatcher.w = total;
        dispatcher.h = 1;
        dispatcher.c = 1;

        cmd.record_pipeline(pipeline_attention, bindings, constants, dispatcher);
    }

    // 阶段五：output_projection，输出投影 + 残差写回 top。
    {
        const uint32_t total = static_cast<uint32_t>(frames) * tokens * channels;
        std::vector<ncnn::VkMat> bindings(5);
        bindings[0] = attended_workspace;
        bindings[1] = out_weight_gpu;
        bindings[2] = out_bias_gpu;
        bindings[3] = bottom_blob;
        bindings[4] = top_blob;

        std::vector<ncnn::vk_constant_type> constants(5);
        constants[0].u32 = static_cast<uint32_t>(frames);
        constants[1].u32 = static_cast<uint32_t>(tokens);
        constants[2].u32 = static_cast<uint32_t>(channels);
        constants[3].u32 = static_cast<uint32_t>(cstep);
        constants[4].u32 = total;

        ncnn::VkMat dispatcher;
        dispatcher.w = total;
        dispatcher.h = 1;
        dispatcher.c = 1;

        cmd.record_pipeline(pipeline_output_projection, bindings, constants, dispatcher);
    }

    return 0;
}

#endif // NCNN_VULKAN

DEFINE_LAYER_CREATOR(DynamicFramewiseSpatialAttention)
