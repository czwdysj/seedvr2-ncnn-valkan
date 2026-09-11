// 本文件实现 SeedVR2 DiT 动态输出层的 FP32 CPU 数值基线。
// 实现包括 affine RMSNorm、与原 PyTorch Cache 键碰撞结果一致的 output Ada、
// NCNN InnerProduct 投影以及按 (h,w,c) 顺序执行的 2x2 unpatch。
//
// Vulkan 路径复用同一份权重顺序与动态形状约束：三个自定义 shader
// （norm_reduce / norm_apply / unpatchify）负责原生算子表达不了的动态逻辑，
// 输出投影直接复用 InnerProduct 的 Vulkan 实现。整层是 DiTInput 的对称逆操作。
#include "layers/seedvr2_dit_output.h"

#include <cmath>

#if NCNN_VULKAN
#include <gpu.h>
#include <vector>
#endif

namespace
{
ncnn::Layer* load_inner_product(const ncnn::ModelBin& mb, int input_size,
                                int output_size, bool bias,
                                const ncnn::VulkanDevice* vkdev)
{
    ncnn::Layer* layer = ncnn::create_layer("InnerProduct");
    if (!layer)
        return nullptr;
#if NCNN_VULKAN
    // 关键：必须在 load_param 之前设置 vkdev。Layer_final::load_param 里若 vkdev
    // 为空会把 layer_vulkan 删除（fallback 到 CPU），之后就无法走 Vulkan 了。
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
} // namespace

#if NCNN_VULKAN

// ============================================================================
// shader 1：norm_reduce —— 逐 token 平方和归约
// ============================================================================
// RMSNorm 是「先归约、再归一化」的两段式算子。reduce 阶段把每个 token 的
// dim 个元素归约成一个 square_sum，写进 fp32 workspace（统计量用 fp32 而非
// sfp，避免半精度累积误差，与 ncnn 内置 reduce_mean 约定一致）。
//
// 并行策略：每个 workgroup 处理一个 token（gid = token），workgroup 内先按
// strided 方式让每个线程累加自己负责的通道，再用 shared-memory 树形归约把
// 局部和折叠成全局和。树形归约误差 O(log n)，优于串行 O(n)。
//
// 关键索引：输入 video 是 [tokens, dim] 的 2D pack1 张量，VkMat 2D 为
// row-major（行步长 = w = dim），元素地址 = token * dim + channel。
static const char* output_reduce_shader_source = R"VKGLSL(
#version 450

layout(binding = 0, std430) readonly buffer video_data { sfp video_blob[]; };
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

    // 阶段一：每个线程 strided 累加自己负责的通道。
    float local_sqsum = 0.0;
    for (uint c = tid; c < p.dim; c += nthreads)
    {
        float v = buffer_ld1(video_blob, token * p.dim + c);
        local_sqsum += v * v;
    }
    sqsum_shared[tid] = local_sqsum;
    barrier();

    // 阶段二：shared-memory 树形归约（nthreads 为 2 的幂，256 满足）。
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
// shader 2：norm_apply —— affine RMSNorm + output Ada 调制
// ============================================================================
// 对每个 (token, channel) 计算：
//   value     = data[c] * inverse_rms * norm[c]           （RMSNorm）
//   result    = value * (emb[c*6+1] + scale[c]) + emb[c*6] + shift[c]  （Ada）
// 其中 inverse_rms = 1 / sqrt(square_sum[token] / dim + eps)，对同一 token 的
// 所有通道共享。
//
// 合并成一次 FMA 形式（与 CPU 一致）：
//   scale_factor = inverse_rms * norm[c] * (emb[c*6+1] + scale[c])
//   bias_factor  = emb[c*6] + shift[c]
//   result[c]    = data[c] * scale_factor + bias_factor
//
// 关键布局：输入输出全 pack1。normalized 保持 pack1（w=dim）是为了让后续
// InnerProduct 命中 gemm 分支（其条件是 bottom.w == num_input，仅 pack1 满足），
// 从而输出 pack1 的 projected；否则 pack4 输入会走 flatten 分支，产生 2D pack4
// 的「行打包」布局，与逐通道语义不符。每个 work item 计算一个输出标量。
static const char* output_norm_apply_shader_source = R"VKGLSL(
#version 450

layout(binding = 0, std430) readonly buffer video_data { sfp video_blob[]; };
layout(binding = 1, std430) writeonly buffer normalized_data { sfp normalized_blob[]; };
layout(binding = 2, std430) readonly buffer norm_data { sfp norm_blob[]; };
layout(binding = 3, std430) readonly buffer shift_data { sfp shift_blob[]; };
layout(binding = 4, std430) readonly buffer scale_data { sfp scale_blob[]; };
layout(binding = 5, std430) readonly buffer emb_data { sfp emb_blob[]; };
layout(binding = 6, std430) readonly buffer sqsum_data { float sqsum_blob[]; };

layout(push_constant) uniform parameter {
    uint dim;
    float eps;
    uint total;
} p;

void main()
{
    uint gi = gl_GlobalInvocationID.x;
    if (gi >= p.total)
        return;

    uint c = gi % p.dim;        // 通道 [0, dim)
    uint token = gi / p.dim;    // token [0, tokens)

    float inv_rms = 1.0 / sqrt(sqsum_blob[token] / float(p.dim) + p.eps);

    afp v = buffer_ld1(video_blob, token * p.dim + c);
    afp n = buffer_ld1(norm_blob, c);
    afp s = buffer_ld1(scale_blob, c);
    afp sh = buffer_ld1(shift_blob, c);
    afp e0 = buffer_ld1(emb_blob, c * 6u);
    afp e1 = buffer_ld1(emb_blob, c * 6u + 1u);

    afp result = v * (inv_rms * n * (e1 + s)) + (e0 + sh);
    buffer_st1(normalized_blob, token * p.dim + c, result);
}
)VKGLSL";

// ============================================================================
// shader 3：unpatchify —— 2x2 空间反重排
// ============================================================================
// 输入 projected 是 [tokens, output_channels*4] 的 2D pack1 张量（InnerProduct
// gemm 分支输出，每 patch 把 2x2 的 4 个位置按 (dy,dx,c) 顺序拼成 oc*4 通道）。
// 输出是 [output_channels, spatial] 的 2D pack1 张量，每个空间位置 oc 个通道
// 连续。
//
// 每个 work item 计算一个 (output_index, channel)，反推源位置：
//   - output_index 解码为 (t, y_out, x_out)（2 倍分辨率空间坐标）；
//   - y = y_out/2, dy = y_out%2；x = x_out/2, dx = x_out%2；
//   - patch = t*(H*W) + y*W + x；
//   - src（标量索引）= patch*(oc*4) + dy*(2*oc) + dx*oc + channel。
//
// 输入输出全 pack1（projected 由 InnerProduct gemm 分支输出 pack1），标量
// buffer_ld1/st1 读写，无需处理 2D pack4 的行打包布局。
static const char* output_unpatchify_shader_source = R"VKGLSL(
#version 450

layout(binding = 0, std430) readonly buffer projected_data { sfp projected_blob[]; };
layout(binding = 1, std430) writeonly buffer output_data { sfp output_blob[]; };

layout(push_constant) uniform parameter {
    uint height;
    uint width;
    uint output_height;
    uint output_width;
    uint output_channels;
    uint total;
} p;

void main()
{
    uint gi = gl_GlobalInvocationID.x;
    if (gi >= p.total)
        return;

    uint channel = gi % p.output_channels;
    uint output_index = gi / p.output_channels;

    // output_index 解码为 (t, y_out, x_out)。
    uint x_out = output_index % p.output_width;
    uint y_out = (output_index / p.output_width) % p.output_height;
    uint t = output_index / (p.output_width * p.output_height);

    // 反推 patch 网格坐标与 2x2 子像素偏移（patch 是 GLSL 关键字，用 pidx）。
    uint x = x_out / 2u;
    uint dx = x_out % 2u;
    uint y = y_out / 2u;
    uint dy = y_out % 2u;
    uint pidx = t * (p.height * p.width) + y * p.width + x;

    // 源标量索引：patch 内 (dy, dx, channel) 顺序，与 CPU 的 offset 一致。
    uint src = pidx * (p.output_channels * 4u)
        + dy * (p.output_channels * 2u) + dx * p.output_channels + channel;

    afp v = buffer_ld1(projected_blob, src);
    buffer_st1(output_blob, output_index * p.output_channels + channel, v);
}
)VKGLSL";

#endif // NCNN_VULKAN

SeedVR2DiTOutput::SeedVR2DiTOutput()
    : dim(2560), output_channels(16), norm_eps(1e-5f), projection(nullptr),
      runtime_frames(0), runtime_height(0), runtime_width(0), runtime_shape_valid(false)
{
    one_blob_only = false;
    support_inplace = false;
    support_packing = false;
#if NCNN_VULKAN
    support_vulkan = true;
    support_vulkan_packing = false;
    pipeline_norm_reduce = 0;
    pipeline_norm_apply = 0;
    pipeline_unpatchify = 0;
#endif
}

void SeedVR2DiTOutput::set_runtime_shape(int frames, int height, int width)
{
    runtime_frames = frames;
    runtime_height = height;
    runtime_width = width;
    runtime_shape_valid = frames > 0 && height > 0 && width > 0;
}

SeedVR2DiTOutput::~SeedVR2DiTOutput()
{
    delete projection;
}

int SeedVR2DiTOutput::load_param(const ncnn::ParamDict& pd)
{
    dim = pd.get(0, 2560);
    output_channels = pd.get(1, 16);
    norm_eps = pd.get(2, 1e-5f);
    return dim > 0 && output_channels > 0 ? 0 : -1;
}

int SeedVR2DiTOutput::load_model(const ncnn::ModelBin& mb)
{
    norm_weight = mb.load(dim, 1);
    output_shift = mb.load(dim, 1);
    output_scale = mb.load(dim, 1);
    projection = load_inner_product(mb, dim, output_channels * 4, true, vkdev);
    return norm_weight.empty() || output_shift.empty() || output_scale.empty() || !projection
        ? -100
        : 0;
}

int SeedVR2DiTOutput::create_pipeline(const ncnn::Option& opt)
{
    if (!projection)
        return -1;
#if NCNN_VULKAN
    // 关键：Net 只给顶层自定义层设置 vkdev，这里复用 InnerProduct 的子层需要
    // 手动把 vkdev 传递下去，否则 Layer_final 委托的 Vulkan 实现拿不到设备。
    if (vkdev)
        projection->vkdev = vkdev;
#endif
    if (projection->create_pipeline(opt) != 0)
        return -1;

#if NCNN_VULKAN
    // CPU 模式下 vkdev 尚未赋值，跳过三个自定义 shader 的管线创建。
    if (!vkdev)
        return 0;

    {
        std::vector<uint32_t> spirv;
        const int compiled = ncnn::compile_spirv_module(output_reduce_shader_source, opt, spirv);
        if (compiled != 0)
        {
            std::fprintf(stderr, "SeedVR2DiTOutput: norm_reduce compile failed %d\n", compiled);
            return compiled;
        }
        pipeline_norm_reduce = new ncnn::Pipeline(vkdev);
        pipeline_norm_reduce->set_optimal_local_size_xyz(256, 1, 1);
        pipeline_norm_reduce->create(spirv.data(), spirv.size() * sizeof(uint32_t), std::vector<ncnn::vk_specialization_type>());
    }
    {
        std::vector<uint32_t> spirv;
        const int compiled = ncnn::compile_spirv_module(output_norm_apply_shader_source, opt, spirv);
        if (compiled != 0)
        {
            std::fprintf(stderr, "SeedVR2DiTOutput: norm_apply compile failed %d\n", compiled);
            return compiled;
        }
        pipeline_norm_apply = new ncnn::Pipeline(vkdev);
        pipeline_norm_apply->set_optimal_local_size_xyz(64, 1, 1);
        pipeline_norm_apply->create(spirv.data(), spirv.size() * sizeof(uint32_t), std::vector<ncnn::vk_specialization_type>());
    }
    {
        std::vector<uint32_t> spirv;
        const int compiled = ncnn::compile_spirv_module(output_unpatchify_shader_source, opt, spirv);
        if (compiled != 0)
        {
            std::fprintf(stderr, "SeedVR2DiTOutput: unpatchify compile failed %d\n", compiled);
            return compiled;
        }
        pipeline_unpatchify = new ncnn::Pipeline(vkdev);
        pipeline_unpatchify->set_optimal_local_size_xyz(64, 1, 1);
        pipeline_unpatchify->create(spirv.data(), spirv.size() * sizeof(uint32_t), std::vector<ncnn::vk_specialization_type>());
    }
#endif
    return 0;
}

int SeedVR2DiTOutput::destroy_pipeline(const ncnn::Option& opt)
{
    if (projection)
        projection->destroy_pipeline(opt);

#if NCNN_VULKAN
    delete pipeline_norm_reduce;
    pipeline_norm_reduce = 0;
    delete pipeline_norm_apply;
    pipeline_norm_apply = 0;
    delete pipeline_unpatchify;
    pipeline_unpatchify = 0;
#endif
    return 0;
}

int SeedVR2DiTOutput::forward(const std::vector<ncnn::Mat>& bottom_blobs,
                              std::vector<ncnn::Mat>& top_blobs,
                              const ncnn::Option& opt) const
{
    if (bottom_blobs.size() != 3 || top_blobs.size() != 2)
        return -1;
    const ncnn::Mat& video = bottom_blobs[0];
    const ncnn::Mat& embedding = bottom_blobs[1];
    const ncnn::Mat& shape = bottom_blobs[2];
    if (video.dims != 2 || video.w != dim || embedding.w != dim * 6
        || shape.dims != 1 || shape.w != 3)
        return -1;
    const int frames = static_cast<const int*>(shape.data)[0];
    const int height = static_cast<const int*>(shape.data)[1];
    const int width = static_cast<const int*>(shape.data)[2];
    if (frames <= 0 || height <= 0 || width <= 0 || video.h != frames * height * width)
        return -1;

    ncnn::Mat normalized = video.clone(opt.blob_allocator);
    if (normalized.empty())
        return -100;
    const float* norm = norm_weight;
    const float* shift = output_shift;
    const float* scale = output_scale;
    const float* emb = embedding;
#pragma omp parallel for num_threads(opt.num_threads)
    for (int row = 0; row < normalized.h; row++)
    {
        float* data = normalized.row(row);
        double square_sum = 0.0;
        for (int channel = 0; channel < dim; channel++)
            square_sum += static_cast<double>(data[channel]) * data[channel];
        const float inverse_rms = 1.f / std::sqrt(static_cast<float>(square_sum / dim) + norm_eps);
        for (int channel = 0; channel < dim; channel++)
        {
            const float value = data[channel] * inverse_rms * norm[channel];
            // 完整 PyTorch 图复用了 block-0 的 emb_repeat_0_vid cache，故这里
            // 必须按 [dim,2,3] 的第 0 层解释 embedding，而不是按 [5120,1,3]。
            data[channel] = value * (emb[channel * 6 + 1] + scale[channel])
                + emb[channel * 6] + shift[channel];
        }
    }

    ncnn::Mat projected;
    if (projection->forward(normalized, projected, opt) != 0)
        return -100;
    const int output_height = height * 2;
    const int output_width = width * 2;
    ncnn::Mat output(output_channels, frames * output_height * output_width, 4u,
                     opt.blob_allocator);
    if (output.empty())
        return -100;
#pragma omp parallel for num_threads(opt.num_threads)
    for (int patch = 0; patch < projected.h; patch++)
    {
        const int x = patch % width;
        const int y = (patch / width) % height;
        const int t = patch / (width * height);
        const float* source = projected.row(patch);
        int offset = 0;
        for (int dy = 0; dy < 2; dy++)
            for (int dx = 0; dx < 2; dx++)
            {
                const int output_index = (t * output_height + y * 2 + dy) * output_width
                    + x * 2 + dx;
                float* destination = output.row(output_index);
                for (int channel = 0; channel < output_channels; channel++)
                    destination[channel] = source[offset++];
            }
    }

    ncnn::Mat output_shape(3, 4u, opt.blob_allocator);
    if (output_shape.empty())
        return -100;
    int* output_shape_data = output_shape;
    output_shape_data[0] = frames;
    output_shape_data[1] = output_height;
    output_shape_data[2] = output_width;
    top_blobs[0] = output;
    top_blobs[1] = output_shape;
    return 0;
}

#if NCNN_VULKAN

int SeedVR2DiTOutput::upload_model(ncnn::VkTransfer& cmd, const ncnn::Option& opt)
{
    // 三个调制权重一次性上传 GPU；投影权重由 InnerProduct 的 upload_model 上传。
    cmd.record_upload(norm_weight, norm_weight_gpu, opt);
    cmd.record_upload(output_shift, output_shift_gpu, opt);
    cmd.record_upload(output_scale, output_scale_gpu, opt);
    if (!projection)
        return -1;
    const int ret = projection->upload_model(cmd, opt);
    if (ret != 0)
    {
        std::fprintf(stderr, "SeedVR2DiTOutput: projection upload_model failed %d\n", ret);
        return -1;
    }
    return 0;
}

int SeedVR2DiTOutput::forward(const std::vector<ncnn::VkMat>& bottom_blobs,
                              std::vector<ncnn::VkMat>& top_blobs,
                              ncnn::VkCompute& cmd,
                              const ncnn::Option& opt) const
{
    if (bottom_blobs.size() != 3 || top_blobs.size() != 2)
        return -1;
    const ncnn::VkMat& video = bottom_blobs[0];
    const ncnn::VkMat& embedding = bottom_blobs[1];
    const ncnn::VkMat& shape = bottom_blobs[2];
    if (video.dims != 2 || video.w != dim || embedding.w != dim * 6 || shape.w != 3)
        return -1;

    // 输出形状由调度器在 CPU 侧已知并注入；shape blob 仅为 param 兼容保留。
    // 删除此处下载后，output head 可以紧接第 31 个 block 在同一命令流执行。
    if (!runtime_shape_valid)
        return -1;
    const int frames = runtime_frames;
    const int height = runtime_height;
    const int width = runtime_width;
    if (frames <= 0 || height <= 0 || width <= 0 || video.h != frames * height * width)
        return -1;
    const int tokens = frames * height * width;
    const int output_height = height * 2;
    const int output_width = width * 2;
    const int spatial = frames * output_height * output_width;

    // 阶段一：norm_reduce，逐 token 平方和归约（fp32 workspace）。
    ncnn::VkMat sqsum_workspace(tokens, 4u, 1, opt.workspace_vkallocator);
    if (sqsum_workspace.empty())
        return -100;
    {
        std::vector<ncnn::VkMat> bindings(2);
        bindings[0] = video;
        bindings[1] = sqsum_workspace;

        std::vector<ncnn::vk_constant_type> constants(2);
        constants[0].u32 = static_cast<uint32_t>(dim);
        constants[1].u32 = static_cast<uint32_t>(tokens);

        ncnn::VkMat dispatcher;
        // reduce 用 gl_WorkGroupID.x 作为 token 索引，期望 workgroup 数 == tokens。
        // ncnn 的 group_count_x = ceil(dispatcher.w / local_size_x)，因此
        // dispatcher.w 必须等于 tokens × local_size_x，才能启动恰好 tokens 个
        // workgroup（与 GroupNorm reduce 同款语义陷阱）。
        dispatcher.w = tokens * pipeline_norm_reduce->local_size_x();
        dispatcher.h = 1;
        dispatcher.c = 1;

        cmd.record_pipeline(pipeline_norm_reduce, bindings, constants, dispatcher);
    }

    // 阶段二：norm_apply，RMSNorm + Ada，输出 pack1 normalized（喂给 InnerProduct
    // 的 gemm 分支，命中 bottom.w == num_input 条件，输出 pack1 projected）。
    ncnn::VkMat normalized;
    normalized.create(dim, tokens, 4u, 1, opt.workspace_vkallocator);
    if (normalized.empty())
        return -100;
    {
        std::vector<ncnn::VkMat> bindings(7);
        bindings[0] = video;
        bindings[1] = normalized;
        bindings[2] = norm_weight_gpu;
        bindings[3] = output_shift_gpu;
        bindings[4] = output_scale_gpu;
        bindings[5] = embedding;
        bindings[6] = sqsum_workspace;

        const uint32_t total = static_cast<uint32_t>(tokens) * dim;
        std::vector<ncnn::vk_constant_type> constants(3);
        constants[0].u32 = static_cast<uint32_t>(dim);
        constants[1].f = norm_eps;
        constants[2].u32 = total;

        ncnn::VkMat dispatcher;
        dispatcher.w = total;
        dispatcher.h = 1;
        dispatcher.c = 1;

        cmd.record_pipeline(pipeline_norm_apply, bindings, constants, dispatcher);
    }

    // 阶段三：输出投影（复用 InnerProduct 的 Vulkan 实现，pack1 → pack1）。
    ncnn::VkMat projected;
    if (projection->forward(normalized, projected, cmd, opt) != 0)
        return -100;

    // 阶段四：unpatchify，projected [tokens, oc*4] → output [oc, spatial]。
    top_blobs[0].create(output_channels, spatial, 4u, 1, opt.blob_vkallocator);
    if (top_blobs[0].empty())
        return -100;
    {
        std::vector<ncnn::VkMat> bindings(2);
        bindings[0] = projected;
        bindings[1] = top_blobs[0];

        const uint32_t total = static_cast<uint32_t>(spatial) * output_channels;
        std::vector<ncnn::vk_constant_type> constants(6);
        constants[0].u32 = static_cast<uint32_t>(height);
        constants[1].u32 = static_cast<uint32_t>(width);
        constants[2].u32 = static_cast<uint32_t>(output_height);
        constants[3].u32 = static_cast<uint32_t>(output_width);
        constants[4].u32 = static_cast<uint32_t>(output_channels);
        constants[5].u32 = total;

        ncnn::VkMat dispatcher;
        dispatcher.w = total;
        dispatcher.h = 1;
        dispatcher.c = 1;

        cmd.record_pipeline(pipeline_unpatchify, bindings, constants, dispatcher);
    }

    // 阶段五：output_shape 输出（int 位模式，经 record_upload 传上 GPU）。
    ncnn::Mat output_shape_cpu(3, static_cast<size_t>(4u), opt.blob_allocator);
    int* os = output_shape_cpu;
    os[0] = frames;
    os[1] = output_height;
    os[2] = output_width;
    top_blobs[1].create(3, 4u, 1, opt.blob_vkallocator);
    if (top_blobs[1].empty())
        return -100;
    cmd.record_upload(output_shape_cpu, top_blobs[1], opt);

    return 0;
}

#endif // NCNN_VULKAN

DEFINE_LAYER_CREATOR(SeedVR2DiTOutput)
