// 本文件实现 SeedVR2 VAE 的动态逐帧 GroupNorm 自定义层。
// 普通 pnnx/NCNN 图很容易把视频帧数 T 固定进 reshape 或 reduction 中；
// 这里把 GroupNorm 写成 layer，让同一份 param/bin 可以处理运行时变化的 T,H,W。
// 输入输出统一使用 ncnn::Mat(w=W,h=H,d=T,c=C)，和参考张量的 C,T,H,W 逻辑对应。
//
// 本文件同时提供 CPU 与 Vulkan 两条 forward 路径：
//   - CPU 路径（forward 的 Mat 重载）作为数值基线，已与 PyTorch reference 对齐；
//   - Vulkan 路径（forward 的 VkMat 重载）把「逐帧统计 + 归一化」拆成两个 compute
//     shader：reduce 阶段用 shared-memory 树形归约求出每个 (帧, 组) 的 sum 与
//     sq_sum，normalize 阶段每个 work item 计算一个输出元素的仿射变换。
// 两条路径共享 load_param/load_model，保证动态语义与权重完全一致。
#include "dynamic_framewise_group_norm.h"

#include <cmath>

#if NCNN_VULKAN
#include <gpu.h>
#include <vector>
#endif

DynamicFramewiseGroupNorm::DynamicFramewiseGroupNorm()
    : channels(0), groups(32), eps(1e-6f)
{
    one_blob_only = true;
    support_inplace = false;
    support_packing = false;
#if NCNN_VULKAN
    // 标量布局（elempack=1），只声明支持 Vulkan、不支持 Vulkan packing。
    support_vulkan = true;
    support_vulkan_packing = false;
    pipeline_reduce = 0;
    pipeline_normalize = 0;
#endif
}

DynamicFramewiseGroupNorm::~DynamicFramewiseGroupNorm()
{
    // pipeline 由 create_pipeline 分配、destroy_pipeline 释放；析构只兜底。
    // GPU 权重缓冲的引用由 VkMat 析构自带 release。
}

int DynamicFramewiseGroupNorm::load_param(const ncnn::ParamDict& pd)
{
    // 导出脚本把 bias 的 shape 写到参数 10，用它恢复通道数。
    // 这样 layer 不需要在 param 里额外维护一份容易出错的 channels 字段。
    const ncnn::Mat bias_shape = pd.get(10, ncnn::Mat());
    if (bias_shape.empty())
        return -1;

    channels = static_cast<const int*>(bias_shape)[0];
    return channels > 0 && channels % groups == 0 ? 0 : -1;
}

int DynamicFramewiseGroupNorm::load_model(const ncnn::ModelBin& mb)
{
    // pnnx 会按自定义模块 tensor 属性名顺序写入 bin；当前导出约定为 bias 后 weight。
    bias_data = mb.load(channels, 1);
    weight_data = mb.load(channels, 1);
    return bias_data.empty() || weight_data.empty() ? -100 : 0;
}

int DynamicFramewiseGroupNorm::forward(
    const ncnn::Mat& bottom_blob,
    ncnn::Mat& top_blob,
    const ncnn::Option& opt) const
{
    if (bottom_blob.dims != 4 || bottom_blob.c != channels || bottom_blob.elempack != 1)
        return -1;

    const int width = bottom_blob.w;
    const int height = bottom_blob.h;
    const int frames = bottom_blob.d;
    const int channels_per_group = channels / groups;
    const int spatial_size = width * height;
    top_blob.create(width, height, frames, channels, 4u, 1, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const float* weight = weight_data;
    const float* bias = bias_data;
#pragma omp parallel for collapse(2) num_threads(opt.num_threads)
    for (int frame = 0; frame < frames; frame++)
    {
        for (int group = 0; group < groups; group++)
        {
            // PyTorch 的 VAE 这里按「单帧内的 group」统计均值方差，
            // 不能把不同时间帧混在一起，否则动态视频长度下会和 reference 偏离。
            double sum = 0.0;
            double square_sum = 0.0;
            const int begin_channel = group * channels_per_group;
            const int count = channels_per_group * spatial_size;
            for (int local_channel = 0; local_channel < channels_per_group; local_channel++)
            {
                const int channel = begin_channel + local_channel;
                const float* input = bottom_blob.channel(channel).depth(frame);
                for (int index = 0; index < spatial_size; index++)
                {
                    const float value = input[index];
                    sum += value;
                    square_sum += static_cast<double>(value) * value;
                }
            }

            const float mean = static_cast<float>(sum / count);
            const float variance = static_cast<float>(square_sum / count - static_cast<double>(mean) * mean);
            const float inverse_std = 1.f / std::sqrt(std::max(variance, 0.f) + eps);
            for (int local_channel = 0; local_channel < channels_per_group; local_channel++)
            {
                const int channel = begin_channel + local_channel;
                const float* input = bottom_blob.channel(channel).depth(frame);
                float* output = top_blob.channel(channel).depth(frame);
                const float scale = weight[channel] * inverse_std;
                const float offset = bias[channel] - mean * scale;
                for (int index = 0; index < spatial_size; index++)
                    output[index] = input[index] * scale + offset;
            }
        }
    }
    return 0;
}

#if NCNN_VULKAN

// ============================================================================
// reduce shader：统计每个 (帧, 组) 的 sum 与 sq_sum
// ============================================================================
//
// GroupNorm 是「先归约、再归一化」的两段式算子，无法像逐元素映射那样单 shader
// 完成。reduce 阶段把每个 (帧, 组) 的 count = channels_per_group × W×H 个元素
// 归约成一对 (sum, sq_sum)，写进 fp32 workspace（统计量用 fp32 而非 sfp，避免
// 半精度累积误差）。
//
// 并行策略：每个 workgroup 处理一个 (帧, 组)（gid = frame × groups + group），
// workgroup 内先按 strided 方式让每个线程累加自己负责的元素，再用 shared-memory
// 树形归约把局部和折叠成全局和。树形归约的误差是 O(log n)，优于单线程串行的
// O(n)，对大规模 count 的数值稳定性更友好。
//
// 关键索引映射（与 CPU 逐行等价）：
//   - 线性元素 i ∈ [0, count) 解码为 (local_channel = i / spatial_size,
//     spatial_idx = i % spatial_size)；
//   - 真实通道 channel = begin_channel + local_channel；
//   - 地址 = channel × cstep + frame × spatial_size + spatial_idx。
//
// 运行时形状（spatial_size/cpg/groups/frames/cstep/count）走 push constant。
static const char* groupnorm_reduce_shader_source = R"VKGLSL(
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

    // 从全局 workgroup 索引解出 (帧, 组)。
    uint frame = gid / p.groups;
    uint group = gid % p.groups;
    uint begin_channel = group * p.channels_per_group;

    // 阶段一：每个线程 strided 累加自己负责的元素（覆盖 count 远大于线程数的情况）。
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

    // 阶段二：shared-memory 树形归约（要求 nthreads 为 2 的幂，256 满足）。
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
// normalize shader：用 sum/sq_sum 对每个元素做仿射归一化
// ============================================================================
//
// 每个 work item 计算一个输出元素。先由稠密索引 gi 解码出 (x, y, t, c)（ncnn
// 4D 布局：x 最内 → y → t(d) → c 最外），再由 c 反推所属 group，读该 (帧, 组)
// 的 sum/sq_sum 计算 mean/var/inverse_std，最后做 out = x·scale + offset。
//
// 仿射系数重排（与 CPU 一致）：
//   mean = sum / count
//   var  = sq_sum / count − mean²
//   inv  = 1 / √(max(var, 0) + eps)
//   scale  = weight[c] · inv
//   offset = bias[c] − mean · scale
//   out    = x · scale + offset
//
// buffer 一律用 ncnn 注入的 sfp 类型 + buffer_ld1/st1 宏读写（精度自适应），
// 累加用 afp；统计量 buffer 强制 float（fp32）。
static const char* groupnorm_normalize_shader_source = R"VKGLSL(
#version 450

layout(constant_id = 0) const uint cpg = 1u;

layout(binding = 0, std430) readonly buffer bottom_data { sfp bottom_blob[]; };
layout(binding = 1, std430) writeonly buffer top_data { sfp top_blob[]; };
layout(binding = 2, std430) readonly buffer weight_data { sfp weight_blob[]; };
layout(binding = 3, std430) readonly buffer bias_data { sfp bias_blob[]; };
layout(binding = 4, std430) readonly buffer sum_data { float sum_blob[]; };
layout(binding = 5, std430) readonly buffer sqsum_data { float sqsum_blob[]; };

layout(push_constant) uniform parameter {
    uint w;
    uint h;
    uint t;
    uint cstep;
    uint spatial_size;
    uint groups;
    uint count;
    float eps;
    uint total;
} p;

void main()
{
    uint gi = gl_GlobalInvocationID.x;
    if (gi >= p.total)
        return;

    // 稠密索引解码（x 最内 → y → t → c 最外）。
    uint x = gi % p.w;
    uint y = (gi / p.w) % p.h;
    uint t_frame = (gi / (p.w * p.h)) % p.t;
    uint c = gi / (p.w * p.h * p.t);

    // 由通道反推所属组，读该 (帧, 组) 的统计量。
    uint group = c / cpg;
    uint fg = t_frame * p.groups + group;
    float mean = sum_blob[fg] / float(p.count);
    float sqmean = sqsum_blob[fg] / float(p.count);
    float var = sqmean - mean * mean;
    float inv = 1.0 / sqrt(max(var, 0.0) + p.eps);

    // 仿射系数重排。
    float gamma = buffer_ld1(weight_blob, c);
    float beta = buffer_ld1(bias_blob, c);
    float scale = gamma * inv;
    float offset = beta - mean * scale;

    // 地址 = c×cstep + t×spatial_size + y×w + x（c 最外层，cstep 对齐）。
    uint addr = c * p.cstep + t_frame * p.spatial_size + y * p.w + x;
    afp v = buffer_ld1(bottom_blob, addr);
    buffer_st1(top_blob, addr, v * scale + offset);
}
)VKGLSL";

int DynamicFramewiseGroupNorm::create_pipeline(const ncnn::Option& opt)
{
    // CPU 模式下 Net 同样会调用 create_pipeline，但此时 vkdev 尚未赋值；
    // 没有 Vulkan 设备就直接返回，避免空指针构造 Pipeline。
    if (!vkdev)
        return 0;

    const int channels_per_group = channels / groups;

    // reduce pipeline：无 specialization，运行时形状全走 push constant。
    {
        std::vector<uint32_t> spirv;
        const int compiled = ncnn::compile_spirv_module(groupnorm_reduce_shader_source, opt, spirv);
        if (compiled != 0)
        {
            std::fprintf(stderr, "DynamicFramewiseGroupNorm: reduce compile failed %d\n", compiled);
            return compiled;
        }
        pipeline_reduce = new ncnn::Pipeline(vkdev);
        pipeline_reduce->set_optimal_local_size_xyz(256, 1, 1);
        pipeline_reduce->create(spirv.data(), spirv.size() * sizeof(uint32_t), std::vector<ncnn::vk_specialization_type>());
    }

    // normalize pipeline：channels_per_group 用 specialization 烤进 SPIR-V，
    // 让 group = c / cpg 的整数除法在编译期被替换成乘法。
    {
        std::vector<ncnn::vk_specialization_type> specializations(1);
        specializations[0].u32 = static_cast<uint32_t>(channels_per_group);

        std::vector<uint32_t> spirv;
        const int compiled = ncnn::compile_spirv_module(groupnorm_normalize_shader_source, opt, spirv);
        if (compiled != 0)
        {
            std::fprintf(stderr, "DynamicFramewiseGroupNorm: normalize compile failed %d\n", compiled);
            return compiled;
        }
        pipeline_normalize = new ncnn::Pipeline(vkdev);
        pipeline_normalize->set_optimal_local_size_xyz(64, 1, 1);
        pipeline_normalize->create(spirv.data(), spirv.size() * sizeof(uint32_t), specializations);
    }

    return 0;
}

int DynamicFramewiseGroupNorm::destroy_pipeline(const ncnn::Option& /*opt*/)
{
    delete pipeline_reduce;
    pipeline_reduce = 0;
    delete pipeline_normalize;
    pipeline_normalize = 0;
    return 0;
}

int DynamicFramewiseGroupNorm::upload_model(ncnn::VkTransfer& cmd, const ncnn::Option& opt)
{
    // 仿射权重/偏置在推理前一次性上传 GPU，之后 forward 只引用 buffer，零搬运。
    cmd.record_upload(weight_data, weight_data_gpu, opt);
    cmd.record_upload(bias_data, bias_data_gpu, opt);
    return 0;
}

int DynamicFramewiseGroupNorm::forward(
    const ncnn::VkMat& bottom_blob,
    ncnn::VkMat& top_blob,
    ncnn::VkCompute& cmd,
    const ncnn::Option& opt) const
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int t = bottom_blob.d;
    const int cstep = static_cast<int>(bottom_blob.cstep);
    const int spatial_size = w * h;
    const int channels_per_group = channels / groups;
    const int count = channels_per_group * spatial_size;
    const int stat_count = t * groups; // 每个 (帧, 组) 一对统计量

    top_blob.create(w, h, t, channels, 4u, 1, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    // 统计量 workspace 用 fp32（4u），因为 sum/sq_sum 需要足够精度；
    // 使用 workspace allocator，层结束后由框架统一回收。
    ncnn::VkMat sum_workspace(stat_count, 4u, 1, opt.workspace_vkallocator);
    ncnn::VkMat sqsum_workspace(stat_count, 4u, 1, opt.workspace_vkallocator);
    if (sum_workspace.empty() || sqsum_workspace.empty())
        return -100;

    // 阶段一：reduce，统计每个 (帧, 组) 的 sum 与 sq_sum。
    {
        std::vector<ncnn::VkMat> bindings(3);
        bindings[0] = bottom_blob;
        bindings[1] = sum_workspace;
        bindings[2] = sqsum_workspace;

        std::vector<ncnn::vk_constant_type> constants(5);
        constants[0].u32 = static_cast<uint32_t>(spatial_size);
        constants[1].u32 = static_cast<uint32_t>(channels_per_group);
        constants[2].u32 = static_cast<uint32_t>(groups);
        constants[3].u32 = static_cast<uint32_t>(cstep);
        constants[4].u32 = static_cast<uint32_t>(count);

        ncnn::VkMat dispatcher;
        // reduce 用 gl_WorkGroupID.x 作为 (帧, 组) 索引，期望 workgroup 数 == stat_count。
        // ncnn 的 group_count_x = ceil(dispatcher.w / local_size_x)，因此 dispatcher.w
        // 必须等于 stat_count × local_size_x，才能启动恰好 stat_count 个 workgroup。
        dispatcher.w = stat_count * pipeline_reduce->local_size_x();
        dispatcher.h = 1;
        dispatcher.c = 1;

        cmd.record_pipeline(pipeline_reduce, bindings, constants, dispatcher);
    }

    // 阶段二：normalize，每个输出元素做仿射变换。
    {
        std::vector<ncnn::VkMat> bindings(6);
        bindings[0] = bottom_blob;
        bindings[1] = top_blob;
        bindings[2] = weight_data_gpu;
        bindings[3] = bias_data_gpu;
        bindings[4] = sum_workspace;
        bindings[5] = sqsum_workspace;

        const uint32_t total = static_cast<uint32_t>(channels) * t * h * w;
        std::vector<ncnn::vk_constant_type> constants(9);
        constants[0].u32 = static_cast<uint32_t>(w);
        constants[1].u32 = static_cast<uint32_t>(h);
        constants[2].u32 = static_cast<uint32_t>(t);
        constants[3].u32 = static_cast<uint32_t>(cstep);
        constants[4].u32 = static_cast<uint32_t>(spatial_size);
        constants[5].u32 = static_cast<uint32_t>(groups);
        constants[6].u32 = static_cast<uint32_t>(count);
        constants[7].f = eps;
        constants[8].u32 = total;

        ncnn::VkMat dispatcher;
        dispatcher.w = total;
        dispatcher.h = 1;
        dispatcher.c = 1;

        cmd.record_pipeline(pipeline_normalize, bindings, constants, dispatcher);
    }

    return 0;
}

#endif // NCNN_VULKAN

DEFINE_LAYER_CREATOR(DynamicFramewiseGroupNorm)
