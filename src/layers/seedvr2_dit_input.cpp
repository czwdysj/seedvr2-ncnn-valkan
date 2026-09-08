// 本文件实现 SeedVR2 DiT 动态输入层的 FP32 CPU 数值基线。
// 2x2 patchify 保持 PyTorch einops 的 (h,w,c) 展开顺序；文本与时间 MLP
// 由 NCNN InnerProduct 执行。Vulkan 路径复用同一份权重顺序与动态形状约束：
// 三个自定义 shader（patchify / sinusoidal / silu）负责原生算子表达不了的
// 动态逻辑，5 个矩阵乘投影直接复用 InnerProduct 的 Vulkan 实现。
#include "layers/seedvr2_dit_input.h"

#include <cmath>
#include <cstring>

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

inline float silu(float value)
{
    return value / (1.f + std::exp(-value));
}
} // namespace

#if NCNN_VULKAN

// ============================================================================
// shader 1：patchify —— 2x2 空间 patch 重排
// ============================================================================
// 输入 video 是 [L, video_channels] 的 2D token-major 张量（L = T*H*W），
// 每个空间位置 video_channels 个通道。输出 patches 是 [patched_rows, vc*4]，
// 每个 patch 把 2x2 的 4 个源位置按 einops 的 (dy,dx,c) 顺序拼接成 vc*4 通道。
// 每个 work item 计算一个 (patch, patch_channel)，解码出源位置后复制单个元素。
//
// 关键：2D VkMat 的 buffer 是 row-major（行步长 = w，即每行 w 个元素连续），
// 不能用 cstep 寻址——cstep 对 dims=2 是 w*h（总大小），只有 4D Mat 的 cstep
// 才是「通道步长」。这里统一用 video_channels / patch_channels（= w）做行步长。
static const char* patchify_shader_source = R"VKGLSL(
#version 450

layout(binding = 0, std430) readonly buffer video_data { sfp video_blob[]; };
layout(binding = 1, std430) writeonly buffer patches_data { sfp patches_blob[]; };

layout(push_constant) uniform parameter {
    uint width;
    uint height;
    uint patched_width;
    uint patched_height;
    uint video_channels;
    uint patch_channels;
    uint total;
} p;

void main()
{
    uint gi = gl_GlobalInvocationID.x;
    if (gi >= p.total) return;

    uint pidx = gi / p.patch_channels;   // patch 索引（注意：patch 是 GLSL 关键字，不能用）
    uint pc = gi % p.patch_channels;     // patch 内通道 [0, vc*4)

    // patch 索引解码为 (t, y, x)（patch 后空间坐标）。
    uint x = pidx % p.patched_width;
    uint y = (pidx / p.patched_width) % p.patched_height;
    uint t = pidx / (p.patched_width * p.patched_height);

    // patch 内通道解码为 (dy, dx, c)：vc*4 = 4 × vc，顺序 dy -> dx -> c。
    uint block = pc / p.video_channels; // [0, 4)
    uint dy = block / 2u;
    uint dx = block % 2u;
    uint c = pc % p.video_channels;

    uint src = (t * p.height + y * 2u + dy) * p.width + x * 2u + dx;
    afp v = buffer_ld1(video_blob, src * p.video_channels + c);
    buffer_st1(patches_blob, pidx * p.patch_channels + pc, v);
}
)VKGLSL";

// ============================================================================
// shader 2：sinusoidal —— 时间步的正弦/余弦位置编码
// ============================================================================
// 输入标量 timestep，输出 sinusoidal_dim 维编码（前一半 sin、后一半 cos），
// 频率按 exp(-log(10000) * i / half) 递减。每个 work item 计算一个维度。
static const char* sinusoidal_shader_source = R"VKGLSL(
#version 450

layout(binding = 0, std430) writeonly buffer sinusoidal_data { sfp sinusoidal_blob[]; };

layout(push_constant) uniform parameter {
    float timestep;
    uint half_dim;   // 注意：half 是 GLSL 保留字，不能作成员名
    uint total;
} p;

void main()
{
    uint i = gl_GlobalInvocationID.x;
    if (i >= p.total) return;

    uint idx = i % p.half_dim;   // 频率索引，i >= half_dim 时复用同一频率
    float freq = exp(-log(10000.0) * float(idx) / float(p.half_dim));
    float value = p.timestep * freq;
    float result = (i < p.half_dim) ? sin(value) : cos(value);
    buffer_st1(sinusoidal_blob, i, result);
}
)VKGLSL";

// ============================================================================
// shader 3：silu —— SiLU 激活（inplace，pack4）
// ============================================================================
// silu(x) = x / (1 + exp(-x))，逐元素 inplace。输入是 InnerProduct 的输出，
// 当 num_output % 4 == 0 时其 elempack=4，因此这里用 sfpvec4 + buffer_ld4/st4
// 处理 pack4 布局（与 ncnn 内置层的约定一致）。
static const char* silu_shader_source = R"VKGLSL(
#version 450

layout(binding = 0, std430) buffer bottom_top_data { sfpvec4 bottom_top_blob[]; };

layout(push_constant) uniform parameter {
    uint total;
} p;

void main()
{
    uint i = gl_GlobalInvocationID.x;
    if (i >= p.total) return;

    afpvec4 v = buffer_ld4(bottom_top_blob, i);
    v = v / (afpvec4(1.0f) + exp(-v));
    buffer_st4(bottom_top_blob, i, v);
}
)VKGLSL";

#endif // NCNN_VULKAN

SeedVR2DiTInput::SeedVR2DiTInput()
    : dim(2560), video_channels(33), text_channels(5120), sinusoidal_dim(256),
      embedding_dim(15360), video_projection(nullptr), text_projection(nullptr),
      time_projection_in(nullptr), time_projection_hidden(nullptr),
      time_projection_out(nullptr)
{
    one_blob_only = false;
    support_inplace = false;
    support_packing = false;
#if NCNN_VULKAN
    support_vulkan = true;
    support_vulkan_packing = false;
    pipeline_patchify = 0;
    pipeline_sinusoidal = 0;
    pipeline_silu = 0;
#endif
}

SeedVR2DiTInput::~SeedVR2DiTInput()
{
    destroy_layers();
}

void SeedVR2DiTInput::destroy_layers()
{
    delete video_projection;
    delete text_projection;
    delete time_projection_in;
    delete time_projection_hidden;
    delete time_projection_out;
    video_projection = text_projection = nullptr;
    time_projection_in = time_projection_hidden = time_projection_out = nullptr;
}

int SeedVR2DiTInput::load_param(const ncnn::ParamDict& pd)
{
    dim = pd.get(0, 2560);
    video_channels = pd.get(1, 33);
    text_channels = pd.get(2, 5120);
    sinusoidal_dim = pd.get(3, 256);
    embedding_dim = pd.get(4, 15360);
    return sinusoidal_dim % 2 == 0 && embedding_dim == dim * 6 ? 0 : -1;
}

int SeedVR2DiTInput::load_model(const ncnn::ModelBin& mb)
{
    video_projection = load_inner_product(mb, video_channels * 4, dim, true, vkdev);
    text_projection = load_inner_product(mb, text_channels, dim, true, vkdev);
    time_projection_in = load_inner_product(mb, sinusoidal_dim, dim, true, vkdev);
    time_projection_hidden = load_inner_product(mb, dim, dim, true, vkdev);
    time_projection_out = load_inner_product(mb, dim, embedding_dim, true, vkdev);
    return video_projection && text_projection && time_projection_in
            && time_projection_hidden && time_projection_out
        ? 0
        : -100;
}

int SeedVR2DiTInput::create_pipeline(const ncnn::Option& opt)
{
    for (ncnn::Layer* layer : {video_projection, text_projection, time_projection_in,
                              time_projection_hidden, time_projection_out})
    {
        if (!layer)
            return -1;
#if NCNN_VULKAN
        // 关键：Net 只给顶层自定义层设置 vkdev，这里复用 InnerProduct 的子层
        // 需要手动把 vkdev 传递下去，否则 Layer_final 委托的 Vulkan 实现拿不到
        // 设备，upload_model/forward(VkMat) 都会失败。
        if (vkdev)
            layer->vkdev = vkdev;
#endif
        if (layer->create_pipeline(opt) != 0)
            return -1;
    }

#if NCNN_VULKAN
    // CPU 模式下 vkdev 尚未赋值，跳过三个自定义 shader 的管线创建。
    if (!vkdev)
        return 0;

    {
        std::vector<uint32_t> spirv;
        const int compiled = ncnn::compile_spirv_module(patchify_shader_source, opt, spirv);
        if (compiled != 0)
        {
            std::fprintf(stderr, "SeedVR2DiTInput: patchify compile failed %d\n", compiled);
            return compiled;
        }
        pipeline_patchify = new ncnn::Pipeline(vkdev);
        pipeline_patchify->set_optimal_local_size_xyz(64, 1, 1);
        pipeline_patchify->create(spirv.data(), spirv.size() * sizeof(uint32_t), std::vector<ncnn::vk_specialization_type>());
    }
    {
        std::vector<uint32_t> spirv;
        const int compiled = ncnn::compile_spirv_module(sinusoidal_shader_source, opt, spirv);
        if (compiled != 0)
        {
            std::fprintf(stderr, "SeedVR2DiTInput: sinusoidal compile failed %d\n", compiled);
            return compiled;
        }
        pipeline_sinusoidal = new ncnn::Pipeline(vkdev);
        pipeline_sinusoidal->set_optimal_local_size_xyz(64, 1, 1);
        pipeline_sinusoidal->create(spirv.data(), spirv.size() * sizeof(uint32_t), std::vector<ncnn::vk_specialization_type>());
    }
    {
        std::vector<uint32_t> spirv;
        const int compiled = ncnn::compile_spirv_module(silu_shader_source, opt, spirv);
        if (compiled != 0)
        {
            std::fprintf(stderr, "SeedVR2DiTInput: silu compile failed %d\n", compiled);
            return compiled;
        }
        pipeline_silu = new ncnn::Pipeline(vkdev);
        pipeline_silu->set_optimal_local_size_xyz(64, 1, 1);
        pipeline_silu->create(spirv.data(), spirv.size() * sizeof(uint32_t), std::vector<ncnn::vk_specialization_type>());
    }
#endif
    return 0;
}

int SeedVR2DiTInput::destroy_pipeline(const ncnn::Option& opt)
{
    for (ncnn::Layer* layer : {video_projection, text_projection, time_projection_in,
                              time_projection_hidden, time_projection_out})
        if (layer)
            layer->destroy_pipeline(opt);

#if NCNN_VULKAN
    delete pipeline_patchify;
    pipeline_patchify = 0;
    delete pipeline_sinusoidal;
    pipeline_sinusoidal = 0;
    delete pipeline_silu;
    pipeline_silu = 0;
#endif
    return 0;
}

int SeedVR2DiTInput::forward(const std::vector<ncnn::Mat>& bottom_blobs,
                             std::vector<ncnn::Mat>& top_blobs,
                             const ncnn::Option& opt) const
{
    if (bottom_blobs.size() != 4 || top_blobs.size() != 4)
        return -1;
    const ncnn::Mat& video = bottom_blobs[0];
    const ncnn::Mat& text = bottom_blobs[1];
    const ncnn::Mat& timestep = bottom_blobs[2];
    const ncnn::Mat& shape = bottom_blobs[3];
    if (video.dims != 2 || video.w != video_channels || text.dims != 2
        || text.w != text_channels || timestep.w < 1 || shape.dims != 1 || shape.w != 3)
        return -1;

    const int frames = static_cast<const int*>(shape.data)[0];
    const int height = static_cast<const int*>(shape.data)[1];
    const int width = static_cast<const int*>(shape.data)[2];
    if (frames <= 0 || height <= 0 || width <= 0 || height % 2 != 0 || width % 2 != 0
        || video.h != frames * height * width)
        return -1;
    const int patched_height = height / 2;
    const int patched_width = width / 2;
    const int patched_rows = frames * patched_height * patched_width;

    ncnn::Mat patches(video_channels * 4, patched_rows, 4u, opt.workspace_allocator);
    if (patches.empty())
        return -100;
#pragma omp parallel for num_threads(opt.num_threads)
    for (int patch = 0; patch < patched_rows; patch++)
    {
        const int x = patch % patched_width;
        const int y = (patch / patched_width) % patched_height;
        const int t = patch / (patched_width * patched_height);
        float* destination = patches.row(patch);
        int offset = 0;
        for (int dy = 0; dy < 2; dy++)
            for (int dx = 0; dx < 2; dx++)
            {
                const int source_index = (t * height + y * 2 + dy) * width + x * 2 + dx;
                std::memcpy(destination + offset, video.row(source_index),
                            static_cast<size_t>(video_channels) * sizeof(float));
                offset += video_channels;
            }
    }

    ncnn::Mat video_output;
    ncnn::Mat text_output;
    if (video_projection->forward(patches, video_output, opt) != 0
        || text_projection->forward(text, text_output, opt) != 0)
        return -100;

    ncnn::Mat sinusoidal(sinusoidal_dim, 4u, opt.workspace_allocator);
    if (sinusoidal.empty())
        return -100;
    const float time_value = static_cast<const float*>(timestep.data)[0];
    const int half = sinusoidal_dim / 2;
    float* sinusoidal_data = sinusoidal;
    for (int i = 0; i < half; i++)
    {
        const float frequency = std::exp(-std::log(10000.f) * i / half);
        const float value = time_value * frequency;
        sinusoidal_data[i] = std::sin(value);
        sinusoidal_data[i + half] = std::cos(value);
    }
    ncnn::Mat time_hidden;
    if (time_projection_in->forward(sinusoidal, time_hidden, opt) != 0)
        return -100;
    float* hidden_data = time_hidden;
    for (size_t i = 0; i < time_hidden.total(); i++)
        hidden_data[i] = silu(hidden_data[i]);
    ncnn::Mat time_hidden_2;
    if (time_projection_hidden->forward(time_hidden, time_hidden_2, opt) != 0)
        return -100;
    hidden_data = time_hidden_2;
    for (size_t i = 0; i < time_hidden_2.total(); i++)
        hidden_data[i] = silu(hidden_data[i]);
    ncnn::Mat embedding;
    if (time_projection_out->forward(time_hidden_2, embedding, opt) != 0)
        return -100;

    ncnn::Mat patched_shape(3, 4u, opt.blob_allocator);
    if (patched_shape.empty())
        return -100;
    int* patched_shape_data = patched_shape;
    patched_shape_data[0] = frames;
    patched_shape_data[1] = patched_height;
    patched_shape_data[2] = patched_width;
    top_blobs[0] = video_output;
    top_blobs[1] = text_output;
    top_blobs[2] = embedding;
    top_blobs[3] = patched_shape;
    return 0;
}

#if NCNN_VULKAN

int SeedVR2DiTInput::upload_model(ncnn::VkTransfer& cmd, const ncnn::Option& opt)
{
    // 5 个投影层的权重由各自 InnerProduct 的 upload_model 上传（Layer_final 委托）。
    const ncnn::Layer* subs[5] = {video_projection, text_projection, time_projection_in,
                                  time_projection_hidden, time_projection_out};
    const char* names[5] = {"video", "text", "time_in", "time_hidden", "time_out"};
    for (int i = 0; i < 5; i++)
    {
        if (!subs[i])
            return -1;
        const int ret = const_cast<ncnn::Layer*>(subs[i])->upload_model(cmd, opt);
        if (ret != 0)
        {
            std::fprintf(stderr, "SeedVR2DiTInput: %s upload_model failed %d\n", names[i], ret);
            return -1;
        }
    }
    return 0;
}

int SeedVR2DiTInput::forward(const std::vector<ncnn::VkMat>& bottom_blobs,
                             std::vector<ncnn::VkMat>& top_blobs,
                             ncnn::VkCompute& cmd,
                             const ncnn::Option& opt) const
{
    if (bottom_blobs.size() != 4 || top_blobs.size() != 4)
        return -1;
    const ncnn::VkMat& video = bottom_blobs[0];
    const ncnn::VkMat& text = bottom_blobs[1];
    const ncnn::VkMat& timestep = bottom_blobs[2];
    const ncnn::VkMat& shape = bottom_blobs[3];
    if (video.dims != 2 || video.w != video_channels || text.dims != 2
        || text.w != text_channels || timestep.w < 1 || shape.w != 3)
        return -1;

    // shape 存 int 位模式、timestep 存 float。它们是极小的标量/形状数据，
    // 需要 CPU 侧同步读取（作为 patchify/sinusoidal 的 push constant）。
    // 注意：不能直接 mapped_ptr 读，因为 record_upload 的 convert_packing 是
    // 异步记录的，forward 时 dst buffer 尚未写入。这里用 record_download 下载到
    // CPU 后 submit_and_wait 同步等待，保证读到最新值（正确性优先阶段可接受，
    // 数据量仅 4 个标量）。
    ncnn::Mat shape_cpu;
    ncnn::Mat timestep_cpu;
    cmd.record_download(shape, shape_cpu, opt);
    cmd.record_download(timestep, timestep_cpu, opt);
    cmd.submit_and_wait();
    cmd.reset();

    const int* shape_data = static_cast<const int*>(shape_cpu.data);
    const float timestep_val = static_cast<const float*>(timestep_cpu.data)[0];
    const int frames = shape_data[0];
    const int height = shape_data[1];
    const int width = shape_data[2];
    if (frames <= 0 || height <= 0 || width <= 0 || height % 2 != 0 || width % 2 != 0)
        return -1;
    const int patched_height = height / 2;
    const int patched_width = width / 2;
    const int patched_rows = frames * patched_height * patched_width;
    const int patch_channels = video_channels * 4;

    // 阶段一：patchify，video [L,33] -> patches [patched_rows,132]。
    ncnn::VkMat patches;
    patches.create(patch_channels, patched_rows, 4u, 1, opt.workspace_vkallocator);
    if (patches.empty())
        return -100;
    {
        std::vector<ncnn::VkMat> bindings(2);
        bindings[0] = video;
        bindings[1] = patches;

        const uint32_t total = static_cast<uint32_t>(patched_rows) * patch_channels;
        std::vector<ncnn::vk_constant_type> constants(7);
        constants[0].u32 = static_cast<uint32_t>(width);
        constants[1].u32 = static_cast<uint32_t>(height);
        constants[2].u32 = static_cast<uint32_t>(patched_width);
        constants[3].u32 = static_cast<uint32_t>(patched_height);
        constants[4].u32 = static_cast<uint32_t>(video_channels);
        constants[5].u32 = static_cast<uint32_t>(patch_channels);
        constants[6].u32 = total;

        ncnn::VkMat dispatcher;
        dispatcher.w = total;
        dispatcher.h = 1;
        dispatcher.c = 1;

        cmd.record_pipeline(pipeline_patchify, bindings, constants, dispatcher);
    }

    // 阶段二：video/text 投影（复用 InnerProduct 的 Vulkan 实现）。
    ncnn::VkMat video_output;
    ncnn::VkMat text_output;
    if (video_projection->forward(patches, video_output, cmd, opt) != 0
        || text_projection->forward(text, text_output, cmd, opt) != 0)
        return -100;

    // 阶段三：sinusoidal 时间编码。
    ncnn::VkMat sinusoidal;
    sinusoidal.create(sinusoidal_dim, 4u, 1, opt.workspace_vkallocator);
    if (sinusoidal.empty())
        return -100;
    {
        std::vector<ncnn::VkMat> bindings(1);
        bindings[0] = sinusoidal;

        const int half = sinusoidal_dim / 2;
        std::vector<ncnn::vk_constant_type> constants(3);
        constants[0].f = timestep_val;
        constants[1].u32 = static_cast<uint32_t>(half);
        constants[2].u32 = static_cast<uint32_t>(sinusoidal_dim);

        ncnn::VkMat dispatcher;
        dispatcher.w = sinusoidal_dim;
        dispatcher.h = 1;
        dispatcher.c = 1;

        cmd.record_pipeline(pipeline_sinusoidal, bindings, constants, dispatcher);
    }

    // 阶段四：时间 MLP（三层 InnerProduct，中间两层 SiLU 激活）。
    // 关键：InnerProduct 对 num_input/num_output 是 4 的倍数时用 elempack=4，
    // 因此 sinusoidal（pack1）需先 convert_packing 到 pack4 再送入；silu 也用
    // pack4（time_hidden 的输出是 pack4）。这与 net 在层间做 convert_layout 的
    // 行为一致，复用 InnerProduct 时必须手动补上这一步。
    ncnn::VkMat sinusoidal_pack4;
    sinusoidal_pack4.create(sinusoidal_dim / 4, 4u * 4, 4, opt.workspace_vkallocator);
    if (sinusoidal_pack4.empty())
        return -100;
    vkdev->convert_packing(sinusoidal, sinusoidal_pack4, 4, cmd, opt);

    ncnn::VkMat time_hidden;
    if (time_projection_in->forward(sinusoidal_pack4, time_hidden, cmd, opt) != 0)
        return -100;
    {
        std::vector<ncnn::VkMat> bindings(1);
        bindings[0] = time_hidden;
        std::vector<ncnn::vk_constant_type> constants(1);
        constants[0].u32 = static_cast<uint32_t>(dim / 4);
        ncnn::VkMat dispatcher;
        dispatcher.w = dim / 4;
        dispatcher.h = 1;
        dispatcher.c = 1;
        cmd.record_pipeline(pipeline_silu, bindings, constants, dispatcher);
    }

    ncnn::VkMat time_hidden_2;
    if (time_projection_hidden->forward(time_hidden, time_hidden_2, cmd, opt) != 0)
        return -100;
    {
        std::vector<ncnn::VkMat> bindings(1);
        bindings[0] = time_hidden_2;
        std::vector<ncnn::vk_constant_type> constants(1);
        constants[0].u32 = static_cast<uint32_t>(dim / 4);
        ncnn::VkMat dispatcher;
        dispatcher.w = dim / 4;
        dispatcher.h = 1;
        dispatcher.c = 1;
        cmd.record_pipeline(pipeline_silu, bindings, constants, dispatcher);
    }

    ncnn::VkMat embedding;
    if (time_projection_out->forward(time_hidden_2, embedding, cmd, opt) != 0)
        return -100;

    // 阶段五：patched_shape 输出（int 位模式，经 record_upload 传上 GPU）。
    ncnn::Mat patched_shape_cpu(3, static_cast<size_t>(4u), opt.blob_allocator);
    int* ps = patched_shape_cpu;
    ps[0] = frames;
    ps[1] = patched_height;
    ps[2] = patched_width;
    top_blobs[3].create(3, 4u, 1, opt.blob_vkallocator);
    if (top_blobs[3].empty())
        return -100;
    cmd.record_upload(patched_shape_cpu, top_blobs[3], opt);

    top_blobs[0] = video_output;
    top_blobs[1] = text_output;
    top_blobs[2] = embedding;
    return 0;
}

#endif // NCNN_VULKAN

DEFINE_LAYER_CREATOR(SeedVR2DiTInput)
