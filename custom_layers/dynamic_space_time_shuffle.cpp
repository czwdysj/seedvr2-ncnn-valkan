// 本文件实现 SeedVR2 VAE decoder 的动态时空 shuffle 自定义层。
// 它等价于 PyTorch 中 learned 1x1x1 Conv3D 后接 MAGViT 风格的 channel-to-space-time
// 重排：输入输出均为 ncnn::Mat(w=W,h=H,d=T,c=C)，空间倍率固定为 2。
//
// 本文件同时提供 CPU 与 Vulkan 两条 forward 路径：
//   - CPU 路径（forward 的 Mat 重载）作为数值基线，已与 PyTorch reference 对齐；
//   - Vulkan 路径（forward 的 VkMat 重载）把同样的索引/重排语义翻译成 GLSL compute
//     shader。shader 以内嵌字符串携带，create_pipeline 阶段用 ncnn 的运行时 glslang
//     编译成 SPIR-V；权重在 upload_model 阶段上传 GPU，forward 只记录一次 dispatch。
// 两条路径共享 load_param/load_model，保证权重与动态语义完全一致。
#include "dynamic_space_time_shuffle.h"

#if NCNN_VULKAN
#include <gpu.h>
#include <vector>
#endif

DynamicSpaceTimeShuffle::DynamicSpaceTimeShuffle()
    : in_channels(0), projected_channels(0), temporal_ratio(0), spatial_ratio(2)
{
    one_blob_only = true;
    support_inplace = false;
    support_packing = false;
#if NCNN_VULKAN
    // 标量布局（elempack=1），因此只声明支持 Vulkan、不支持 Vulkan packing。
    support_vulkan = true;
    support_vulkan_packing = false;
    pipeline = 0;
#endif
}

DynamicSpaceTimeShuffle::~DynamicSpaceTimeShuffle()
{
#if NCNN_VULKAN
    // 注意：pipeline 由 create_pipeline 分配，destroy_pipeline 负责释放；
    // 析构只兜底清理 GPU 权重缓冲的引用（VkMat 析构自带 release）。
#endif
}

int DynamicSpaceTimeShuffle::load_param(const ncnn::ParamDict& pd)
{
    // bias/weight shape 来自导出脚本写入的参数，用来反推输入通道和展开倍率。
    // 这样同一份 layer 可以适配不同空间尺寸，不需要把 H/W 写死进 param。
    const ncnn::Mat bias_shape = pd.get(10, ncnn::Mat());
    const ncnn::Mat weight_shape = pd.get(11, ncnn::Mat());
    if (bias_shape.empty() || weight_shape.empty())
        return -1;
    projected_channels = static_cast<const int*>(bias_shape)[0];
    in_channels = static_cast<const int*>(weight_shape)[1];
    const int ratio = projected_channels / in_channels;
    temporal_ratio = ratio / (spatial_ratio * spatial_ratio);
    // 投影通道数必须刚好能拆成 spatial_ratio^2 * temporal_ratio 份，
    // 每一份对应输出张量中的一个空间/时间偏移。
    return projected_channels == in_channels * temporal_ratio * 4 ? 0 : -1;
}

int DynamicSpaceTimeShuffle::load_model(const ncnn::ModelBin& mb)
{
    // 1x1x1 projection 的权重形状可视为 [projected_channels, in_channels]。
    bias_data = mb.load(projected_channels, 1);
    weight_data = mb.load(projected_channels * in_channels, 1);
    return bias_data.empty() || weight_data.empty() ? -100 : 0;
}

int DynamicSpaceTimeShuffle::forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const
{
    if (bottom_blob.dims != 4 || bottom_blob.c != in_channels || bottom_blob.elempack != 1)
        return -1;

    const int input_width = bottom_blob.w;
    const int input_height = bottom_blob.h;
    const int input_frames = bottom_blob.d;
    // temporal_ratio > 1 时，PyTorch decoder 会删除一个由头帧扩展带来的重复时间位置，
    // 因此输出 T 不是 input_T * temporal_ratio，而是 input_T * temporal_ratio - 1。
    top_blob.create(
        input_width * spatial_ratio,
        input_height * spatial_ratio,
        temporal_ratio > 1 ? input_frames * temporal_ratio - 1 : input_frames,
        in_channels,
        4u,
        1,
        opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const float* weights = weight_data;
    const float* biases = bias_data;
#pragma omp parallel for num_threads(opt.num_threads)
    for (int output_channel = 0; output_channel < in_channels; output_channel++)
    {
        for (int frame = 0; frame < input_frames; frame++)
        {
            for (int y = 0; y < input_height; y++)
            {
                for (int x = 0; x < input_width; x++)
                {
                    for (int offset_y = 0; offset_y < spatial_ratio; offset_y++)
                    {
                        for (int offset_x = 0; offset_x < spatial_ratio; offset_x++)
                        {
                            for (int offset_t = 0; offset_t < temporal_ratio; offset_t++)
                            {
                                // projected_channel 编码了空间偏移、时间偏移和真实输出通道。
                                // 这一步是把 Conv3D 投影后的通道索引映射回 T,H,W 位置。
                                const int projected_channel =
                                    ((offset_y * spatial_ratio + offset_x) * temporal_ratio + offset_t)
                                        * in_channels
                                    + output_channel;
                                const int raw_output_frame = frame * temporal_ratio + offset_t;
                                // raw_output_frame == 1 是 PyTorch decoder 中被移除的重复头帧位置。
                                // 跳过它之后，后续时间索引整体左移一位，才能和 reference 对齐。
                                if (temporal_ratio > 1 && raw_output_frame == 1)
                                    continue;
                                const int output_frame =
                                    temporal_ratio > 1 && raw_output_frame > 1
                                        ? raw_output_frame - 1
                                        : raw_output_frame;
                                const float* weight_row = weights + static_cast<size_t>(projected_channel) * in_channels;
                                double value = biases[projected_channel];
                                // 这里手写 1x1x1 projection：只混合通道，不混合空间或时间位置。
                                for (int input_channel = 0; input_channel < in_channels; input_channel++)
                                {
                                    const float* input = bottom_blob.channel(input_channel).depth(frame);
                                    value += static_cast<double>(input[y * input_width + x]) * weight_row[input_channel];
                                }
                                float* output = top_blob.channel(output_channel).depth(output_frame);
                                output[(y * spatial_ratio + offset_y) * top_blob.w
                                       + x * spatial_ratio + offset_x] = static_cast<float>(value);
                            }
                        }
                    }
                }
            }
        }
    }
    return 0;
}

#if NCNN_VULKAN

// GLSL compute shader：每个 work item 计算一个输出元素。
//
// 与 CPU 实现完全等价的索引映射（见 forward 的 Mat 版本）：
//   - 输出元素 gi 解码为 (c, x_out, y_out, t_out)；
//   - 由 (x_out,y_out) 反推输入空间位置 (x_in,y_in) 与子像素偏移 (ox,oy)；
//   - 由 t_out 反推 raw_t（r_t==2 时跳过 raw_t==1 的首帧复制位），
//     得到输入帧 t_in 与时间偏移 ot；
//   - projected channel pc = ((oy*2+ox)*r_t + ot)*c_out + c；
//   - 值 = bias[pc] + Σ_ci weight[pc*c_in+ci] * input[ci*cstep + t_in*H*W + y_in*W + x_in]。
//
// 关键约定（对齐 ncnn 内置 shader）：
//   - buffer 一律用 ncnn 注入的 `sfp` 类型声明（fp32=float / fp16_packed=uint /
//     fp16_storage=float16_t），读写必须走 `buffer_ld1`/`buffer_st1` 宏——
//     宏会根据实际 storage 精度做 unpack/pack，裸 `float` + `[]` 会在 fp16_packed
//     下把「两个 half 打包成的 uint32」错当 float 读，得到纯垃圾。
//   - 累加器用 `afp`（arithmetic fp，恒为 fp32 或 fp16 的算术精度），
//     与 ncnn 内置层一致，保证内积精度。
//   - c_in/c_out/r_t 用 specialization constant 传入，让 glslang 对内积循环做
//     编译期展开；运行时形状（T/H/W/cstep 等）走 push constant。
static const char* shuffle_shader_source = R"VKGLSL(
#version 450

layout(constant_id = 0) const uint c_in = 4u;
layout(constant_id = 1) const uint c_out = 4u;
layout(constant_id = 2) const uint r_t = 2u;

layout(binding = 0, std430) buffer bottom_data { sfp bottom_blob[]; };
layout(binding = 1, std430) buffer top_data { sfp top_blob[]; };
layout(binding = 2, std430) buffer weight_data { sfp weight_blob[]; };
layout(binding = 3, std430) buffer bias_data { sfp bias_blob[]; };

layout(push_constant) uniform parameter {
    uint t_in;
    uint h;
    uint w_in;
    uint cstep_in;
    uint t_out;
    uint h_out;
    uint w_out;
    uint cstep_out;
    uint total;
} p;

void main()
{
    uint gi = gl_GlobalInvocationID.x;
    if (gi >= p.total)
        return;

    // gi 是稠密输出索引（不含 cstep padding），按 ncnn 4D 布局解码：
    // 最内层 x -> y -> t(d) -> 最外层 c。
    uint x_out = gi % p.w_out;
    uint y_out = (gi / p.w_out) % p.h_out;
    uint t_out = (gi / (p.w_out * p.h_out)) % p.t_out;
    uint c = gi / (p.w_out * p.h_out * p.t_out);

    // 空间子像素反推（spatial_ratio == 2 固定）。
    uint ox = x_out & 1u;
    uint x_in = x_out >> 1;
    uint oy = y_out & 1u;
    uint y_in = y_out >> 1;

    // 时间索引反推：r_t==2 时 PyTorch 移除了 raw_t==1 的首帧复制位。
    uint t_in;
    uint ot;
    if (r_t == 2u)
    {
        uint raw_t = (t_out == 0u) ? 0u : (t_out + 1u);
        t_in = raw_t >> 1;
        ot = raw_t & 1u;
    }
    else
    {
        t_in = t_out;
        ot = 0u;
    }

    uint pc = ((oy * 2u + ox) * r_t + ot) * c_out + c;

    afp sum = buffer_ld1(bias_blob, pc);

    uint base = t_in * (p.h * p.w_in) + y_in * p.w_in + x_in;
    for (uint ci = 0u; ci < c_in; ci++)
    {
        sum += buffer_ld1(weight_blob, pc * c_in + ci) * buffer_ld1(bottom_blob, ci * p.cstep_in + base);
    }

    // 写入地址：c 在最外层且用 cstep_out 对齐，避免 padding 错位。
    uint dst = c * p.cstep_out + t_out * (p.h_out * p.w_out) + y_out * p.w_out + x_out;
    buffer_st1(top_blob, dst, sum);
}
)VKGLSL";

int DynamicSpaceTimeShuffle::create_pipeline(const ncnn::Option& opt)
{
    // CPU 模式下 Net 同样会调用 create_pipeline，但此时 vkdev 尚未赋值；
    // 没有 Vulkan 设备就直接返回，避免空指针构造 Pipeline。
    if (!vkdev)
        return 0;

    // specialization：把通道数/倍率烤进 SPIR-V，换取内积循环的编译期优化。
    std::vector<ncnn::vk_specialization_type> specializations(3);
    specializations[0].u32 = static_cast<uint32_t>(in_channels);
    specializations[1].u32 = static_cast<uint32_t>(in_channels); // c_out == c_in（load_param 已保证）
    specializations[2].u32 = static_cast<uint32_t>(temporal_ratio);

    // 运行时把 GLSL 源码编译成 SPIR-V（ncnn 内嵌 glslang），避免改动 ncnn 的
    // 静态 shader 注册表，也让自定义层的 shader 跟随代码版本一起走 git。
    std::vector<uint32_t> spirv;
    const int compiled = ncnn::compile_spirv_module(shuffle_shader_source, opt, spirv);
    if (compiled != 0)
    {
        std::fprintf(stderr, "DynamicSpaceTimeShuffle: compile_spirv_module failed %d\n", compiled);
        return compiled;
    }
    if (getenv("PROBE_DEBUG"))
        std::fprintf(stderr, "DynamicSpaceTimeShuffle: spirv=%zu words in_channels=%d r_t=%d\n",
                     spirv.size(), in_channels, temporal_ratio);

    pipeline = new ncnn::Pipeline(vkdev);
    pipeline->set_optimal_local_size_xyz(64, 1, 1);
    return pipeline->create(spirv.data(), spirv.size() * sizeof(uint32_t), specializations);
}

int DynamicSpaceTimeShuffle::destroy_pipeline(const ncnn::Option& /*opt*/)
{
    delete pipeline;
    pipeline = 0;
    return 0;
}

int DynamicSpaceTimeShuffle::upload_model(ncnn::VkTransfer& cmd, const ncnn::Option& opt)
{
    // 权重/偏置在推理前一次性上传 GPU，之后 forward 只引用 buffer，不做主机侧搬运。
    cmd.record_upload(weight_data, weight_data_gpu, opt);
    cmd.record_upload(bias_data, bias_data_gpu, opt);
    return 0;
}

int DynamicSpaceTimeShuffle::forward(const ncnn::VkMat& bottom_blob, ncnn::VkMat& top_blob, ncnn::VkCompute& cmd, const ncnn::Option& opt) const
{
    const int w_in = bottom_blob.w;
    const int h = bottom_blob.h;
    const int t_in = bottom_blob.d;
    const int cstep = static_cast<int>(bottom_blob.cstep);

    const int t_out = temporal_ratio > 1 ? t_in * temporal_ratio - 1 : t_in;
    const int h_out = h * spatial_ratio;
    const int w_out = w_in * spatial_ratio;

    top_blob.create(w_out, h_out, t_out, in_channels, 4u, 1, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    std::vector<ncnn::VkMat> bindings(4);
    bindings[0] = bottom_blob;
    bindings[1] = top_blob;
    bindings[2] = weight_data_gpu;
    bindings[3] = bias_data_gpu;

    const uint32_t total = static_cast<uint32_t>(in_channels) * t_out * h_out * w_out;
    std::vector<ncnn::vk_constant_type> constants(9);
    constants[0].u32 = static_cast<uint32_t>(t_in);
    constants[1].u32 = static_cast<uint32_t>(h);
    constants[2].u32 = static_cast<uint32_t>(w_in);
    constants[3].u32 = static_cast<uint32_t>(cstep);
    constants[4].u32 = static_cast<uint32_t>(t_out);
    constants[5].u32 = static_cast<uint32_t>(h_out);
    constants[6].u32 = static_cast<uint32_t>(w_out);
    constants[7].u32 = static_cast<uint32_t>(top_blob.cstep);
    constants[8].u32 = total;

    ncnn::VkMat dispatcher;
    dispatcher.w = total;
    dispatcher.h = 1;
    dispatcher.c = 1;

    cmd.record_pipeline(pipeline, bindings, constants, dispatcher);
    return 0;
}

#endif // NCNN_VULKAN

DEFINE_LAYER_CREATOR(DynamicSpaceTimeShuffle)
