// 本文件实现 SeedVR2 DiT 动态输入层的 FP32 CPU 数值基线。
// 2x2 patchify 保持 PyTorch einops 的 (h,w,c) 展开顺序；文本与时间 MLP
// 由 NCNN InnerProduct 执行。该文件暂不提供 Vulkan 路径，GPU shader 将复用
// 相同的动态形状约束和权重顺序。
#include "seedvr2_dit_input.h"

#include <cmath>
#include <cstring>

namespace
{
ncnn::Layer* load_inner_product(const ncnn::ModelBin& mb, int input_size,
                                int output_size, bool bias)
{
    ncnn::Layer* layer = ncnn::create_layer("InnerProduct");
    if (!layer)
        return nullptr;
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

SeedVR2DiTInput::SeedVR2DiTInput()
    : dim(2560), video_channels(33), text_channels(5120), sinusoidal_dim(256),
      embedding_dim(15360), video_projection(nullptr), text_projection(nullptr),
      time_projection_in(nullptr), time_projection_hidden(nullptr),
      time_projection_out(nullptr)
{
    one_blob_only = false;
    support_inplace = false;
    support_packing = false;
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
    video_projection = load_inner_product(mb, video_channels * 4, dim, true);
    text_projection = load_inner_product(mb, text_channels, dim, true);
    time_projection_in = load_inner_product(mb, sinusoidal_dim, dim, true);
    time_projection_hidden = load_inner_product(mb, dim, dim, true);
    time_projection_out = load_inner_product(mb, dim, embedding_dim, true);
    return video_projection && text_projection && time_projection_in
            && time_projection_hidden && time_projection_out
        ? 0
        : -100;
}

int SeedVR2DiTInput::create_pipeline(const ncnn::Option& opt)
{
    for (ncnn::Layer* layer : {video_projection, text_projection, time_projection_in,
                              time_projection_hidden, time_projection_out})
        if (!layer || layer->create_pipeline(opt) != 0)
            return -1;
    return 0;
}

int SeedVR2DiTInput::destroy_pipeline(const ncnn::Option& opt)
{
    for (ncnn::Layer* layer : {video_projection, text_projection, time_projection_in,
                              time_projection_hidden, time_projection_out})
        if (layer)
            layer->destroy_pipeline(opt);
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

DEFINE_LAYER_CREATOR(SeedVR2DiTInput)
