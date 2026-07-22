// 本文件实现 SeedVR2 DiT 动态输出层的 FP32 CPU 数值基线。
// 实现包括 affine RMSNorm、与原 PyTorch Cache 键碰撞结果一致的 output Ada、
// NCNN InnerProduct 投影以及按 (h,w,c) 顺序执行的 2x2 unpatch。
#include "seedvr2_dit_output.h"

#include <cmath>

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
} // namespace

SeedVR2DiTOutput::SeedVR2DiTOutput()
    : dim(2560), output_channels(16), norm_eps(1e-5f), projection(nullptr)
{
    one_blob_only = false;
    support_inplace = false;
    support_packing = false;
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
    return 0;
}

int SeedVR2DiTOutput::load_model(const ncnn::ModelBin& mb)
{
    norm_weight = mb.load(dim, 1);
    output_shift = mb.load(dim, 1);
    output_scale = mb.load(dim, 1);
    projection = load_inner_product(mb, dim, output_channels * 4, true);
    return norm_weight.empty() || output_shift.empty() || output_scale.empty() || !projection
        ? -100
        : 0;
}

int SeedVR2DiTOutput::create_pipeline(const ncnn::Option& opt)
{
    return projection ? projection->create_pipeline(opt) : -1;
}

int SeedVR2DiTOutput::destroy_pipeline(const ncnn::Option& opt)
{
    return projection ? projection->destroy_pipeline(opt) : 0;
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

DEFINE_LAYER_CREATOR(SeedVR2DiTOutput)
