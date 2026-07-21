// 本文件实现 SeedVR2 VAE 的动态逐帧 GroupNorm 自定义层。
// 普通 pnnx/NCNN 图很容易把视频帧数 T 固定进 reshape 或 reduction 中；
// 这里把 GroupNorm 写成 layer，让同一份 param/bin 可以处理运行时变化的 T,H,W。
// 输入输出统一使用 ncnn::Mat(w=W,h=H,d=T,c=C)，和参考张量的 C,T,H,W 逻辑对应。
#include "dynamic_framewise_group_norm.h"

#include <cmath>

DynamicFramewiseGroupNorm::DynamicFramewiseGroupNorm()
    : channels(0), groups(32), eps(1e-6f)
{
    one_blob_only = true;
    support_inplace = false;
    support_packing = false;
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
            // PyTorch 的 VAE 这里按“单帧内的 group”统计均值方差，
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

DEFINE_LAYER_CREATOR(DynamicFramewiseGroupNorm)
