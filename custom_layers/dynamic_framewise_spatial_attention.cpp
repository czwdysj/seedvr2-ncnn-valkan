// 本文件实现 SeedVR2 VAE bottleneck 的动态逐帧空间自注意力层。
// 它是正确性优先的 CPU 版本：全部使用 FP32，并在点积和 softmax 分母处用 double
// 累加，目的是先和 PyTorch reference 对齐；后续 Vulkan 版本应复用同一套输入输出测试。
#include "dynamic_framewise_spatial_attention.h"

#include <algorithm>
#include <cmath>
#include <vector>

DynamicFramewiseSpatialAttention::DynamicFramewiseSpatialAttention()
    : channels(0)
{
    one_blob_only = true;
    support_inplace = false;
    support_packing = false;
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

DEFINE_LAYER_CREATOR(DynamicFramewiseSpatialAttention)
