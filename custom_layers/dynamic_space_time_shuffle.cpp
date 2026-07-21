// 本文件实现 SeedVR2 VAE decoder 的动态时空 shuffle 自定义层。
// 它等价于 PyTorch 中 learned 1x1x1 Conv3D 后接 MAGViT 风格的 channel-to-space-time
// 重排：输入输出均为 ncnn::Mat(w=W,h=H,d=T,c=C)，空间倍率固定为 2。
#include "dynamic_space_time_shuffle.h"

DynamicSpaceTimeShuffle::DynamicSpaceTimeShuffle()
    : in_channels(0), projected_channels(0), temporal_ratio(0), spatial_ratio(2)
{
    one_blob_only = true;
    support_inplace = false;
    support_packing = false;
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

int DynamicSpaceTimeShuffle::forward(
    const ncnn::Mat& bottom_blob,
    ncnn::Mat& top_blob,
    const ncnn::Option& opt) const
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

DEFINE_LAYER_CREATOR(DynamicSpaceTimeShuffle)
