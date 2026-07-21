// 本文件声明 SeedVR2 VAE decoder 使用的动态时空 shuffle 自定义层。
// 该层先加载 learned 1x1x1 projection 的权重，再根据运行时输入的 T,H,W
// 把投影后的通道重新排列到时间和空间维度，用于还原 MAGViT 风格的上采样逻辑。
#pragma once

#include <layer.h>

class DynamicSpaceTimeShuffle : public ncnn::Layer
{
public:
    DynamicSpaceTimeShuffle();

    int load_param(const ncnn::ParamDict& pd) override;
    int load_model(const ncnn::ModelBin& mb) override;
    int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const override;

private:
    int in_channels;
    int projected_channels;
    // temporal_ratio/spatial_ratio 决定通道展开到 T/H/W 的比例。
    int temporal_ratio;
    int spatial_ratio;
    ncnn::Mat bias_data;
    ncnn::Mat weight_data;
};

ncnn::Layer* DynamicSpaceTimeShuffle_layer_creator(void* userdata);
