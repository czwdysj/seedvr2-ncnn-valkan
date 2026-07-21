// 本文件声明 SeedVR2 VAE bottleneck 中的动态逐帧空间自注意力层。
// 输入输出使用 ncnn::Mat(w=W,h=H,d=T,c=C)；layer 内部拥有原始 PyTorch
// GroupNorm、Q/K/V 投影和输出投影权重，按每一帧独立执行 H*W token 注意力。
#pragma once

#include <layer.h>

class DynamicFramewiseSpatialAttention : public ncnn::Layer
{
public:
    DynamicFramewiseSpatialAttention();

    int load_param(const ncnn::ParamDict& pd) override;
    int load_model(const ncnn::ModelBin& mb) override;
    int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const override;

private:
    int channels;
    // 注意力前的 GroupNorm 权重。
    ncnn::Mat norm_bias;
    ncnn::Mat norm_weight;
    // Q/K/V/out 的线性层参数由 pnnx 写入 bin，CPU layer 按固定顺序读回。
    ncnn::Mat k_bias;
    ncnn::Mat k_weight;
    ncnn::Mat out_bias;
    ncnn::Mat out_weight;
    ncnn::Mat q_bias;
    ncnn::Mat q_weight;
    ncnn::Mat v_bias;
    ncnn::Mat v_weight;
};

ncnn::Layer* DynamicFramewiseSpatialAttention_layer_creator(void* userdata);
