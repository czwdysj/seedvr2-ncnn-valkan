// 本文件声明 SeedVR2 VAE 使用的动态逐帧 GroupNorm 自定义层。
// 输入和输出都按 ncnn::Mat(w=W,h=H,d=T,c=C) 表示视频 latent；layer 会在运行时
// 根据 T,H,W 计算每一帧的分组均值和方差，并加载 pnnx 打包进模型的仿射权重。
#pragma once

#include <layer.h>

class DynamicFramewiseGroupNorm : public ncnn::Layer
{
public:
    DynamicFramewiseGroupNorm();

    int load_param(const ncnn::ParamDict& pd) override;
    int load_model(const ncnn::ModelBin& mb) override;
    int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const override;

private:
    // channels/groups/eps 对应 PyTorch GroupNorm 的通道数、组数和数值稳定项。
    int channels;
    int groups;
    float eps;
    // pnnx 将自定义模块的 tensor 属性写入 bin，本 layer 负责按约定顺序读取。
    ncnn::Mat bias_data;
    ncnn::Mat weight_data;
};

ncnn::Layer* DynamicFramewiseGroupNorm_layer_creator(void* userdata);
