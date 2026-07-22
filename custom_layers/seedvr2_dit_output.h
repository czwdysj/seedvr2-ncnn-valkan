// 本文件声明 SeedVR2 3B DiT 的动态输出层。
// 输入为最后一个 block 的视频 token、15360 维 timestep embedding 和 patch
// 形状[T,H,W]；输出恢复为 [T,H*2,W*2,16] 的扁平 token 及对应动态形状。
#pragma once

#include <layer.h>

class SeedVR2DiTOutput : public ncnn::Layer
{
public:
    SeedVR2DiTOutput();
    ~SeedVR2DiTOutput() override;

    int load_param(const ncnn::ParamDict& pd) override;
    int load_model(const ncnn::ModelBin& mb) override;
    int create_pipeline(const ncnn::Option& opt) override;
    int destroy_pipeline(const ncnn::Option& opt) override;
    int forward(const std::vector<ncnn::Mat>& bottom_blobs,
                std::vector<ncnn::Mat>& top_blobs,
                const ncnn::Option& opt) const override;

private:
    int dim;
    int output_channels;
    float norm_eps;
    ncnn::Mat norm_weight;
    ncnn::Mat output_shift;
    ncnn::Mat output_scale;
    ncnn::Layer* projection;
};

ncnn::Layer* SeedVR2DiTOutput_layer_creator(void* userdata);
