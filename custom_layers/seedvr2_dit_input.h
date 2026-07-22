// 本文件声明 SeedVR2 3B DiT 的动态输入层。
// 输入是 vid[L,33]、txt[L_txt,5120]、标量 timestep 和原始 latent 形状
// [T,H,W]；输出是 2x2 patch 后的视频 token、文本 token、15360 维时间
// embedding 和 patch 后形状。batch 固定为 1，T/H/W 与文本长度运行时确定。
#pragma once

#include <layer.h>

class SeedVR2DiTInput : public ncnn::Layer
{
public:
    SeedVR2DiTInput();
    ~SeedVR2DiTInput() override;

    int load_param(const ncnn::ParamDict& pd) override;
    int load_model(const ncnn::ModelBin& mb) override;
    int create_pipeline(const ncnn::Option& opt) override;
    int destroy_pipeline(const ncnn::Option& opt) override;
    int forward(const std::vector<ncnn::Mat>& bottom_blobs,
                std::vector<ncnn::Mat>& top_blobs,
                const ncnn::Option& opt) const override;

private:
    void destroy_layers();

private:
    int dim;
    int video_channels;
    int text_channels;
    int sinusoidal_dim;
    int embedding_dim;
    ncnn::Layer* video_projection;
    ncnn::Layer* text_projection;
    ncnn::Layer* time_projection_in;
    ncnn::Layer* time_projection_hidden;
    ncnn::Layer* time_projection_out;
};

ncnn::Layer* SeedVR2DiTInput_layer_creator(void* userdata);
