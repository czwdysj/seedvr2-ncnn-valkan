// 本文件声明 SeedVR2 3B DiT 的动态输入层。
// 输入是 vid[L,33]、txt[L_txt,5120]、标量 timestep 和原始 latent 形状
// [T,H,W]；输出是 2x2 patch 后的视频 token、文本 token、15360 维时间
// embedding 和 patch 后形状。batch 固定为 1，T/H/W 与文本长度运行时确定。
//
// 本层是「自定义层编排 + 原生算子干重活」的典型：5 个矩阵乘投影全部复用
// ncnn 原生 InnerProduct（create_layer 返回 Layer_final，自动获得 CPU + Vulkan
// 双实现），自定义代码只负责原生算子表达不了的动态逻辑——patchify（2x2 空间
// 重排）、sinusoidal 时间编码、SiLU 激活。这三段在 Vulkan 路径下对应三个
// compute shader，以内嵌字符串携带，create_pipeline 阶段用运行时 glslang 编译。
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

    // Vulkan shader 的分派尺寸和时间编码来自调用方已知的 CPU 元数据。
    // 显式注入后，forward 不需要为了读取四个标量而下载 VkMat 并等待 GPU。
    void set_runtime_metadata(int frames, int height, int width, float timestep);

#if NCNN_VULKAN
    int upload_model(ncnn::VkTransfer& cmd, const ncnn::Option& opt) override;
    int forward(const std::vector<ncnn::VkMat>& bottom_blobs,
                std::vector<ncnn::VkMat>& top_blobs,
                ncnn::VkCompute& cmd,
                const ncnn::Option& opt) const override;
#endif

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
    int runtime_frames;
    int runtime_height;
    int runtime_width;
    float runtime_timestep;
    bool runtime_metadata_valid;

#if NCNN_VULKAN
    // 三个自定义 compute shader 的管线（矩阵乘走原生 InnerProduct 的 Vulkan 实现）。
    ncnn::Pipeline* pipeline_patchify;
    ncnn::Pipeline* pipeline_sinusoidal;
    ncnn::Pipeline* pipeline_silu;
#endif
};

ncnn::Layer* SeedVR2DiTInput_layer_creator(void* userdata);
