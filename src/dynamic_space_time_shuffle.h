// 本文件声明 SeedVR2 VAE decoder 使用的动态时空 shuffle 自定义层。
// 该层先加载 learned 1x1x1 projection 的权重，再根据运行时输入的 T,H,W
// 把投影后的通道重新排列到时间和空间维度，用于还原 MAGViT 风格的上采样逻辑。
//
// 除 CPU forward 外，本层还提供 Vulkan 计算路径（forward 的 VkMat 重载）：
// shader 源码以字符串内嵌于 .cpp，在 create_pipeline 阶段用 ncnn 的运行时
// glslang（compile_spirv_module）编译成 SPIR-V，再由 ncnn::Pipeline 加载执行。
// 权重在 upload_model 阶段通过 VkTransfer 一次性上传到 GPU，推理期间零拷贝复用。
#pragma once

#include <layer.h>

class DynamicSpaceTimeShuffle : public ncnn::Layer
{
public:
    DynamicSpaceTimeShuffle();
    ~DynamicSpaceTimeShuffle();

    int load_param(const ncnn::ParamDict& pd) override;
    int load_model(const ncnn::ModelBin& mb) override;
    int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const override;

#if NCNN_VULKAN
    int create_pipeline(const ncnn::Option& opt) override;
    int destroy_pipeline(const ncnn::Option& opt) override;
    int upload_model(ncnn::VkTransfer& cmd, const ncnn::Option& opt) override;
    int forward(const ncnn::VkMat& bottom_blob, ncnn::VkMat& top_blob, ncnn::VkCompute& cmd, const ncnn::Option& opt) const override;
#endif

private:
    int in_channels;
    int projected_channels;
    // temporal_ratio/spatial_ratio 决定通道展开到 T/H/W 的比例。
    int temporal_ratio;
    int spatial_ratio;
    ncnn::Mat bias_data;
    ncnn::Mat weight_data;

#if NCNN_VULKAN
    // Vulkan 计算管线与上传到 GPU 的权重/偏置。权重在 create_pipeline/upload_model
    // 阶段准备好，forward 期间只记录一次 dispatch，不做任何主机侧搬运。
    ncnn::Pipeline* pipeline;
    ncnn::VkMat weight_data_gpu;
    ncnn::VkMat bias_data_gpu;
#endif
};

ncnn::Layer* DynamicSpaceTimeShuffle_layer_creator(void* userdata);
