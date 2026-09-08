// 本文件声明 SeedVR2 VAE 使用的动态逐帧 GroupNorm 自定义层。
// 输入和输出都按 ncnn::Mat(w=W,h=H,d=T,c=C) 表示视频 latent；layer 会在运行时
// 根据 T,H,W 计算每一帧的分组均值和方差，并加载 pnnx 打包进模型的仿射权重。
//
// 除 CPU forward 外，本层还提供 Vulkan 计算路径（forward 的 VkMat 重载）：
// 两个 compute shader（reduce 统计 + normalize 应用）以字符串内嵌于 .cpp，在
// create_pipeline 阶段用 ncnn 的运行时 glslang（compile_spirv_module）编译成
// SPIR-V。统计量（sum/sq_sum）以 fp32 workspace 暂存，权重在 upload_model 阶段
// 通过 VkTransfer 一次性上传到 GPU，推理期间零拷贝复用。
//
// 与 ncnn 原生 GroupNorm 的区别：原生在 dims==4 时把 T 折叠进空间维度做「跨帧」
// 统计；本层按 PyTorch VAE 语义要求「逐帧」统计，因此 mean/var 是 per-(帧, 组)，
// 而非 per-组。这正是该自定义层存在的根本原因。
#pragma once

#include <layer.h>

class DynamicFramewiseGroupNorm : public ncnn::Layer
{
public:
    DynamicFramewiseGroupNorm();
    ~DynamicFramewiseGroupNorm();

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
    // channels/groups/eps 对应 PyTorch GroupNorm 的通道数、组数和数值稳定项。
    int channels;
    int groups;
    float eps;
    // pnnx 将自定义模块的 tensor 属性写入 bin，本 layer 负责按约定顺序读取。
    ncnn::Mat bias_data;
    ncnn::Mat weight_data;

#if NCNN_VULKAN
    // reduce：统计每个 (帧, 组) 的 sum 与 sq_sum；normalize：应用 mean/var 归一化。
    ncnn::Pipeline* pipeline_reduce;
    ncnn::Pipeline* pipeline_normalize;
    // 仿射权重/偏置在 upload_model 阶段上传 GPU，forward 期间只引用 buffer。
    ncnn::VkMat weight_data_gpu;
    ncnn::VkMat bias_data_gpu;
#endif
};

ncnn::Layer* DynamicFramewiseGroupNorm_layer_creator(void* userdata);
