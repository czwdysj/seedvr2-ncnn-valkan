// 本文件声明 SeedVR2 VAE bottleneck 中的动态逐帧空间自注意力层。
// 输入输出使用 ncnn::Mat(w=W,h=H,d=T,c=C)；layer 内部拥有原始 PyTorch
// GroupNorm、Q/K/V 投影和输出投影权重，按每一帧独立执行 H*W token 注意力。
//
// 除 CPU forward 外，本层还提供 Vulkan 计算路径（forward 的 VkMat 重载）。
// 注意力是「先归约、再归一化、再矩阵乘」的多段式算子，拆成五个 compute shader：
//   norm_reduce    —— 统计每个 (帧, 组) 的 sum/sq_sum（复用 GroupNorm 的 reduce 思路）
//   norm_apply     —— 逐元素 GroupNorm 归一化，产出 normalized workspace
//   qkv_projection —— Q/K/V 三个线性投影（同一 shader 三次 dispatch，复用 pipeline）
//   attention      —— softmax(QK^T / sqrt(C)) @ V，每个 work item 一个 query token
//   output_projection —— 输出投影 + 残差连接，写回 top_blob
// 全部 shader 以字符串内嵌，create_pipeline 阶段用 ncnn 运行时 glslang 编译成
// SPIR-V；统计量与中间激活用 fp32 workspace 暂存，权重在 upload_model 一次性上传。
#pragma once

#include <layer.h>

class DynamicFramewiseSpatialAttention : public ncnn::Layer
{
public:
    DynamicFramewiseSpatialAttention();
    ~DynamicFramewiseSpatialAttention();

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

#if NCNN_VULKAN
    // 五个 compute shader 对应的管线。
    ncnn::Pipeline* pipeline_norm_reduce;
    ncnn::Pipeline* pipeline_norm_apply;
    ncnn::Pipeline* pipeline_qkv_projection;
    ncnn::Pipeline* pipeline_attention;
    ncnn::Pipeline* pipeline_output_projection;
    // 全部权重/偏置在 upload_model 阶段上传 GPU，forward 期间只引用 buffer。
    ncnn::VkMat norm_weight_gpu;
    ncnn::VkMat norm_bias_gpu;
    ncnn::VkMat q_weight_gpu;
    ncnn::VkMat q_bias_gpu;
    ncnn::VkMat k_weight_gpu;
    ncnn::VkMat k_bias_gpu;
    ncnn::VkMat v_weight_gpu;
    ncnn::VkMat v_bias_gpu;
    ncnn::VkMat out_weight_gpu;
    ncnn::VkMat out_bias_gpu;
#endif
};

ncnn::Layer* DynamicFramewiseSpatialAttention_layer_creator(void* userdata);
