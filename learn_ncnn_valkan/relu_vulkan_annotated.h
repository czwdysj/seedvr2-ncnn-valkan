// 本文件是 NCNN ReLU Vulkan 后端头文件的教学副本。
// 输入和输出是同一个 GPU VkMat，layer 在原位执行 ReLU 或 LeakyReLU；该文件只展示
// Vulkan layer 需要声明的 Pipeline 生命周期和 GPU forward 接口，不参与 SeedVR2 正式构建。
#ifndef LEARN_NCNN_RELU_VULKAN_ANNOTATED_H
#define LEARN_NCNN_RELU_VULKAN_ANNOTATED_H

// 继承 CPU ReLU，直接复用 slope 参数、load_param() 和 CPU fallback 行为。
#include "relu.h"

// NCNN 的所有内置 layer 都位于 ncnn 命名空间。
namespace ncnn {

// Vulkan 后端使用“CPU 基类名 + _vulkan”的命名约定。
class ReLU_vulkan_annotated : public ReLU
{
public:
    // 构造函数声明 Vulkan/packing 能力，并把资源指针初始化为空。
    ReLU_vulkan_annotated();

    // 根据模型参数和 Option 创建 Vulkan compute pipeline。
    virtual int create_pipeline(const Option& opt);

    // 销毁 create_pipeline() 分配的 Vulkan pipeline。
    virtual int destroy_pipeline(const Option& opt);

    // 避免下面的 VkMat 重载隐藏基类中同名的 CPU forward_inplace(Mat, Option)。
    using ReLU::forward_inplace;

    // VkMat 是 GPU buffer；VkCompute 用于记录命令，本函数不会直接执行 CPU 循环。
    virtual int forward_inplace(
        VkMat& bottom_top_blob,
        VkCompute& cmd,
        const Option& opt) const;

public:
    // Pipeline 封装 VkShaderModule、descriptor layout 和 VkPipeline 等 Vulkan 对象。
    Pipeline* pipeline_relu;
};

// 结束 ncnn 命名空间。
} // namespace ncnn

// 结束头文件防重复包含保护。
#endif // LEARN_NCNN_RELU_VULKAN_ANNOTATED_H
