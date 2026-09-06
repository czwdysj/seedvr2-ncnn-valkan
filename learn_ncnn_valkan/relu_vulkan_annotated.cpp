// 本文件逐行讲解 NCNN ReLU Vulkan 后端在 CPU 侧完成的调度工作。
// 输入是已经位于 GPU buffer 中的 VkMat，输出原位写回同一个 VkMat；真正的逐元素
// 运算位于 relu_annotated.comp，本文件负责创建 Pipeline、绑定资源并记录 dispatch。
#include "relu_vulkan_annotated.h"

// LayerShaderType 是 NCNN 构建阶段为内置 shader 生成的枚举表。
#include "layer_shader_type.h"

// 实现仍放在 ncnn 命名空间，才能直接使用 Option、Pipeline 和 VkMat。
namespace ncnn {

// 构造 Vulkan layer，但此时还没有创建真正的 VkPipeline。
ReLU_vulkan_annotated::ReLU_vulkan_annotated()
{
    // 告诉 Net：这个 layer 可以接收 VkMat 并在 Vulkan 后端执行。
    support_vulkan = true;

    // 告诉 NCNN：shader 能正确处理 Vulkan packing 布局。
    support_vulkan_packing = true;

    // 资源指针必须先置空，便于失败路径和 destroy_pipeline() 安全处理。
    pipeline_relu = 0;
}

// 模型加载完成后调用一次，创建后续 forward 可以复用的 Pipeline。
int ReLU_vulkan_annotated::create_pipeline(const Option& opt)
{
    // top_shapes 是 NCNN 提供的静态形状提示；动态形状时它可能为空。
    const Mat& shape = top_shapes.empty() ? Mat() : top_shapes[0];

    // 两项 specialization constant 必须与 shader 中 constant_id 的顺序一致。
    std::vector<vk_specialization_type> specializations(2);

    // constant_id=0：ReLU 的 slope；0 表示普通 ReLU，非 0 表示 LeakyReLU。
    specializations[0].f = slope;

    // constant_id=1：静态形状已知时，把 vec4 数量编译进 Pipeline。
    // 动态形状下该值为 0，shader 会通过 push constant 读取本次 forward 的 n。
    specializations[1].u32 = shape.total() * shape.elempack / 4;

    // 使用设备 subgroup 宽度作为一维 workgroup 大小的起点。
    const int local_size_x = vkdev->info.subgroup_size();

    // Pipeline 必须绑定当前 layer 所属的 VulkanDevice。
    pipeline_relu = new Pipeline(vkdev);

    // 本算子是一维逐元素计算，因此 y/z 方向 workgroup 大小设为 1。
    pipeline_relu->set_optimal_local_size_xyz(local_size_x, 1, 1);

    // 教学副本对应 NCNN 的 relu shader；正式源码用 LayerShaderType::relu 查找 SPIR-V。
    pipeline_relu->create(LayerShaderType::relu, opt, specializations);

    // NCNN 约定返回 0 表示成功。
    return 0;
}

// Net 销毁或重新创建 Pipeline 时调用，释放 Vulkan 相关资源。
int ReLU_vulkan_annotated::destroy_pipeline(const Option& /*opt*/)
{
    // Pipeline 析构函数负责释放其持有的 Vulkan 对象。
    delete pipeline_relu;

    // 清空悬空指针，防止重复释放。
    pipeline_relu = 0;

    // 返回 0 表示销毁成功。
    return 0;
}

// 每次推理调用：这里只组织并记录 GPU 命令，不在 CPU 上计算 ReLU。
int ReLU_vulkan_annotated::forward_inplace(
    VkMat& bottom_top_blob,
    VkCompute& cmd,
    const Option& /*opt*/) const
{
    // total()*elempack 是标量总数；shader 每次处理 vec4，所以除以 4。
    const size_t n = bottom_top_blob.total() * bottom_top_blob.elempack / 4;

    // bindings 的下标必须对应 shader 的 layout(binding=N)。
    std::vector<VkMat> bindings(1);

    // binding=0 同时用于读取和写入，实现真正的原位计算。
    bindings[0] = bottom_top_blob;

    // push constants 保存每次 forward 都可能变化的小型运行时参数。
    std::vector<vk_constant_type> constants(1);

    // 与 shader 中 parameter.n 对应，告诉 shader 本次有多少个 vec4。
    constants[0].u32 = n;

    // NCNN 使用一个轻量 VkMat 描述 dispatch 的逻辑工作量。
    VkMat dispatcher;

    // 一维任务数量是 n；Pipeline 会结合 local_size_x 换算 workgroup 数量。
    dispatcher.w = n;

    // ReLU 不需要二维和三维调度。
    dispatcher.h = 1;

    // c 同样设为 1，使总工作量保持一维。
    dispatcher.c = 1;

    // 把 Pipeline、descriptor bindings、push constants 和 dispatch 记录进 VkCompute。
    cmd.record_pipeline(pipeline_relu, bindings, constants, dispatcher);

    // 这里只代表命令记录成功；实际 GPU 完成时间由外层提交和同步控制。
    return 0;
}

// 结束 ncnn 命名空间。
} // namespace ncnn
