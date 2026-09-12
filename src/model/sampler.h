// 本文件声明 SeedVR2 使用的采样策略接口和 Euler 实现。
// 张量保持 NCNN C,T,H,W 布局；CFG 实现正负分支融合，Euler 按 lerp schedule
// 和 v_lerp prediction 更新 latent。该模块不加载模型，因此可独立做确定性单元测试。
#pragma once

#include <net.h>

#include <string>
#include <vector>

#if NCNN_VULKAN
#include "core/runtime_context.h"
#endif

namespace seedvr2
{
class EulerSampler
{
public:
    explicit EulerSampler(int steps);
    ~EulerSampler();

    const std::vector<float>& timesteps() const noexcept { return timesteps_; }
    int apply_cfg(const ncnn::Mat& positive,
                  const ncnn::Mat& negative,
                  float scale,
                  float rescale,
                  ncnn::Mat& output,
                  std::string& error) const;
    int step_to(const ncnn::Mat& prediction,
                const ncnn::Mat& current,
                float timestep,
                float next_timestep,
                ncnn::Mat& output,
                std::string& error) const;
    int endpoint(const ncnn::Mat& prediction,
                 const ncnn::Mat& current,
                 float timestep,
                 ncnn::Mat& output,
                 std::string& error) const;

#if NCNN_VULKAN
    int initialize_vulkan(const RuntimeContext& context, std::string& error);
    int make_condition_vulkan(const ncnn::VkMat& encoded,
                              const ncnn::VkMat& augment_noise,
                              float ratio,
                              VulkanExecutionContext& execution,
                              ncnn::VkCompute& command,
                              ncnn::VkMat& output,
                              std::string& error) const;
    int make_dit_input_vulkan(const ncnn::VkMat& latent,
                              const ncnn::VkMat& condition,
                              VulkanExecutionContext& execution,
                              ncnn::VkCompute& command,
                              ncnn::VkMat& output,
                              std::string& error) const;
    int apply_cfg_vulkan(const ncnn::VkMat& positive,
                         const ncnn::VkMat& negative,
                         float scale,
                         float rescale,
                         VulkanExecutionContext& execution,
                         ncnn::VkCompute& command,
                         ncnn::VkMat& output,
                         std::string& error) const;
    int step_vulkan(const ncnn::VkMat& prediction,
                    const ncnn::VkMat& current,
                    float timestep,
                    float next_timestep,
                    VulkanExecutionContext& execution,
                    ncnn::VkCompute& command,
                    ncnn::VkMat& output,
                    std::string& error) const;
#endif

private:
    std::vector<float> timesteps_;
#if NCNN_VULKAN
    ncnn::Pipeline* pipeline_condition_ = nullptr;
    ncnn::Pipeline* pipeline_dit_input_ = nullptr;
    ncnn::Pipeline* pipeline_cfg_ = nullptr;
    ncnn::Pipeline* pipeline_cfg_stats_ = nullptr;
    ncnn::Pipeline* pipeline_cfg_rescale_ = nullptr;
    ncnn::Pipeline* pipeline_euler_ = nullptr;
#endif
};
} // namespace seedvr2
