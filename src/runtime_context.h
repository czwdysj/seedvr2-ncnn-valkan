// 本文件声明内部运行时上下文，集中保存 CPU/Vulkan、线程数和精度选项。
// 各模型组件只从该上下文获得一致的 ncnn::Option，避免 VAE、DiT 和测试入口
// 分别设置线程或精度而产生不可复现的行为。Vulkan 生命周期由 VulkanContext 管理。
#pragma once

#include <net.h>

#include <string>

#include "seedvr2/engine.h"

namespace seedvr2
{
class VulkanContext
{
public:
    VulkanContext();
    ~VulkanContext();

    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;

    int initialize(int device_index, std::string& error);
    void shutdown();
    bool available() const noexcept;

private:
    bool initialized_ = false;
};

class RuntimeContext
{
public:
    int initialize(const RuntimeOptions& options, std::string& error);
    void configure(ncnn::Net& net) const;

    const RuntimeOptions& options() const noexcept { return options_; }
    const ncnn::Option& ncnn_option() const noexcept { return ncnn_option_; }

private:
    RuntimeOptions options_;
    ncnn::Option ncnn_option_;
    VulkanContext vulkan_;
};
} // namespace seedvr2
