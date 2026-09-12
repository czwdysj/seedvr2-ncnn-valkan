// 本文件声明内部运行时上下文，集中保存 CPU/Vulkan、线程数和精度选项。
// 各模型组件只从该上下文获得一致的 ncnn::Option，避免 VAE、DiT 和测试入口
// 分别设置线程或精度而产生不可复现的行为。Vulkan 生命周期由 VulkanContext 管理。
#pragma once

#include <net.h>

#if NCNN_VULKAN
#include <command.h>
#include <gpu.h>
#endif

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

#if NCNN_VULKAN
// 一次 Engine::process 期间共享的 Vulkan 张量执行上下文。
// 它持有同一设备上的 blob/staging allocator，使 VAE、DiT 和 sampler 返回的
// VkMat 可以跨组件存活；上下文销毁前，调用方必须确保已提交的 GPU 命令完成。
class VulkanExecutionContext
{
public:
    explicit VulkanExecutionContext(const RuntimeContext& runtime);
    ~VulkanExecutionContext();

    VulkanExecutionContext(const VulkanExecutionContext&) = delete;
    VulkanExecutionContext& operator=(const VulkanExecutionContext&) = delete;

    bool valid() const noexcept;
    const ncnn::VulkanDevice* device() const noexcept { return device_; }
    ncnn::VkAllocator* blob_allocator() const noexcept { return blob_allocator_; }
    ncnn::VkAllocator* staging_allocator() const noexcept { return staging_allocator_; }
    ncnn::Option option() const;
    void configure(ncnn::Extractor& extractor) const;

private:
    const RuntimeContext& runtime_;
    const ncnn::VulkanDevice* device_ = nullptr;
    ncnn::VkAllocator* blob_allocator_ = nullptr;
    ncnn::VkAllocator* staging_allocator_ = nullptr;
};
#endif
} // namespace seedvr2
