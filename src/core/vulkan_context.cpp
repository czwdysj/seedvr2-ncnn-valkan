// 本文件实现 NCNN 运行时选项和 Vulkan 实例的统一管理。
// CPU 构建中请求 Vulkan 会返回明确的 UnsupportedBackend；打开 NCNN_VULKAN 后，
// 这里负责创建全局 GPU 实例并校验设备编号。各模型不直接持有 Vulkan 全局资源。
#include "core/runtime_context.h"

#include <algorithm>
#include <thread>

#if NCNN_VULKAN
#include <gpu.h>
#endif

namespace seedvr2
{
VulkanContext::VulkanContext() = default;

VulkanContext::~VulkanContext()
{
    shutdown();
}

int VulkanContext::initialize(int device_index, std::string& error)
{
#if !SEEDVR2_CUSTOM_LAYERS_VULKAN
    (void)device_index;
    error = "Vulkan is blocked until all six custom layers implement forward_vkcompute";
    return static_cast<int>(Status::UnsupportedBackend);
#elif NCNN_VULKAN
    ncnn::create_gpu_instance();
    const int gpu_count = ncnn::get_gpu_count();
    if (device_index < 0 || device_index >= gpu_count)
    {
        error = "Vulkan device index is out of range";
        ncnn::destroy_gpu_instance();
        return static_cast<int>(Status::UnsupportedBackend);
    }
    initialized_ = true;
    return static_cast<int>(Status::Ok);
#else
    (void)device_index;
    error = "this build has NCNN_VULKAN disabled";
    return static_cast<int>(Status::UnsupportedBackend);
#endif
}

void VulkanContext::shutdown()
{
#if NCNN_VULKAN
    if (initialized_)
        ncnn::destroy_gpu_instance();
#endif
    initialized_ = false;
}

bool VulkanContext::available() const noexcept
{
    return initialized_;
}

int RuntimeContext::initialize(const RuntimeOptions& options, std::string& error)
{
    if (options.sampling_steps <= 0 || options.cfg_scale < 0.0f
        || options.cfg_rescale < 0.0f || options.cfg_rescale > 1.0f
        || options.condition_noise_scale < 0.0f || options.condition_noise_scale > 1.0f
        || options.vae_scaling_factor <= 0.0f)
    {
        error = "invalid runtime sampling or scaling option";
        return static_cast<int>(Status::InvalidArgument);
    }

    options_ = options;
    if (options_.num_threads <= 0)
        options_.num_threads = std::max(1u, std::thread::hardware_concurrency());

    if (options_.device == DeviceType::Vulkan)
    {
        const int result = vulkan_.initialize(options_.vulkan_device_index, error);
        if (result != 0)
            return result;
    }

    ncnn_option_.num_threads = options_.num_threads;
    ncnn_option_.use_vulkan_compute = options_.device == DeviceType::Vulkan;
    // 当前 CPU 对齐使用标量布局。Vulkan layer 完成后可在这里统一打开 packing。
    ncnn_option_.use_packing_layout = false;
    ncnn_option_.use_fp16_packed = options_.use_fp16_storage;
    ncnn_option_.use_fp16_storage = options_.use_fp16_storage;
    ncnn_option_.use_fp16_arithmetic = options_.use_fp16_arithmetic;
    ncnn_option_.lightmode = true;
    return static_cast<int>(Status::Ok);
}

void RuntimeContext::configure(ncnn::Net& net) const
{
    net.opt = ncnn_option_;
#if NCNN_VULKAN
    if (options_.device == DeviceType::Vulkan)
        net.set_vulkan_device(options_.vulkan_device_index);
#endif
}
} // namespace seedvr2
