// 本文件声明 SeedVR2 动态 VAE 组件。
// Encoder 接收 C,T,H,W 视频并输出经 posterior 采样和 scaling_factor 缩放的
// 16 通道 latent；Decoder 执行逆缩放并还原视频。CPU 接口使用 C,T,H,W Mat；
// Vulkan 接口直接连接 token-major sampler VkMat，避免 moments/latent 主机往返。
#pragma once

#include <net.h>

#include <memory>
#include <random>
#include <string>

#include "core/runtime_context.h"

namespace seedvr2
{
class SeedVR2VAE
{
public:
    SeedVR2VAE();
    ~SeedVR2VAE();

    int load(const std::string& model_dir, const RuntimeContext& context);
    int encode(const ncnn::Mat& video,
               std::mt19937_64& random,
               bool stochastic,
               float scaling_factor,
               ncnn::Mat& latent);
    int decode(const ncnn::Mat& latent, float scaling_factor, ncnn::Mat& video);

#if NCNN_VULKAN
    // Encoder 直接输出 token-major [T*H*W,16] latent；posterior_noise 使用相同布局。
    int encode_vkmat(const ncnn::VkMat& video,
                     const ncnn::VkMat& posterior_noise,
                     bool stochastic,
                     float scaling_factor,
                     VulkanExecutionContext& execution,
                     ncnn::VkMat& latent);
    // Decoder 接收 token-major latent，在 GPU 上恢复 C,T,H,W 并执行逆 scaling。
    int decode_vkmat(const ncnn::VkMat& latent,
                     int frames,
                     int height,
                     int width,
                     float scaling_factor,
                     VulkanExecutionContext& execution,
                     ncnn::VkMat& video);
#endif

    const std::string& last_error() const noexcept { return last_error_; }

private:
    std::unique_ptr<ncnn::Net> encoder_;
    std::unique_ptr<ncnn::Net> decoder_;
    std::string last_error_;
#if NCNN_VULKAN
    ncnn::Pipeline* pipeline_posterior_ = nullptr;
    ncnn::Pipeline* pipeline_decode_layout_ = nullptr;
#endif
};
} // namespace seedvr2
