// 本文件声明 SeedVR2 动态 VAE 组件。
// Encoder 接收 C,T,H,W 视频并输出经 posterior 采样和 scaling_factor 缩放的
// 16 通道 latent；Decoder 执行逆缩放并还原视频。两个 NCNN 图共享运行时选项，
// 注册三种动态 VAE 自定义层，输入 T/H/W 在模型约束内均由运行时决定。
#pragma once

#include <net.h>

#include <memory>
#include <random>
#include <string>

#include "runtime_context.h"

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

    const std::string& last_error() const noexcept { return last_error_; }

private:
    std::unique_ptr<ncnn::Net> encoder_;
    std::unique_ptr<ncnn::Net> decoder_;
    std::string last_error_;
};
} // namespace seedvr2
