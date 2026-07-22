// 本文件定义 SeedVR2 NCNN Runtime 的唯一稳定公开接口。
// 调用方只需提供 THWC、FP32、[0,1] 范围的视频和预计算文本 embedding，
// Engine 内部负责视频对齐、VAE、33 通道条件构造、DiT、CFG/Euler 与后处理。
// 头文件使用 PIMPL 隐藏 NCNN 和自定义层，业务项目无需依赖任何内部实现头文件。
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace seedvr2
{
enum class DeviceType
{
    Cpu,
    Vulkan,
};

enum class Status
{
    Ok = 0,
    InvalidArgument = -1,
    NotLoaded = -2,
    ModelNotFound = -3,
    ModelLoadFailed = -4,
    InferenceFailed = -5,
    UnsupportedBackend = -6,
    OutOfMemory = -7,
    IoError = -8,
};

struct RuntimeOptions
{
    DeviceType device = DeviceType::Cpu;
    int num_threads = 0;
    int vulkan_device_index = 0;
    int sampling_steps = 1;
    float cfg_scale = 1.0f;
    float cfg_rescale = 0.0f;
    float condition_noise_scale = 0.0f;
    float vae_scaling_factor = 0.9152f;
    std::uint64_t seed = 666;
    bool stochastic_vae = true;
    bool use_fp16_storage = false;
    bool use_fp16_arithmetic = false;
};

struct Video
{
    // data 按 T,H,W,C 连续存放；当前 SeedVR2 3B 超分入口要求 channels=3。
    int frames = 0;
    int height = 0;
    int width = 0;
    int channels = 3;
    float fps = 24.0f;
    std::vector<float> data;

    bool valid() const noexcept;
};

struct TextEmbedding
{
    // data 按 tokens,channels 连续存放；SeedVR2 3B 的 channels 固定为 5120。
    int tokens = 0;
    int channels = 5120;
    std::vector<float> data;

    bool valid() const noexcept;
};

class SeedVR2Engine
{
public:
    SeedVR2Engine();
    ~SeedVR2Engine();

    SeedVR2Engine(SeedVR2Engine&&) noexcept;
    SeedVR2Engine& operator=(SeedVR2Engine&&) noexcept;
    SeedVR2Engine(const SeedVR2Engine&) = delete;
    SeedVR2Engine& operator=(const SeedVR2Engine&) = delete;

    int load(const std::string& model_dir, const RuntimeOptions& options = {});
    int process(const Video& input,
                const TextEmbedding& positive,
                const TextEmbedding& negative,
                Video& output);

    bool loaded() const noexcept;
    const std::string& last_error() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

const char* status_message(Status status) noexcept;
} // namespace seedvr2
