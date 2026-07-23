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
    // 构造时只创建一个空的内部实现对象，不加载任何模型文件。
    // 真正的 NCNN Net、采样器和运行时选项都在 load() 成功后才可用。
    SeedVR2Engine();

    // 析构函数定义在 .cpp 中，避免公开头文件需要看到 Impl 的完整定义。
    ~SeedVR2Engine();

    // Engine 持有模型和运行时状态，禁止复制以避免多个对象共享同一套 NCNN 资源。
    // 允许移动，便于调用方把已加载的 Engine 放入容器或从工厂函数返回。
    SeedVR2Engine(SeedVR2Engine&&) noexcept;
    SeedVR2Engine& operator=(SeedVR2Engine&&) noexcept;
    SeedVR2Engine(const SeedVR2Engine&) = delete;
    SeedVR2Engine& operator=(const SeedVR2Engine&) = delete;

    // 加载模型目录并初始化运行时。model_dir 可以是总目录，也可以直接指向包含
    // VAE/DiT param 和 bin 的目录；实现会优先查找 vae_dynamic 和 dit_full_fp16 子目录。
    // 返回 0 表示成功，非 0 对应 Status；详细错误可通过 last_error() 读取。
    int load(const std::string& model_dir, const RuntimeOptions& options = {});

    // 执行一次 SeedVR2 处理：
    // input: THWC、RGB、FP32、[0,1] 视频；
    // positive/negative: 预计算文本 embedding，布局为 [tokens,5120]；
    // output: 成功后写入 THWC、RGB、FP32、[0,1] 的结果视频。
    int process(const Video& input,
                const TextEmbedding& positive,
                const TextEmbedding& negative,
                Video& output);

    // 返回当前对象是否已经成功 load()。process() 前应为 true。
    bool loaded() const noexcept;

    // 返回最近一次 load()/process() 失败留下的阶段化错误信息。
    const std::string& last_error() const noexcept;

private:
    // PIMPL：把 NCNN、VAE、DiT、自定义层和采样器等内部实现细节从公开 ABI 中隔离。
    class Impl;
    std::unique_ptr<Impl> impl_;
};

const char* status_message(Status status) noexcept;
} // namespace seedvr2
