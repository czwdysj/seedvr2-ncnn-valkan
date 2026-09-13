// 本文件对 SeedVR2 VAE decoder 做独立 Vulkan 性能测试。
// 输入是模型目录与 C=16,T,H,W latent 尺寸；每轮通过 ncnn Extractor 执行完整
// decoder，并输出端到端中位数。该工具用于隔离 VAE decoder 热点，不包含视频
// 编解码、DiT 和采样器；边界上传下载在新旧实现中保持完全一致。
#include <gpu.h>
#include <net.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "layers/dynamic_framewise_group_norm.h"
#include "layers/dynamic_framewise_spatial_attention.h"
#include "layers/dynamic_space_time_shuffle.h"

namespace
{
using Clock = std::chrono::steady_clock;

double milliseconds_since(const Clock::time_point& start)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void register_vae_layers(ncnn::Net& net)
{
    net.register_custom_layer("DynamicFramewiseGroupNorm",
                              DynamicFramewiseGroupNorm_layer_creator);
    net.register_custom_layer("DynamicFramewiseSpatialAttention",
                              DynamicFramewiseSpatialAttention_layer_creator);
    net.register_custom_layer("DynamicSpaceTimeShuffle",
                              DynamicSpaceTimeShuffle_layer_creator);
}
} // namespace

int main(int argc, char** argv)
{
    if (argc != 6)
    {
        std::fprintf(stderr, "usage: %s decoder.param decoder.bin T H W\n", argv[0]);
        return 2;
    }

    const int frames = std::atoi(argv[3]);
    const int height = std::atoi(argv[4]);
    const int width = std::atoi(argv[5]);
    if (frames <= 0 || height <= 0 || width <= 0)
        return 3;

    ncnn::create_gpu_instance();
    const ncnn::VulkanDevice* device = ncnn::get_gpu_device(0);
    if (!device)
        return 4;

    ncnn::Net net;
    net.opt.use_vulkan_compute = true;
    net.opt.use_packing_layout = false;
    net.opt.use_fp16_packed = false;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;
    net.set_vulkan_device(device);
    register_vae_layers(net);
    const int param_result = net.load_param(argv[1]);
    const int model_result = param_result == 0 ? net.load_model(argv[2]) : param_result;
    if (param_result != 0 || model_result != 0)
    {
        std::fprintf(stderr, "load failed: param=%d model=%d\n", param_result, model_result);
        return 5;
    }

    ncnn::Mat input(width, height, frames, 16, 4u, 1);
    size_t logical_index = 0;
    for (int channel = 0; channel < input.c; ++channel)
        for (int frame = 0; frame < input.d; ++frame)
            for (int y = 0; y < input.h; ++y)
            {
                float* row = input.channel(channel).depth(frame).row(y);
                for (int x = 0; x < input.w; ++x, ++logical_index)
                    row[x] = static_cast<float>(static_cast<int>(logical_index % 37) - 18)
                        * 0.001f;
            }

    std::vector<double> samples;
    ncnn::Mat output;
    for (int iteration = 0; iteration < 7; ++iteration)
    {
        ncnn::Extractor extractor = net.create_extractor();
        if (extractor.input(net.input_names()[0], input) != 0)
            return 7;

        const auto start = Clock::now();
        if (extractor.extract(net.output_names()[0], output) != 0)
            return 8;
        const double elapsed = milliseconds_since(start);
        if (iteration >= 2)
            samples.push_back(elapsed);
    }

    std::sort(samples.begin(), samples.end());
    double checksum = 0.0;
    for (int channel = 0; channel < output.c; ++channel)
        for (int frame = 0; frame < output.d; ++frame)
            for (int y = 0; y < output.h; ++y)
            {
                const float* row = output.channel(channel).depth(frame).row(y);
                for (int x = 0; x < output.w; ++x)
                    checksum += row[x];
            }
    std::printf("latent C,T,H,W=16,%d,%d,%d output C,T,H,W=%d,%d,%d,%d\n",
                frames, height, width, output.c, output.d, output.h, output.w);
    std::printf("decoder median_ms=%.3f samples=5 checksum=%.9g\n",
                samples[samples.size() / 2], checksum);

    // Net 持有 Vulkan pipeline，必须在销毁全局 GPU instance 前显式释放。
    net.clear();
    ncnn::destroy_gpu_instance();
    return 0;
}
