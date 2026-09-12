// e2e_profile.cpp —— SeedVR2 端到端推理测试（小参数），配合 SEEDVR2_PROFILE=1
// 输出分阶段耗时（preprocess / vae_encode / dit / vae_decode / postprocess）。
//
// 用法：
//   SEEDVR2_PROFILE=1 ./seedvr2_e2e_profile <model_dir> [T H W] [tokens]
//
// 默认小参数：T=5（满足 (T-1)%4==0 的最小多帧数）、H=W=64、text tokens=8。
#include "seedvr2/engine.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::fprintf(stderr, "usage: %s <model_dir> [T H W] [text_tokens]\n", argv[0]);
        return 2;
    }
    const std::string model_dir = argv[1];
    const int T = argc > 2 ? std::atoi(argv[2]) : 5;
    const int H = argc > 3 ? std::atoi(argv[3]) : 64;
    const int W = argc > 4 ? std::atoi(argv[4]) : 64;
    const int tokens = argc > 5 ? std::atoi(argv[5]) : 8;

    // 构造输入视频：THWC、RGB、FP32、[0,1]，随机值。
    seedvr2::Video input;
    input.frames = T;
    input.height = H;
    input.width = W;
    input.channels = 3;
    input.data.resize(static_cast<std::size_t>(T) * H * W * 3);
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (float& v : input.data)
        v = dist(rng);

    // 构造文本 embedding：[tokens, 5120]。
    seedvr2::TextEmbedding positive;
    positive.tokens = tokens;
    positive.channels = 5120;
    positive.data.resize(static_cast<std::size_t>(tokens) * 5120);
    for (float& v : positive.data)
        v = dist(rng);
    seedvr2::TextEmbedding negative;
    negative.tokens = tokens;
    negative.channels = 5120;
    negative.data.resize(static_cast<std::size_t>(tokens) * 5120);
    for (float& v : negative.data)
        v = dist(rng);

    seedvr2::RuntimeOptions opts;
    const char* device = std::getenv("SEEDVR2_DEVICE");
    opts.device = device && std::strcmp(device, "cpu") == 0
        ? seedvr2::DeviceType::Cpu
        : seedvr2::DeviceType::Vulkan;
    opts.num_threads = 8;
    const char* steps = std::getenv("SEEDVR2_STEPS");
    const char* cfg_scale = std::getenv("SEEDVR2_CFG_SCALE");
    const char* cfg_rescale = std::getenv("SEEDVR2_CFG_RESCALE");
    opts.sampling_steps = steps ? std::max(1, std::atoi(steps)) : 1;
    opts.cfg_scale = cfg_scale ? std::strtof(cfg_scale, nullptr) : 1.0f;
    opts.cfg_rescale = cfg_rescale ? std::strtof(cfg_rescale, nullptr) : 0.0f;
    opts.seed = 666;
    // 通过环境变量 SEEDVR2_DIT_RESIDENT=1 切换常驻模式（默认流式）。
    opts.dit_resident = std::getenv("SEEDVR2_DIT_RESIDENT") != nullptr;

    seedvr2::SeedVR2Engine engine;
    const int load_ret = engine.load(model_dir, opts);
    if (load_ret != 0)
    {
        std::fprintf(stderr, "load failed (%d): %s\n", load_ret, engine.last_error().c_str());
        return 1;
    }
    std::printf("model loaded OK\n");

    seedvr2::Video output;
    const int ret = engine.process(input, positive, negative, output);
    if (ret != 0)
    {
        std::fprintf(stderr, "process failed (%d): %s\n", ret, engine.last_error().c_str());
        return 1;
    }
    std::printf("process OK: output %dx%dx%d frames=%d channels=%d\n",
                output.width, output.height, output.frames, output.frames, output.channels);
    const char* output_path = std::getenv("SEEDVR2_OUTPUT_RAW");
    if (output_path)
    {
        std::ofstream stream(output_path, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(output.data.data()),
                     static_cast<std::streamsize>(output.data.size() * sizeof(float)));
        if (!stream)
        {
            std::fprintf(stderr, "failed to write output: %s\n", output_path);
            return 1;
        }
    }
    return 0;
}
