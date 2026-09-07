// 本文件对不依赖大权重的核心调度数学进行快速单元测试。
// 覆盖 Video/Text 数据契约、16 倍数中心裁剪、4n+1 补帧、后处理裁帧、CFG 和
// Euler endpoint；这些测试用于在每次重构后先发现布局或公式错误，再跑大模型。
#include "postprocessing.h"
#include "preprocessing.h"
#include "sampler.h"
#include "seedvr2/engine.h"

#include <cmath>
#include <iostream>
#include <string>

namespace
{
bool close(float a, float b, float tolerance = 1e-6f)
{
    return std::abs(a - b) <= tolerance;
}

int fail(const char* message)
{
    std::cerr << message << '\n';
    return 1;
}
} // namespace

int main()
{
    seedvr2::Video input;
    input.frames = 3;
    input.height = 18;
    input.width = 34;
    input.channels = 3;
    input.data.resize(static_cast<std::size_t>(3 * 18 * 34 * 3));
    for (std::size_t index = 0; index < input.data.size(); ++index)
        input.data[index] = static_cast<float>(index % 255) / 255.0f;
    if (!input.valid())
        return fail("Video::valid rejected a valid tensor");

    seedvr2::PreparedVideo prepared;
    std::string error;
    if (seedvr2::preprocess_video(input, prepared, error) != 0)
        return fail(error.c_str());
    if (prepared.tensor.w != 32 || prepared.tensor.h != 16 || prepared.tensor.d != 5
        || prepared.tensor.c != 3)
        return fail("preprocess shape is not C=3,T=5,H=16,W=32");
    const float first_padded = prepared.tensor.channel(0).depth(2).row(0)[0];
    const float last_padded = prepared.tensor.channel(0).depth(4).row(0)[0];
    if (!close(first_padded, last_padded))
        return fail("temporal padding did not replicate the final frame");

    ncnn::Mat decoded(32, 16, 5, 3, 4u, 1);
    decoded.fill(0.0f);
    seedvr2::Video restored;
    if (seedvr2::postprocess_video(decoded, prepared, restored, error) != 0)
        return fail(error.c_str());
    if (restored.frames != 3 || restored.height != 16 || restored.width != 32
        || !close(restored.data.front(), 0.5f))
        return fail("postprocess did not crop frames or map values correctly");

    ncnn::Mat positive(2, 1, 1, 1, 4u, 1);
    ncnn::Mat negative(2, 1, 1, 1, 4u, 1);
    positive.fill(2.0f);
    negative.fill(0.0f);
    seedvr2::EulerSampler sampler(1);
    ncnn::Mat cfg;
    if (sampler.apply_cfg(positive, negative, 1.5f, 0.0f, cfg, error) != 0
        || !close(cfg.channel(0).depth(0).row(0)[0], 3.0f))
        return fail("CFG formula mismatch");
    ncnn::Mat current(2, 1, 1, 1, 4u, 1);
    ncnn::Mat prediction(2, 1, 1, 1, 4u, 1);
    current.fill(1.0f);
    prediction.fill(0.25f);
    ncnn::Mat endpoint;
    if (sampler.endpoint(prediction, current, 1000.0f, endpoint, error) != 0
        || !close(endpoint.channel(0).depth(0).row(0)[0], 0.75f))
        return fail("Euler v_lerp endpoint mismatch");

    seedvr2::SeedVR2Engine engine;
    if (engine.load("/path/that/does/not/exist")
        != static_cast<int>(seedvr2::Status::ModelNotFound))
        return fail("Engine did not report a missing model directory");

    seedvr2::RuntimeOptions vulkan_options;
    vulkan_options.device = seedvr2::DeviceType::Vulkan;
#if NCNN_VULKAN
    // 六个自定义层均已实现 forward_vkcompute，Vulkan 后端应通过能力闸门，
    // 在模型目录缺失时返回 ModelNotFound（而非 UnsupportedBackend）。
    if (engine.load("/path/that/does/not/exist", vulkan_options)
        != static_cast<int>(seedvr2::Status::ModelNotFound))
        return fail("Engine did not accept the Vulkan backend after the capability gate opened");
#else
    // 未开启 Vulkan 的构建仍需拒绝 Vulkan 后端。
    if (engine.load("/path/that/does/not/exist", vulkan_options)
        != static_cast<int>(seedvr2::Status::UnsupportedBackend))
        return fail("Engine did not reject the Vulkan backend in a CPU-only build");
#endif

    std::cout << "runtime unit tests passed\n";
    return 0;
}
