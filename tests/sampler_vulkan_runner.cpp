// 本文件验证 Engine 多步采样所需的 Vulkan 张量算子。
// 输入是 token-major、pack1 的 [tokens,16] VkMat，依次检查 condition 混合、
// 33 通道 DiT 拼接、CFG rescale 和 Euler 更新；结果下载后与 CPU 公式逐元素比较。
#include "core/runtime_context.h"
#include "model/sampler.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

namespace
{
bool close_mat(const ncnn::Mat& actual, const ncnn::Mat& expected, float tolerance)
{
    if (actual.w != expected.w || actual.h != expected.h)
        return false;
    const float* a = actual;
    const float* b = expected;
    for (int i = 0; i < actual.w * actual.h; ++i)
        if (std::abs(a[i] - b[i]) > tolerance)
            return false;
    return true;
}

int upload_pack1(const ncnn::Mat& source,
                 seedvr2::VulkanExecutionContext& execution,
                 ncnn::VkMat& output)
{
    ncnn::VkCompute command(execution.device());
    ncnn::VkMat packed;
    command.record_upload(source, packed, execution.option());
    execution.device()->convert_packing(packed, output, 1, command, execution.option());
    return command.submit_and_wait();
}

int download(const ncnn::VkMat& source,
             seedvr2::VulkanExecutionContext& execution,
             ncnn::Mat& output)
{
    ncnn::VkCompute command(execution.device());
    command.record_download(source, output, execution.option());
    return command.submit_and_wait();
}
} // namespace

int main()
{
    seedvr2::RuntimeOptions options;
    options.device = seedvr2::DeviceType::Vulkan;
    seedvr2::RuntimeContext runtime;
    std::string error;
    if (runtime.initialize(options, error) != 0)
    {
        std::cerr << error << '\n';
        return 1;
    }
    seedvr2::VulkanExecutionContext execution(runtime);
    seedvr2::EulerSampler sampler(2);
    if (!execution.valid() || sampler.initialize_vulkan(runtime, error) != 0)
    {
        std::cerr << error << '\n';
        return 2;
    }

    constexpr int tokens = 7;
    ncnn::Mat positive(16, tokens, static_cast<size_t>(4u), 1);
    ncnn::Mat negative(16, tokens, static_cast<size_t>(4u), 1);
    for (int i = 0; i < 16 * tokens; ++i)
    {
        static_cast<float*>(positive.data)[i] = (i % 13 - 6) * 0.07f;
        static_cast<float*>(negative.data)[i] = (i % 9 - 4) * 0.05f;
    }
    ncnn::VkMat positive_gpu;
    ncnn::VkMat negative_gpu;
    if (upload_pack1(positive, execution, positive_gpu) != 0
        || upload_pack1(negative, execution, negative_gpu) != 0)
        return 3;

    ncnn::VkCompute command(execution.device());
    ncnn::VkMat condition_gpu;
    ncnn::VkMat dit_input_gpu;
    ncnn::VkMat cfg_gpu;
    ncnn::VkMat next_gpu;
    if (sampler.make_condition_vulkan(positive_gpu, negative_gpu, 0.25f,
                                      execution, command, condition_gpu, error) != 0
        || sampler.make_dit_input_vulkan(positive_gpu, condition_gpu,
                                         execution, command, dit_input_gpu, error) != 0
        || sampler.apply_cfg_vulkan(positive_gpu, negative_gpu, 2.5f, 0.4f,
                                    execution, command, cfg_gpu, error) != 0
        || sampler.step_vulkan(cfg_gpu, positive_gpu, 1000.0f, 500.0f,
                               execution, command, next_gpu, error) != 0
        || command.submit_and_wait() != 0)
    {
        std::cerr << error << '\n';
        return 4;
    }

    ncnn::Mat condition;
    ncnn::Mat dit_input;
    ncnn::Mat cfg;
    ncnn::Mat next;
    if (download(condition_gpu, execution, condition) != 0
        || download(dit_input_gpu, execution, dit_input) != 0
        || download(cfg_gpu, execution, cfg) != 0
        || download(next_gpu, execution, next) != 0)
        return 5;

    ncnn::Mat expected_condition(16, tokens, static_cast<size_t>(4u), 1);
    for (int i = 0; i < 16 * tokens; ++i)
        static_cast<float*>(expected_condition.data)[i] =
            0.75f * static_cast<float*>(positive.data)[i]
            + 0.25f * static_cast<float*>(negative.data)[i];
    if (!close_mat(condition, expected_condition, 1e-6f))
        return 6;
    for (int token = 0; token < tokens; ++token)
    {
        const float* row = dit_input.row(token);
        for (int channel = 0; channel < 16; ++channel)
            if (std::abs(row[channel] - positive.row(token)[channel]) > 1e-6f
                || std::abs(row[channel + 16] - expected_condition.row(token)[channel]) > 1e-6f)
                return 7;
        if (std::abs(row[32] - 1.0f) > 1e-6f)
            return 8;
    }

    // CPU sampler 使用 4D 视图，但数据连续顺序与 [tokens,16] 测试矩阵一致。
    ncnn::Mat positive_4d(16, 1, 1, tokens, 4u, 1);
    ncnn::Mat negative_4d(16, 1, 1, tokens, 4u, 1);
    std::copy(static_cast<float*>(positive.data), static_cast<float*>(positive.data) + 16*tokens,
              static_cast<float*>(positive_4d.data));
    std::copy(static_cast<float*>(negative.data), static_cast<float*>(negative.data) + 16*tokens,
              static_cast<float*>(negative_4d.data));
    ncnn::Mat expected_cfg_4d;
    if (sampler.apply_cfg(positive_4d, negative_4d, 2.5f, 0.4f, expected_cfg_4d, error) != 0)
        return 9;
    ncnn::Mat expected_cfg(16, tokens, static_cast<size_t>(4u), 1);
    std::copy(static_cast<float*>(expected_cfg_4d.data),
              static_cast<float*>(expected_cfg_4d.data) + 16*tokens,
              static_cast<float*>(expected_cfg.data));
    if (!close_mat(cfg, expected_cfg, 2e-5f))
        return 10;
    for (int i = 0; i < 16*tokens; ++i)
        if (std::abs(static_cast<float*>(next.data)[i]
                     - (static_cast<float*>(positive.data)[i]
                        - 0.5f*static_cast<float*>(cfg.data)[i])) > 2e-5f)
            return 11;

    std::cout << "sampler Vulkan tests passed\n";
    return 0;
}
