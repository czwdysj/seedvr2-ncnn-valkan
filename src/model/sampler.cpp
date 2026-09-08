// 本文件实现与 SeedVR2 PyTorch 基准一致的 CFG 和 Euler/v_lerp 更新。
// lerp schedule 满足 A(t)=1-t/T、B(t)=t/T；最终 endpoint 直接得到
// x0=x_t-B(t)*prediction。全部计算使用 FP32，便于先完成 CPU 数值基线。
#include "model/sampler.h"

#include <algorithm>
#include <cmath>

#include "seedvr2/engine.h"

namespace seedvr2
{
namespace
{
bool same_shape(const ncnn::Mat& a, const ncnn::Mat& b)
{
    return a.dims == 4 && b.dims == 4 && a.w == b.w && a.h == b.h && a.d == b.d
        && a.c == b.c && a.elemsize == 4u && b.elemsize == 4u && a.elempack == 1
        && b.elempack == 1;
}

template <typename Function>
void transform_4d(const ncnn::Mat& a, const ncnn::Mat& b, ncnn::Mat& output, Function function)
{
    for (int channel = 0; channel < a.c; ++channel)
        for (int frame = 0; frame < a.d; ++frame)
            for (int y = 0; y < a.h; ++y)
            {
                const float* pa = a.channel(channel).depth(frame).row(y);
                const float* pb = b.channel(channel).depth(frame).row(y);
                float* po = output.channel(channel).depth(frame).row(y);
                for (int x = 0; x < a.w; ++x)
                    po[x] = function(pa[x], pb[x]);
            }
}

double standard_deviation(const ncnn::Mat& value)
{
    const std::size_t count =
        static_cast<std::size_t>(value.w) * value.h * value.d * value.c;
    double sum = 0.0;
    double square_sum = 0.0;
    for (int channel = 0; channel < value.c; ++channel)
        for (int frame = 0; frame < value.d; ++frame)
            for (int y = 0; y < value.h; ++y)
            {
                const float* row = value.channel(channel).depth(frame).row(y);
                for (int x = 0; x < value.w; ++x)
                {
                    sum += row[x];
                    square_sum += static_cast<double>(row[x]) * row[x];
                }
            }
    if (count <= 1)
        return 0.0;
    const double variance = std::max(0.0, (square_sum - sum * sum / count) / (count - 1));
    return std::sqrt(variance);
}
} // namespace

EulerSampler::EulerSampler(int steps)
{
    steps = std::max(1, steps);
    timesteps_.reserve(steps);
    for (int index = 0; index < steps; ++index)
        timesteps_.push_back(1000.0f * (1.0f - static_cast<float>(index) / steps));
}

int EulerSampler::apply_cfg(const ncnn::Mat& positive,
                            const ncnn::Mat& negative,
                            float scale,
                            float rescale,
                            ncnn::Mat& output,
                            std::string& error) const
{
    if (!same_shape(positive, negative))
    {
        error = "CFG inputs must have identical FP32 C,T,H,W shapes";
        return static_cast<int>(Status::InvalidArgument);
    }
    output.create(positive.w, positive.h, positive.d, positive.c, 4u, 1);
    if (output.empty())
        return static_cast<int>(Status::OutOfMemory);

    transform_4d(positive, negative, output,
                 [scale](float pos, float neg) { return neg + scale * (pos - neg); });
    if (rescale != 0.0f)
    {
        const double cfg_std = standard_deviation(output);
        if (cfg_std > 0.0)
        {
            const float factor = static_cast<float>(rescale * standard_deviation(positive) / cfg_std
                                                    + (1.0f - rescale));
            for (int channel = 0; channel < output.c; ++channel)
                for (int frame = 0; frame < output.d; ++frame)
                    for (int y = 0; y < output.h; ++y)
                    {
                        float* row = output.channel(channel).depth(frame).row(y);
                        for (int x = 0; x < output.w; ++x)
                            row[x] *= factor;
                    }
        }
    }
    return static_cast<int>(Status::Ok);
}

int EulerSampler::step_to(const ncnn::Mat& prediction,
                          const ncnn::Mat& current,
                          float timestep,
                          float next_timestep,
                          ncnn::Mat& output,
                          std::string& error) const
{
    if (!same_shape(prediction, current))
    {
        error = "Euler prediction and current latent shapes differ";
        return static_cast<int>(Status::InvalidArgument);
    }
    output.create(current.w, current.h, current.d, current.c, 4u, 1);
    if (output.empty())
        return static_cast<int>(Status::OutOfMemory);

    const float bt = timestep / 1000.0f;
    const float at = 1.0f - bt;
    const float bs = std::clamp(next_timestep / 1000.0f, 0.0f, 1.0f);
    const float as = 1.0f - bs;
    transform_4d(current, prediction, output, [=](float x, float pred) {
        const float x0 = x - bt * pred;
        const float xT = x + at * pred;
        return as * x0 + bs * xT;
    });
    return static_cast<int>(Status::Ok);
}

int EulerSampler::endpoint(const ncnn::Mat& prediction,
                           const ncnn::Mat& current,
                           float timestep,
                           ncnn::Mat& output,
                           std::string& error) const
{
    if (!same_shape(prediction, current))
    {
        error = "Euler endpoint inputs have different shapes";
        return static_cast<int>(Status::InvalidArgument);
    }
    output.create(current.w, current.h, current.d, current.c, 4u, 1);
    if (output.empty())
        return static_cast<int>(Status::OutOfMemory);
    const float bt = timestep / 1000.0f;
    transform_4d(current, prediction, output,
                 [bt](float x, float pred) { return x - bt * pred; });
    return static_cast<int>(Status::Ok);
}
} // namespace seedvr2
