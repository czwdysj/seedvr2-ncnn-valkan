// 本文件声明 SeedVR2 使用的采样策略接口和 Euler 实现。
// 张量保持 NCNN C,T,H,W 布局；CFG 实现正负分支融合，Euler 按 lerp schedule
// 和 v_lerp prediction 更新 latent。该模块不加载模型，因此可独立做确定性单元测试。
#pragma once

#include <net.h>

#include <string>
#include <vector>

namespace seedvr2
{
class EulerSampler
{
public:
    explicit EulerSampler(int steps);

    const std::vector<float>& timesteps() const noexcept { return timesteps_; }
    int apply_cfg(const ncnn::Mat& positive,
                  const ncnn::Mat& negative,
                  float scale,
                  float rescale,
                  ncnn::Mat& output,
                  std::string& error) const;
    int step_to(const ncnn::Mat& prediction,
                const ncnn::Mat& current,
                float timestep,
                float next_timestep,
                ncnn::Mat& output,
                std::string& error) const;
    int endpoint(const ncnn::Mat& prediction,
                 const ncnn::Mat& current,
                 float timestep,
                 ncnn::Mat& output,
                 std::string& error) const;

private:
    std::vector<float> timesteps_;
};
} // namespace seedvr2
