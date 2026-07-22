// 本文件实现公开 Video 到 VAE 输入 Mat 的确定性转换。
// 空间采用 SeedVR2 官方 DivisibleCrop 的中心裁剪语义，不偷偷缩放输入；时间轴
// 复制最后一帧到 4n+1，使因果 VAE 的四倍时间下采样满足结构约束。
#include "preprocessing.h"

#include <algorithm>
#include <cstddef>

namespace seedvr2
{
int preprocess_video(const Video& input, PreparedVideo& output, std::string& error)
{
    if (!input.valid() || input.channels != 3)
    {
        error = "input video must be valid THWC RGB data";
        return static_cast<int>(Status::InvalidArgument);
    }

    const int target_height = input.height / 16 * 16;
    const int target_width = input.width / 16 * 16;
    if (target_height <= 0 || target_width <= 0)
    {
        error = "input width and height must be at least 16";
        return static_cast<int>(Status::InvalidArgument);
    }

    int target_frames = input.frames;
    if (target_frames > 1 && (target_frames - 1) % 4 != 0)
        target_frames += 4 - ((target_frames - 1) % 4);

    ncnn::Mat tensor(target_width, target_height, target_frames, 3, 4u, 1);
    if (tensor.empty())
    {
        error = "failed to allocate preprocessed video";
        return static_cast<int>(Status::OutOfMemory);
    }

    const int top = (input.height - target_height) / 2;
    const int left = (input.width - target_width) / 2;
    for (int channel = 0; channel < 3; ++channel)
    {
        for (int frame = 0; frame < target_frames; ++frame)
        {
            const int source_frame = std::min(frame, input.frames - 1);
            for (int y = 0; y < target_height; ++y)
            {
                float* destination = tensor.channel(channel).depth(frame).row(y);
                for (int x = 0; x < target_width; ++x)
                {
                    const std::size_t index =
                        (((static_cast<std::size_t>(source_frame) * input.height + y + top)
                          * input.width
                          + x + left)
                         * input.channels)
                        + channel;
                    const float value = std::clamp(input.data[index], 0.0f, 1.0f);
                    destination[x] = value * 2.0f - 1.0f;
                }
            }
        }
    }

    output.tensor = tensor;
    output.original_frames = input.frames;
    output.processed_frames = target_frames;
    output.processed_height = target_height;
    output.processed_width = target_width;
    output.fps = input.fps;
    return static_cast<int>(Status::Ok);
}
} // namespace seedvr2
