// 本文件实现 VAE 输出的裁帧、限幅和布局转换。
// 它只移除预处理为满足 4n+1 而复制的尾帧；空间尺寸保留中心裁剪后的结果，
// 防止无定义的插值步骤影响 PyTorch 与 NCNN 的数值对齐。
#include "pipeline/postprocessing.h"

#include <algorithm>
#include <cstddef>

namespace seedvr2
{
int postprocess_video(const ncnn::Mat& decoded,
                      const PreparedVideo& prepared,
                      Video& output,
                      std::string& error)
{
    if (decoded.dims != 4 || decoded.c != 3 || decoded.d < prepared.original_frames
        || decoded.h != prepared.processed_height || decoded.w != prepared.processed_width
        || decoded.elemsize != 4u || decoded.elempack != 1)
    {
        error = "decoder output shape is inconsistent with preprocessed video";
        return static_cast<int>(Status::InferenceFailed);
    }

    Video result;
    result.frames = prepared.original_frames;
    result.height = decoded.h;
    result.width = decoded.w;
    result.channels = 3;
    result.fps = prepared.fps;
    result.data.resize(static_cast<std::size_t>(result.frames) * result.height * result.width * 3);

    for (int frame = 0; frame < result.frames; ++frame)
    {
        for (int y = 0; y < result.height; ++y)
        {
            for (int x = 0; x < result.width; ++x)
            {
                for (int channel = 0; channel < 3; ++channel)
                {
                    const float* source = decoded.channel(channel).depth(frame).row(y);
                    const std::size_t index =
                        (((static_cast<std::size_t>(frame) * result.height + y) * result.width + x)
                         * 3)
                        + channel;
                    result.data[index] = std::clamp(source[x], -1.0f, 1.0f) * 0.5f + 0.5f;
                }
            }
        }
    }
    output = std::move(result);
    return static_cast<int>(Status::Ok);
}
} // namespace seedvr2
