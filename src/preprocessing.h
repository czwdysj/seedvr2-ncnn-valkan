// 本文件声明视频进入 VAE 前的预处理边界。
// 输入是公开 API 的 THWC、[0,1] 视频；输出是 NCNN 的 W,H,T,C 四维 Mat，
// 数值范围转换为 [-1,1]，空间中心裁剪到 16 的倍数，时间复制到 4n+1。
#pragma once

#include <net.h>

#include <string>

#include "seedvr2/engine.h"

namespace seedvr2
{
struct PreparedVideo
{
    ncnn::Mat tensor;
    int original_frames = 0;
    int processed_frames = 0;
    int processed_height = 0;
    int processed_width = 0;
    float fps = 24.0f;
};

int preprocess_video(const Video& input, PreparedVideo& output, std::string& error);
} // namespace seedvr2
