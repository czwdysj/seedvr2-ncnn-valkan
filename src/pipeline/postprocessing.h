// 本文件声明 VAE decoder 输出到公开 Video 的后处理。
// 输入 Mat 的逻辑布局是 C,T,H,W、范围约为 [-1,1]；输出裁掉预处理补出的
// 尾帧，并转换成调用方使用的 THWC、FP32、[0,1] 视频。
#pragma once

#include <net.h>

#include <string>

#include "pipeline/preprocessing.h"
#include "seedvr2/engine.h"

namespace seedvr2
{
int postprocess_video(const ncnn::Mat& decoded,
                      const PreparedVideo& prepared,
                      Video& output,
                      std::string& error);
} // namespace seedvr2
