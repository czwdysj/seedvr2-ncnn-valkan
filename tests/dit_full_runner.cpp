// 本文件是正式 SeedVR2DiT 类的端到端数值测试入口。
// 它兼容旧验证脚本的 [L,33] raw 参数，把输入还原成 [33,T,H,W] 后调用库 API，
// 再把 [16,T,H,W] 输出展平写回。32 层调度不再复制在 runner 中。
#include "model/dit.h"
#include "core/runtime_context.h"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

namespace
{
bool read_matrix(const std::string& path, int rows, int columns, ncnn::Mat& output)
{
    output.create(columns, rows, static_cast<size_t>(4u), 1);
    std::ifstream stream(path, std::ios::binary);
    if (output.empty() || !stream)
        return false;
    const std::streamsize bytes = static_cast<std::streamsize>(rows) * columns * sizeof(float);
    stream.read(static_cast<char*>(output.data), bytes);
    return stream.gcount() == bytes;
}

ncnn::Mat unflatten(const ncnn::Mat& value, int frames, int height, int width)
{
    ncnn::Mat output(width, height, frames, value.w, 4u, 1);
    int token = 0;
    for (int frame = 0; frame < frames; ++frame)
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x, ++token)
                for (int channel = 0; channel < value.w; ++channel)
                    output.channel(channel).depth(frame).row(y)[x] = value.row(token)[channel];
    return output;
}

ncnn::Mat flatten(const ncnn::Mat& value)
{
    ncnn::Mat output(value.c,
                     value.d * value.h * value.w,
                     static_cast<size_t>(4u),
                     1);
    int token = 0;
    for (int frame = 0; frame < value.d; ++frame)
        for (int y = 0; y < value.h; ++y)
            for (int x = 0; x < value.w; ++x, ++token)
                for (int channel = 0; channel < value.c; ++channel)
                    output.row(token)[channel] = value.channel(channel).depth(frame).row(y)[x];
    return output;
}

bool write_matrix(const std::string& path, const ncnn::Mat& value)
{
    std::ofstream stream(path, std::ios::binary);
    if (!stream || value.dims != 2)
        return false;
    const std::int32_t shape[2] = {value.h, value.w};
    stream.write(reinterpret_cast<const char*>(shape), sizeof(shape));
    stream.write(static_cast<const char*>(value.data),
                 static_cast<std::streamsize>(value.h) * value.w * sizeof(float));
    return stream.good();
}
} // namespace

int main(int argc, char** argv)
{
    if (argc != 12)
    {
        std::cerr << "usage: " << argv[0]
                  << " io_model_dir block_model_dir vid.f32 vid_rows txt.f32 txt_rows"
                     " timestep T H W output.f32\n";
        return 2;
    }
    const int frames = std::atoi(argv[8]);
    const int height = std::atoi(argv[9]);
    const int width = std::atoi(argv[10]);
    ncnn::Mat flat_video;
    ncnn::Mat text;
    if (!read_matrix(argv[3], std::atoi(argv[4]), 33, flat_video)
        || !read_matrix(argv[5], std::atoi(argv[6]), 5120, text)
        || flat_video.h != frames * height * width)
        return 3;

    seedvr2::RuntimeOptions options;
    options.num_threads = 8;
    seedvr2::RuntimeContext context;
    std::string error;
    if (context.initialize(options, error) != 0)
    {
        std::cerr << error << '\n';
        return 4;
    }
    seedvr2::SeedVR2DiT dit;
    if (dit.load(argv[2], context) != 0)
    {
        std::cerr << dit.last_error() << '\n';
        return 5;
    }
    ncnn::Mat output;
    if (dit.forward(unflatten(flat_video, frames, height, width),
                    text,
                    std::strtof(argv[7], nullptr),
                    output)
        != 0)
    {
        std::cerr << dit.last_error() << '\n';
        return 6;
    }
    if (!write_matrix(argv[11], flatten(output)))
        return 7;
    std::cout << "DiT complete: C,T,H,W=" << output.c << ',' << output.d << ','
              << output.h << ',' << output.w << '\n';
    return 0;
}
