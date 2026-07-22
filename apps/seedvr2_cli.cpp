// 本文件提供 SeedVR2Engine 的正式命令行入口和最小文件协议。
// 输入视频是无头 FP32 THWC raw，文本是无头 FP32 [tokens,5120] raw；输出先写
// int32[T,H,W,C] 再写 FP32 THWC。媒体编解码不进入核心库，调用方可用 ffmpeg 转换。
#include "seedvr2/engine.h"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

namespace
{
template <typename T>
bool read_values(const std::string& path, std::vector<T>& values)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        return false;
    stream.read(reinterpret_cast<char*>(values.data()),
                static_cast<std::streamsize>(values.size() * sizeof(T)));
    return stream.gcount() == static_cast<std::streamsize>(values.size() * sizeof(T));
}

bool write_video(const std::string& path, const seedvr2::Video& video)
{
    std::ofstream stream(path, std::ios::binary);
    if (!stream)
        return false;
    const std::int32_t shape[4] = {video.frames, video.height, video.width, video.channels};
    stream.write(reinterpret_cast<const char*>(shape), sizeof(shape));
    stream.write(reinterpret_cast<const char*>(video.data.data()),
                 static_cast<std::streamsize>(video.data.size() * sizeof(float)));
    return stream.good();
}
} // namespace

int main(int argc, char** argv)
{
    if (argc != 12)
    {
        std::cerr << "usage: " << argv[0]
                  << " model_dir input_thwc.f32 T H W positive.f32 pos_tokens"
                     " negative.f32 neg_tokens output.f32 threads\n";
        return 2;
    }

    seedvr2::Video input;
    input.frames = std::atoi(argv[3]);
    input.height = std::atoi(argv[4]);
    input.width = std::atoi(argv[5]);
    input.channels = 3;
    input.data.resize(static_cast<std::size_t>(input.frames) * input.height * input.width * 3);

    seedvr2::TextEmbedding positive;
    positive.tokens = std::atoi(argv[7]);
    positive.data.resize(static_cast<std::size_t>(positive.tokens) * positive.channels);
    seedvr2::TextEmbedding negative;
    negative.tokens = std::atoi(argv[9]);
    negative.data.resize(static_cast<std::size_t>(negative.tokens) * negative.channels);
    if (!read_values(argv[2], input.data) || !read_values(argv[6], positive.data)
        || !read_values(argv[8], negative.data))
    {
        std::cerr << "failed to read an input raw file\n";
        return 3;
    }

    seedvr2::RuntimeOptions options;
    options.num_threads = std::atoi(argv[11]);
    seedvr2::SeedVR2Engine engine;
    int result = engine.load(argv[1], options);
    if (result != 0)
    {
        std::cerr << "load failed: " << engine.last_error() << '\n';
        return 4;
    }
    seedvr2::Video output;
    result = engine.process(input, positive, negative, output);
    if (result != 0)
    {
        std::cerr << "process failed: " << engine.last_error() << '\n';
        return 5;
    }
    if (!write_video(argv[10], output))
    {
        std::cerr << "failed to write output video tensor\n";
        return 6;
    }
    std::cout << "complete: T,H,W,C=" << output.frames << ',' << output.height << ','
              << output.width << ',' << output.channels << '\n';
    return 0;
}
