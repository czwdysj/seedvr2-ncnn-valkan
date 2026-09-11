// 本文件是 seedvr2-ncnn-vulkan 的正式命令行入口，对齐 zimage-ncnn-vulkan 的
// 使用体验：一条命令完成 mp4 → mp4 视频修复。
//
// 媒体编解码不进入核心库：CLI 通过调用系统 ffmpeg/ffprobe 完成 mp4 解码与
// 编码（帧数据经 rgb24 raw 交换），因此用户只需安装 ffmpeg 并加入 PATH。
// 文本条件未提供时自动回退到模型目录内的官方默认 embedding（default_*_emb.bin），
// 开箱即用，无需 PyTorch。
//
// 用法：
//   seedvr2-ncnn-vulkan -i input.mp4 -o output.mp4 [选项]
// 选项：
//   --models DIR    模型目录（默认 ./models）
//   --steps N       采样步数（默认 1，官方 one-step）
//   --cfg X         CFG 强度（默认 1.0；>1 时启用负向文本）
//   --seed S        随机种子（默认 666）
//   --resident      DiT 32 block 权重常驻显存（快约 66 倍，多占约 12GB 显存）
//   --threads N     CPU 线程数（默认 8）
#include "seedvr2/engine.h"

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#ifdef _WIN32
#define POPEN _popen
#define POPEN_READ_MODE "rb"
#define POPEN_WRITE_MODE "wb"
#define PCLOSE _pclose
#else
// POSIX popen 的 mode 只接受 "r"/"w"（"rb" 会 EINVAL），二进制语义即默认行为。
#define POPEN popen
#define POPEN_READ_MODE "r"
#define POPEN_WRITE_MODE "w"
#define PCLOSE pclose
#endif

namespace
{

struct Options
{
    std::string input;
    std::string output;
    std::string models_dir = "models";
    int steps = 1;
    float cfg = 1.0f;
    long long seed = 666;
    int threads = 8;
    bool resident = false;
};

bool run_command(const std::string& command)
{
    FILE* stream = POPEN(command.c_str(), "r");
    if (stream == nullptr)
        return false;
    char buffer[256];
    while (std::fread(buffer, 1, sizeof(buffer), stream) > 0)
    {
    }
    const int status = PCLOSE(stream);
#ifdef _WIN32
    return status == 0;
#else
    return status != -1 && WIFEXITED(status) && WEXITSTATUS(status) == 0;
#endif
}

bool tool_available(const char* tool)
{
    std::string command = std::string(tool)
        + " -version" + std::string(" >")
#ifdef _WIN32
        + "nul"
#else
        + "/dev/null"
#endif
        + " 2>&1";
    return run_command(command);
}

// ffprobe 读取视频宽度、高度与帧率。
bool probe_video(const std::string& path, int& width, int& height, float& fps)
{
    const std::string command =
        "ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate "
        "-of csv=p=0 \"" + path + "\"";
    FILE* stream = POPEN(command.c_str(), "r");
    if (stream == nullptr)
        return false;
    char line[256] = {0};
    const bool ok = std::fgets(line, sizeof(line), stream) != nullptr;
    PCLOSE(stream);
    if (!ok)
        return false;
    int num = 0, den = 1;
    if (std::sscanf(line, "%d,%d,%d/%d", &width, &height, &num, &den) != 4
        || width <= 0 || height <= 0 || num <= 0 || den <= 0)
        return false;
    fps = static_cast<float>(num) / static_cast<float>(den);
    if (fps <= 0.0f || fps > 240.0f)
        fps = 24.0f;
    return true;
}

// 解码整段视频为 rgb24 帧（每帧 width*height*3 字节），返回 false 表示失败。
bool decode_frames(const std::string& path,
                   int width,
                   int height,
                   std::vector<unsigned char>& frames)
{
    const std::string command =
        "ffmpeg -v error -i \"" + path + "\" -f rawvideo -pix_fmt rgb24 -";
    FILE* stream = POPEN(command.c_str(), POPEN_READ_MODE);
    if (stream == nullptr)
        return false;
    const std::size_t frame_bytes = static_cast<std::size_t>(width) * height * 3;
    frames.clear();
    std::vector<unsigned char> buffer(frame_bytes);
    while (true)
    {
        const std::size_t got = std::fread(buffer.data(), 1, frame_bytes, stream);
        if (got == 0)
            break;
        if (got != frame_bytes)
        {
            std::fprintf(stderr, "[decode] partial frame: got %zu of %zu bytes\n",
                         got, frame_bytes);
            PCLOSE(stream);
            return false;   // 尾部残帧（尺寸不完整）直接丢弃
        }
        frames.insert(frames.end(), buffer.begin(), buffer.end());
    }
    const int status = PCLOSE(stream);
    std::fprintf(stderr, "[decode] frames=%zu pclose=%d\n", frames.size(), status);
#ifndef _WIN32
    if (status == -1 || !WIFEXITED(status) || WEXITSTATUS(status) != 0)
        return false;
#endif
    return !frames.empty();
}

bool encode_video(const std::string& output,
                  const std::string& input,
                  const unsigned char* rgb,
                  int frames,
                  int width,
                  int height,
                  float fps)
{
    const std::string command =
        "ffmpeg -v error -y -f rawvideo -pix_fmt rgb24 -s "
        + std::to_string(width) + "x" + std::to_string(height)
        + " -r " + std::to_string(fps)
        + " -i - -i \"" + input + "\" -map 0:v -map 1:a? "
        + "-c:v libx264 -crf 18 -pix_fmt yuv420p -c:a copy \"" + output + "\"";
    FILE* stream = POPEN(command.c_str(), POPEN_WRITE_MODE);
    if (stream == nullptr)
        return false;
    const std::size_t total = static_cast<std::size_t>(frames) * width * height * 3;
    const bool ok = std::fwrite(rgb, 1, total, stream) == total;
    const int status = PCLOSE(stream);
#ifdef _WIN32
    return ok && status == 0;
#else
    return ok && status != -1 && WEXITSTATUS(status) == 0;
#endif
}

void print_usage(const char* program)
{
    std::cerr
        << "usage: " << program << " -i input.mp4 -o output.mp4 [options]\n"
        << "  -i PATH          input video (any format ffmpeg supports)\n"
        << "  -o PATH          output video (mp4)\n"
        << "  --models DIR     model directory (default: ./models)\n"
        << "  --steps N        sampling steps (default: 1, official one-step)\n"
        << "  --cfg X          classifier-free guidance scale (default: 1.0)\n"
        << "  --seed S         random seed (default: 666)\n"
        << "  --threads N      CPU threads (default: 8)\n"
        << "  --resident       keep all 32 DiT blocks in VRAM (much faster, +12GB VRAM)\n";
}

} // namespace

int main(int argc, char** argv)
{
    Options options;
    for (int index = 1; index < argc; ++index)
    {
        const std::string arg = argv[index];
        auto next_value = [&](std::string& value) -> bool {
            if (index + 1 >= argc)
                return false;
            value = argv[++index];
            return true;
        };
        if (arg == "-i" || arg == "--input")
        {
            if (!next_value(options.input))
                return print_usage(argv[0]), 2;
        }
        else if (arg == "-o" || arg == "--output")
        {
            if (!next_value(options.output))
                return print_usage(argv[0]), 2;
        }
        else if (arg == "--models")
        {
            if (!next_value(options.models_dir))
                return print_usage(argv[0]), 2;
        }
        else if (arg == "--steps")
        {
            std::string value;
            if (!next_value(value))
                return print_usage(argv[0]), 2;
            options.steps = std::atoi(value.c_str());
        }
        else if (arg == "--cfg")
        {
            std::string value;
            if (!next_value(value))
                return print_usage(argv[0]), 2;
            options.cfg = std::strtof(value.c_str(), nullptr);
        }
        else if (arg == "--seed")
        {
            std::string value;
            if (!next_value(value))
                return print_usage(argv[0]), 2;
            options.seed = std::atoll(value.c_str());
        }
        else if (arg == "--threads")
        {
            std::string value;
            if (!next_value(value))
                return print_usage(argv[0]), 2;
            options.threads = std::atoi(value.c_str());
        }
        else if (arg == "--resident")
        {
            options.resident = true;
        }
        else
        {
            std::cerr << "unknown option: " << arg << "\n";
            return print_usage(argv[0]), 2;
        }
    }

    if (options.input.empty() || options.output.empty())
        return print_usage(argv[0]), 2;
    if (!tool_available("ffmpeg") || !tool_available("ffprobe"))
    {
        std::cerr << "error: ffmpeg/ffprobe not found in PATH.\n"
                  << "install ffmpeg first, e.g. `sudo apt install ffmpeg` "
                  << "or download from https://ffmpeg.org\n";
        return 1;
    }

    int width = 0, height = 0;
    float fps = 24.0f;
    if (!probe_video(options.input, width, height, fps))
    {
        std::cerr << "error: failed to probe input video: " << options.input << "\n";
        return 1;
    }

    std::cout << "decoding " << options.input << " (" << width << "x" << height
              << ") ...\n";
    std::vector<unsigned char> raw_frames;
    if (!decode_frames(options.input, width, height, raw_frames))
    {
        std::cerr << "error: ffmpeg decode failed.\n";
        return 1;
    }
    const std::size_t frame_bytes = static_cast<std::size_t>(width) * height * 3;
    std::size_t frame_count = raw_frames.size() / frame_bytes;
    if (frame_count == 0)
    {
        std::cerr << "error: input video has no decodable frames.\n";
        return 1;
    }
    // SeedVR2 的 VAE 要求帧数满足 (T-1)%4==0；不足时重复末帧 padding，编码前裁回。
    const std::size_t aligned_frames = (frame_count - 1 + 3) / 4 * 4 + 1;
    if (aligned_frames != frame_count)
    {
        std::cout << "padding frames " << frame_count << " -> " << aligned_frames
                  << " (VAE requires (T-1)%4==0)\n";
        raw_frames.resize(aligned_frames * frame_bytes, 0);
        const unsigned char* last_frame = raw_frames.data() + (frame_count - 1) * frame_bytes;
        for (std::size_t frame = frame_count; frame < aligned_frames; ++frame)
            std::memcpy(raw_frames.data() + frame * frame_bytes, last_frame, frame_bytes);
        frame_count = aligned_frames;
    }
    std::cout << "frames: " << frame_count << ", fps: " << fps << "\n";

    // rgb24 与 Video 的 THWC 布局逐帧等价（均为 H,W,C 连续），只需 u8 → fp32 归一化。
    seedvr2::Video input;
    input.frames = static_cast<int>(frame_count);
    input.height = height;
    input.width = width;
    input.channels = 3;
    input.fps = fps;
    input.data.resize(raw_frames.size());
    for (std::size_t index = 0; index < raw_frames.size(); ++index)
        input.data[index] = static_cast<float>(raw_frames[index]) / 255.0f;

    seedvr2::RuntimeOptions runtime;
    runtime.device = seedvr2::DeviceType::Vulkan;
    runtime.num_threads = options.threads;
    runtime.sampling_steps = options.steps > 0 ? options.steps : 1;
    runtime.cfg_scale = options.cfg;
    runtime.seed = static_cast<std::uint64_t>(options.seed);
    runtime.dit_resident = options.resident;

    std::cout << "loading models from " << options.models_dir << " ...\n";
    seedvr2::SeedVR2Engine engine;
    if (engine.load(options.models_dir, runtime) != 0)
    {
        std::cerr << "error: model load failed: " << engine.last_error() << "\n"
                  << "run ./download-models.sh first, or pass --models DIR.\n";
        return 1;
    }

    // 文本条件留空 → Engine 自动使用模型目录内的官方默认 embedding。
    seedvr2::Video output;
    std::cout << "running SeedVR2 (" << runtime.sampling_steps << " step(s), "
              << (runtime.dit_resident ? "resident" : "stream") << ") ...\n";
    if (engine.process(input, seedvr2::TextEmbedding{}, seedvr2::TextEmbedding{}, output) != 0)
    {
        std::cerr << "error: inference failed: " << engine.last_error() << "\n";
        return 1;
    }

    // fp32 THWC → rgb24（布局逐帧等价，仅 fp32 → u8 量化）。
    std::vector<unsigned char> out_rgb(output.data.size());
    for (std::size_t index = 0; index < output.data.size(); ++index)
    {
        float value = output.data[index] * 255.0f;
        value = value < 0.0f ? 0.0f : (value > 255.0f ? 255.0f : value);
        out_rgb[index] = static_cast<unsigned char>(value + 0.5f);
    }

    std::cout << "encoding " << options.output << " ...\n";
    const int out_frames = output.frames < static_cast<int>(frame_count)
        ? output.frames
        : static_cast<int>(frame_count);
    if (!encode_video(options.output, options.input, out_rgb.data(),
                      out_frames, output.width, output.height, fps))
    {
        std::cerr << "error: ffmpeg encode failed.\n";
        return 1;
    }
    std::cout << "done: " << options.output << "\n";
    return 0;
}
