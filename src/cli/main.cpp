// 本文件是 seedvr2-ncnn-vulkan 的 Linux/WSL2 正式命令行入口，对齐
// zimage-ncnn-vulkan 的使用体验：一条命令完成 mp4 → mp4 视频修复。
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
//   --resident      DiT 32 block 权重常驻显存（更快，但需要更多显存）
//   --gpu N         Vulkan 设备编号（默认 0）
//   --threads N     CPU 线程数（默认 8）
#include "seedvr2/engine.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <sys/wait.h>

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
    int gpu = 0;
    bool resident = false;
};

// ffmpeg/ffprobe 由 POSIX shell 启动，因此所有来自用户的路径必须作为一个参数转义。
// 单引号中的单引号需要结束当前字符串、写入转义字符，再重新进入单引号字符串。
std::string shell_quote(const std::string& value)
{
    std::string result = "'";
    for (const char character : value)
        result += character == '\'' ? "'\\''" : std::string(1, character);
    return result + "'";
}

bool parse_int(const std::string& text, int minimum, int maximum, int& output)
{
    errno = 0;
    char* end = nullptr;
    const long value = std::strtol(text.c_str(), &end, 10);
    if (errno != 0 || end == text.c_str() || *end != '\0' || value < minimum || value > maximum)
        return false;
    output = static_cast<int>(value);
    return true;
}

bool parse_long_long(const std::string& text, long long& output)
{
    errno = 0;
    char* end = nullptr;
    const long long value = std::strtoll(text.c_str(), &end, 10);
    if (errno != 0 || end == text.c_str() || *end != '\0')
        return false;
    output = value;
    return true;
}

bool parse_float(const std::string& text, float minimum, float maximum, float& output)
{
    errno = 0;
    char* end = nullptr;
    const float value = std::strtof(text.c_str(), &end);
    if (errno != 0 || end == text.c_str() || *end != '\0' || !std::isfinite(value)
        || value < minimum || value > maximum)
        return false;
    output = value;
    return true;
}

bool run_command(const std::string& command)
{
    FILE* stream = popen(command.c_str(), "r");
    if (stream == nullptr)
        return false;
    char buffer[256];
    while (std::fread(buffer, 1, sizeof(buffer), stream) > 0)
    {
    }
    const int status = pclose(stream);
    return status != -1 && WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

bool tool_available(const char* tool)
{
    std::string command = std::string(tool)
        + " -version" + std::string(" >")
        + "/dev/null"
        + " 2>&1";
    return run_command(command);
}

// ffprobe 读取视频宽度、高度与帧率。
bool probe_video(const std::string& path, int& width, int& height, float& fps)
{
    const std::string command =
        "ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate "
        "-of csv=p=0 " + shell_quote(path);
    FILE* stream = popen(command.c_str(), "r");
    if (stream == nullptr)
        return false;
    char line[256] = {0};
    const bool ok = std::fgets(line, sizeof(line), stream) != nullptr;
    pclose(stream);
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
        "ffmpeg -v error -i " + shell_quote(path) + " -f rawvideo -pix_fmt rgb24 -";
    // POSIX popen 只接受 "r"/"w"；Linux 管道本身不执行文本换行转换。
    FILE* stream = popen(command.c_str(), "r");
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
            pclose(stream);
            return false;   // 尾部残帧（尺寸不完整）直接丢弃
        }
        frames.insert(frames.end(), buffer.begin(), buffer.end());
    }
    const int status = pclose(stream);
    if (status == -1 || !WIFEXITED(status) || WEXITSTATUS(status) != 0)
        return false;
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
        + " -i - -i " + shell_quote(input) + " -map 0:v -map 1:a? "
        + "-c:v libx264 -crf 18 -pix_fmt yuv420p -c:a aac -b:a 192k -shortest "
        + shell_quote(output);
    FILE* stream = popen(command.c_str(), "w");
    if (stream == nullptr)
        return false;
    const std::size_t total = static_cast<std::size_t>(frames) * width * height * 3;
    const bool ok = std::fwrite(rgb, 1, total, stream) == total;
    const int status = pclose(stream);
    return ok && status != -1 && WIFEXITED(status) && WEXITSTATUS(status) == 0;
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
        << "  --gpu N          Vulkan device index (default: 0)\n"
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
        if (arg == "-h" || arg == "--help")
        {
            print_usage(argv[0]);
            return 0;
        }
        else if (arg == "-i" || arg == "--input")
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
            if (!parse_int(value, 1, 1000, options.steps))
            {
                std::cerr << "invalid --steps value: " << value << "\n";
                return 2;
            }
        }
        else if (arg == "--cfg")
        {
            std::string value;
            if (!next_value(value))
                return print_usage(argv[0]), 2;
            if (!parse_float(value, 0.0f, 1000.0f, options.cfg))
            {
                std::cerr << "invalid --cfg value: " << value << "\n";
                return 2;
            }
        }
        else if (arg == "--seed")
        {
            std::string value;
            if (!next_value(value))
                return print_usage(argv[0]), 2;
            if (!parse_long_long(value, options.seed))
            {
                std::cerr << "invalid --seed value: " << value << "\n";
                return 2;
            }
        }
        else if (arg == "--threads")
        {
            std::string value;
            if (!next_value(value))
                return print_usage(argv[0]), 2;
            if (!parse_int(value, 1, 4096, options.threads))
            {
                std::cerr << "invalid --threads value: " << value << "\n";
                return 2;
            }
        }
        else if (arg == "--gpu" || arg == "-g")
        {
            std::string value;
            if (!next_value(value))
                return print_usage(argv[0]), 2;
            if (!parse_int(value, 0, 255, options.gpu))
            {
                std::cerr << "invalid --gpu value: " << value << "\n";
                return 2;
            }
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
    std::error_code filesystem_error;
    if (!std::filesystem::is_regular_file(options.input, filesystem_error))
    {
        std::cerr << "error: input video does not exist: " << options.input << "\n";
        return 1;
    }
    if (std::filesystem::absolute(options.input, filesystem_error)
        == std::filesystem::absolute(options.output, filesystem_error))
    {
        std::cerr << "error: input and output paths must be different.\n";
        return 1;
    }
    const std::filesystem::path output_parent = std::filesystem::path(options.output).parent_path();
    if (!output_parent.empty())
    {
        std::filesystem::create_directories(output_parent, filesystem_error);
        if (filesystem_error)
        {
            std::cerr << "error: failed to create output directory: " << output_parent << "\n";
            return 1;
        }
    }
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
    const std::size_t frame_count = raw_frames.size() / frame_bytes;
    if (frame_count == 0)
    {
        std::cerr << "error: input video has no decodable frames.\n";
        return 1;
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
    runtime.vulkan_device_index = options.gpu;
    runtime.sampling_steps = options.steps;
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
    if (!output.valid() || output.frames != static_cast<int>(frame_count))
    {
        std::cerr << "error: Engine returned an invalid frame count.\n";
        return 1;
    }
    const int out_frames = output.frames;
    if (!encode_video(options.output, options.input, out_rgb.data(),
                      out_frames, output.width, output.height, fps))
    {
        std::cerr << "error: ffmpeg encode failed.\n";
        return 1;
    }
    std::cout << "done: " << options.output << "\n";
    return 0;
}
