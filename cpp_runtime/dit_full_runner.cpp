// 本文件提供 SeedVR2 3B 完整 DiT 的 NCNN CPU 流式推理入口。
// 输入层、32 个 Transformer block 和输出层均加载真实 param/bin；block
// 每次只保留一个实例，输出 token 传给下一层后立即释放当前权重，从而避免
// 3B 权重同时常驻内存。输入 H/W 动态，但必须满足原模型 2x2 patch 约束。
#include "seedvr2_dit_block.h"
#include "seedvr2_dit_input.h"
#include "seedvr2_dit_output.h"

#include <net.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

namespace
{
bool read_matrix(const std::string& path, int rows, int columns, ncnn::Mat& output)
{
    output.create(columns, rows, static_cast<size_t>(4u));
    std::ifstream stream(path, std::ios::binary);
    if (output.empty() || !stream)
        return false;
    const std::streamsize bytes = static_cast<std::streamsize>(rows) * columns * sizeof(float);
    stream.read(static_cast<char*>(output.data), bytes);
    return stream.gcount() == bytes;
}

bool write_matrix(const std::string& path, const ncnn::Mat& value)
{
    if (value.dims != 2 || value.elemsize != 4u || value.elempack != 1)
        return false;
    std::ofstream stream(path, std::ios::binary);
    if (!stream)
        return false;
    const int32_t shape[2] = {value.h, value.w};
    stream.write(reinterpret_cast<const char*>(shape), sizeof(shape));
    stream.write(static_cast<const char*>(value.data),
                 static_cast<std::streamsize>(value.h) * value.w * sizeof(float));
    return stream.good();
}

void configure_cpu(ncnn::Net& net)
{
    net.opt.use_vulkan_compute = false;
    net.opt.use_packing_layout = false;
    net.opt.num_threads = 8;
}

int run_input(const std::string& model_dir, const ncnn::Mat& input_video,
              const ncnn::Mat& input_text, const ncnn::Mat& timestep,
              const ncnn::Mat& input_shape, ncnn::Mat& video, ncnn::Mat& text,
              ncnn::Mat& embedding, ncnn::Mat& patched_shape)
{
    ncnn::Net net;
    configure_cpu(net);
    net.register_custom_layer("SeedVR2DiTInput", SeedVR2DiTInput_layer_creator);
    if (net.load_param((model_dir + "/seedvr2_dit_input.ncnn.param").c_str()) != 0
        || net.load_model((model_dir + "/seedvr2_dit_input.ncnn.bin").c_str()) != 0)
        return -1;
    ncnn::Extractor ex = net.create_extractor();
    if (ex.input("vid", input_video) != 0 || ex.input("txt", input_text) != 0
        || ex.input("timestep", timestep) != 0 || ex.input("vid_shape", input_shape) != 0)
        return -2;
    if (ex.extract("vid_out", video) != 0 || ex.extract("txt_out", text) != 0
        || ex.extract("emb", embedding) != 0 || ex.extract("patched_shape", patched_shape) != 0)
        return -3;
    return 0;
}

int run_block(const std::string& model_dir, int index, const ncnn::Mat& video,
              const ncnn::Mat& text, const ncnn::Mat& embedding,
              const ncnn::Mat& shape, ncnn::Mat& video_output,
              ncnn::Mat& text_output)
{
    char stem[64];
    std::snprintf(stem, sizeof(stem), "seedvr2_dit_block_%02d.ncnn", index);
    ncnn::Net net;
    configure_cpu(net);
    net.register_custom_layer("SeedVR2DiTBlock", SeedVR2DiTBlock_layer_creator);
    if (net.load_param((model_dir + "/" + stem + ".param").c_str()) != 0
        || net.load_model((model_dir + "/" + stem + ".bin").c_str()) != 0)
        return -1;
    ncnn::Extractor ex = net.create_extractor();
    if (ex.input("vid", video) != 0 || ex.input("txt", text) != 0
        || ex.input("emb", embedding) != 0 || ex.input("vid_shape", shape) != 0)
        return -2;
    if (ex.extract("vid_out", video_output) != 0 || ex.extract("txt_out", text_output) != 0)
        return -3;
    return 0;
}

int run_output(const std::string& model_dir, const ncnn::Mat& video,
               const ncnn::Mat& embedding, const ncnn::Mat& shape,
               ncnn::Mat& output)
{
    ncnn::Net net;
    configure_cpu(net);
    net.register_custom_layer("SeedVR2DiTOutput", SeedVR2DiTOutput_layer_creator);
    if (net.load_param((model_dir + "/seedvr2_dit_output.ncnn.param").c_str()) != 0
        || net.load_model((model_dir + "/seedvr2_dit_output.ncnn.bin").c_str()) != 0)
        return -1;
    ncnn::Extractor ex = net.create_extractor();
    if (ex.input("vid", video) != 0 || ex.input("emb", embedding) != 0
        || ex.input("vid_shape", shape) != 0)
        return -2;
    ncnn::Mat output_shape;
    if (ex.extract("vid_out", output) != 0 || ex.extract("output_shape", output_shape) != 0)
        return -3;
    return 0;
}
} // namespace

int main(int argc, char** argv)
{
    if (argc != 12)
    {
        std::fprintf(stderr, "usage: %s io_model_dir block_model_dir vid.f32 vid_rows "
                             "txt.f32 txt_rows timestep T H W output.f32\n", argv[0]);
        return 2;
    }
    ncnn::Mat input_video;
    ncnn::Mat input_text;
    if (!read_matrix(argv[3], std::atoi(argv[4]), 33, input_video)
        || !read_matrix(argv[5], std::atoi(argv[6]), 5120, input_text))
        return 3;
    ncnn::Mat timestep(1, static_cast<size_t>(4u));
    static_cast<float*>(timestep.data)[0] = std::strtof(argv[7], nullptr);
    ncnn::Mat input_shape(3, static_cast<size_t>(4u));
    int* shape_data = input_shape;
    shape_data[0] = std::atoi(argv[8]);
    shape_data[1] = std::atoi(argv[9]);
    shape_data[2] = std::atoi(argv[10]);

    ncnn::Mat video;
    ncnn::Mat text;
    ncnn::Mat embedding;
    ncnn::Mat patched_shape;
    const int input_result = run_input(argv[1], input_video, input_text, timestep,
                                       input_shape, video, text, embedding, patched_shape);
    if (input_result != 0)
    {
        std::fprintf(stderr, "DiT input failed (%d)\n", input_result);
        return 4;
    }
    for (int index = 0; index < 32; index++)
    {
        ncnn::Mat video_output;
        ncnn::Mat text_output;
        const int result = run_block(argv[2], index, video, text, embedding,
                                     patched_shape, video_output, text_output);
        if (result != 0)
        {
            std::fprintf(stderr, "DiT block %02d failed (%d)\n", index, result);
            return 5;
        }
        video = video_output;
        text = text_output;
        std::printf("block %02d complete\n", index);
        std::fflush(stdout);
    }
    ncnn::Mat output;
    const int output_result = run_output(argv[1], video, embedding, patched_shape, output);
    if (output_result != 0)
    {
        std::fprintf(stderr, "DiT output failed (%d)\n", output_result);
        return 6;
    }
    if (!write_matrix(argv[11], output))
        return 7;
    std::printf("DiT complete: %dx%d\n", output.h, output.w);
    return 0;
}
