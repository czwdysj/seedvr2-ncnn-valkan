// 本文件提供 SeedVR2DiTInput 的独立 NCNN 数值测试入口。
// 它读取 FP32 视频/文本 token、标量 timestep 和动态 T/H/W，输出带 int32
// [rows,cols] 文件头的视频、文本和时间 embedding，供 PyTorch 逐层对齐。
#include "layers/seedvr2_dit_input.h"

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
    if (value.dims != 1 && value.dims != 2)
        return false;
    std::ofstream stream(path, std::ios::binary);
    if (!stream)
        return false;
    const int32_t shape[2] = {value.dims == 2 ? value.h : 1, value.w};
    stream.write(reinterpret_cast<const char*>(shape), sizeof(shape));
    stream.write(static_cast<const char*>(value.data),
                 static_cast<std::streamsize>(shape[0]) * shape[1] * sizeof(float));
    return stream.good();
}
} // namespace

int main(int argc, char** argv)
{
    if (argc != 12)
    {
        std::fprintf(stderr, "usage: %s model.param model.bin vid.f32 vid_rows "
                             "txt.f32 txt_rows timestep T H W output_prefix\n", argv[0]);
        return 2;
    }
    const int video_rows = std::atoi(argv[4]);
    const int text_rows = std::atoi(argv[6]);
    ncnn::Mat video;
    ncnn::Mat text;
    if (!read_matrix(argv[3], video_rows, 33, video)
        || !read_matrix(argv[5], text_rows, 5120, text))
        return 3;
    ncnn::Mat timestep(1, static_cast<size_t>(4u));
    static_cast<float*>(timestep.data)[0] = std::strtof(argv[7], nullptr);
    ncnn::Mat shape(3, static_cast<size_t>(4u));
    int* shape_data = shape;
    shape_data[0] = std::atoi(argv[8]);
    shape_data[1] = std::atoi(argv[9]);
    shape_data[2] = std::atoi(argv[10]);

    ncnn::Net net;
    net.opt.use_vulkan_compute = false;
    net.opt.use_packing_layout = false;
    net.opt.num_threads = 8;
    net.register_custom_layer("SeedVR2DiTInput", SeedVR2DiTInput_layer_creator);
    if (net.load_param(argv[1]) != 0 || net.load_model(argv[2]) != 0)
        return 4;
    ncnn::Extractor extractor = net.create_extractor();
    if (extractor.input("vid", video) != 0 || extractor.input("txt", text) != 0
        || extractor.input("timestep", timestep) != 0 || extractor.input("vid_shape", shape) != 0)
        return 5;
    ncnn::Mat video_output;
    ncnn::Mat text_output;
    ncnn::Mat embedding;
    ncnn::Mat patched_shape;
    const int r0 = extractor.extract("vid_out", video_output);
    const int r1 = extractor.extract("txt_out", text_output);
    const int r2 = extractor.extract("emb", embedding);
    const int r3 = extractor.extract("patched_shape", patched_shape);
    if (r0 != 0 || r1 != 0 || r2 != 0 || r3 != 0)
    {
        std::fprintf(stderr, "input inference failed (%d,%d,%d,%d)\n", r0, r1, r2, r3);
        return 6;
    }
    const std::string prefix = argv[11];
    if (!write_matrix(prefix + "_vid.f32", video_output)
        || !write_matrix(prefix + "_txt.f32", text_output)
        || !write_matrix(prefix + "_emb.f32", embedding))
        return 7;
    const int* result_shape = patched_shape;
    std::printf("vid=%dx%d txt=%dx%d emb=%d shape=%d,%d,%d\n",
                video_output.h, video_output.w, text_output.h, text_output.w,
                embedding.w, result_shape[0], result_shape[1], result_shape[2]);
    return 0;
}
