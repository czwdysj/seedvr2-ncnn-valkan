// 本文件提供 SeedVR2DiTOutput 的独立 NCNN 数值测试入口。
// 输入最后一个 block 的 FP32 视频 token、时间 embedding 和动态 patch T/H/W，
// 输出带 int32 [rows,cols] 文件头的反 patch token，用于验证输出头。
#include "layers/seedvr2_dit_output.h"

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
    if (value.dims != 2)
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
} // namespace

int main(int argc, char** argv)
{
    if (argc != 10)
    {
        std::fprintf(stderr, "usage: %s model.param model.bin vid.f32 vid_rows "
                             "emb.f32 T H W output.f32\n", argv[0]);
        return 2;
    }
    const int video_rows = std::atoi(argv[4]);
    ncnn::Mat video;
    ncnn::Mat embedding;
    if (!read_matrix(argv[3], video_rows, 2560, video)
        || !read_matrix(argv[5], 1, 15360, embedding))
        return 3;
    ncnn::Mat shape(3, static_cast<size_t>(4u));
    int* shape_data = shape;
    shape_data[0] = std::atoi(argv[6]);
    shape_data[1] = std::atoi(argv[7]);
    shape_data[2] = std::atoi(argv[8]);

    ncnn::Net net;
    net.opt.use_vulkan_compute = false;
    net.opt.use_packing_layout = false;
    net.opt.num_threads = 8;
    net.register_custom_layer("SeedVR2DiTOutput", SeedVR2DiTOutput_layer_creator);
    if (net.load_param(argv[1]) != 0 || net.load_model(argv[2]) != 0)
        return 4;
    ncnn::Extractor extractor = net.create_extractor();
    if (extractor.input("vid", video) != 0 || extractor.input("emb", embedding) != 0
        || extractor.input("vid_shape", shape) != 0)
        return 5;
    ncnn::Mat output;
    ncnn::Mat output_shape;
    const int r0 = extractor.extract("vid_out", output);
    const int r1 = extractor.extract("output_shape", output_shape);
    if (r0 != 0 || r1 != 0)
    {
        std::fprintf(stderr, "output inference failed (%d,%d)\n", r0, r1);
        return 6;
    }
    if (!write_matrix(argv[9], output))
        return 7;
    const int* result_shape = output_shape;
    std::printf("vid=%dx%d shape=%d,%d,%d\n", output.h, output.w,
                result_shape[0], result_shape[1], result_shape[2]);
    return 0;
}
