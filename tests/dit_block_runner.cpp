// 本文件提供 SeedVR2 DiT 单个 Transformer block 的 NCNN 数值测试入口。
// 命令行输入为 block param/bin、FP32 vid[L_v,2560]、txt[L_t,2560]、
// timestep embedding[15360] 和视频 patch 形状 T/H/W；输出两个带 int32
// [rows,cols] 文件头的 FP32 矩阵。该 runner 只用于逐层对齐和定位误差，完整
// 产品接口由后续 SeedVR2Engine 负责 block 流式加载、CFG、sampler 与 VAE。
#include "seedvr2_dit_block.h"

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
    if (output.empty())
        return false;
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        return false;
    stream.read(static_cast<char*>(output.data), static_cast<std::streamsize>(rows) * columns * sizeof(float));
    return stream.good() || stream.eof() && stream.gcount() == static_cast<std::streamsize>(rows) * columns * sizeof(float);
}

bool write_matrix(const std::string& path, const ncnn::Mat& value)
{
    if (value.dims != 2 || value.elempack != 1 || value.elemsize != 4u)
        return false;
    std::ofstream stream(path, std::ios::binary);
    if (!stream)
        return false;
    const int32_t shape[2] = {value.h, value.w};
    stream.write(reinterpret_cast<const char*>(shape), sizeof(shape));
    stream.write(static_cast<const char*>(value.data), static_cast<std::streamsize>(value.h) * value.w * sizeof(float));
    return stream.good();
}
} // namespace

int main(int argc, char** argv)
{
    if (argc != 13)
    {
        std::fprintf(
            stderr,
            "usage: %s model.param model.bin vid.f32 vid_rows txt.f32 txt_rows "
            "emb.f32 T H W vid_out.f32 txt_out.f32\n",
            argv[0]);
        return 2;
    }
    const int vid_rows = std::atoi(argv[4]);
    const int txt_rows = std::atoi(argv[6]);
    const int frames = std::atoi(argv[8]);
    const int height = std::atoi(argv[9]);
    const int width = std::atoi(argv[10]);

    ncnn::Mat vid;
    ncnn::Mat txt;
    ncnn::Mat embedding;
    if (!read_matrix(argv[3], vid_rows, 2560, vid)
        || !read_matrix(argv[5], txt_rows, 2560, txt)
        || !read_matrix(argv[7], 1, 15360, embedding))
    {
        std::fprintf(stderr, "failed to read input matrices\n");
        return 3;
    }
    ncnn::Mat shape(3, static_cast<size_t>(4u));
    static_cast<int*>(shape.data)[0] = frames;
    static_cast<int*>(shape.data)[1] = height;
    static_cast<int*>(shape.data)[2] = width;

    ncnn::Net net;
    net.opt.use_vulkan_compute = false;
    net.opt.use_packing_layout = false;
    net.opt.num_threads = 8;
    net.register_custom_layer("SeedVR2DiTBlock", SeedVR2DiTBlock_layer_creator);
    if (net.load_param(argv[1]) != 0 || net.load_model(argv[2]) != 0)
    {
        std::fprintf(stderr, "failed to load block model\n");
        return 4;
    }
    ncnn::Extractor extractor = net.create_extractor();
    if (extractor.input("vid", vid) != 0 || extractor.input("txt", txt) != 0
        || extractor.input("emb", embedding) != 0 || extractor.input("vid_shape", shape) != 0)
    {
        std::fprintf(stderr, "failed to bind block inputs\n");
        return 5;
    }
    ncnn::Mat vid_output;
    ncnn::Mat txt_output;
    const int vid_result = extractor.extract("vid_out", vid_output);
    const int txt_result = extractor.extract("txt_out", txt_output);
    if (vid_result != 0 || txt_result != 0)
    {
        std::fprintf(stderr, "block inference failed (vid=%d txt=%d)\n", vid_result, txt_result);
        return 6;
    }
    if (!write_matrix(argv[11], vid_output) || !write_matrix(argv[12], txt_output))
    {
        std::fprintf(stderr, "failed to write block outputs\n");
        return 7;
    }
    std::printf("vid=%dx%d txt=%dx%d\n", vid_output.h, vid_output.w, txt_output.h, txt_output.w);
    return 0;
}
