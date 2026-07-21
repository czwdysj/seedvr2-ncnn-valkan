// 本文件是 SeedVR2 VAE 动态形状 NCNN 子图的命令行测试入口。
// 它读取一个按 C,T,H,W 逻辑顺序保存的 FP32 raw 输入张量，注册 VAE 需要的
// 自定义 CPU layer，执行 NCNN 推理，然后把输出形状和 FP32 raw 结果写回文件。
// 这个 runner 面向“PyTorch reference vs NCNN 子图”的数值对齐，不是最终对外 API。
#include <net.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "dynamic_framewise_group_norm.h"
#include "dynamic_framewise_spatial_attention.h"
#include "dynamic_space_time_shuffle.h"

static bool read_raw(const char* path, ncnn::Mat& input)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        return false;
    // ncnn::Mat 的 cstep 可能包含对齐 padding，不能把整块内存直接读入。
    // 这里逐 C、逐 T、逐 H 行读取，只交换真实的 C,T,H,W 逻辑元素。
    for (int channel = 0; channel < input.c; channel++)
        for (int frame = 0; frame < input.d; frame++)
            for (int row = 0; row < input.h; row++)
                stream.read(
                    reinterpret_cast<char*>(input.channel(channel).depth(frame).row(row)),
                    input.w * sizeof(float));
    return stream.good();
}

static bool write_raw(const char* path, const ncnn::Mat& output)
{
    std::ofstream stream(path, std::ios::binary);
    if (!stream)
        return false;
    // 输出文件先写 C,T,H,W 四个 int32，测试脚本据此还原 NCNN 输出张量形状。
    const int32_t shape[4] = {output.c, output.d, output.h, output.w};
    stream.write(reinterpret_cast<const char*>(shape), sizeof(shape));
    for (int channel = 0; channel < output.c; channel++)
        for (int frame = 0; frame < output.d; frame++)
            for (int row = 0; row < output.h; row++)
                stream.write(
                    reinterpret_cast<const char*>(output.channel(channel).depth(frame).row(row)),
                    output.w * sizeof(float));
    return stream.good();
}

int main(int argc, char** argv)
{
    if (argc != 10)
    {
        std::fprintf(
            stderr,
            "usage: %s param bin C T H W input.raw output.raw threads\n",
            argv[0]);
        return 2;
    }

    const int channels = std::atoi(argv[3]);
    const int frames = std::atoi(argv[4]);
    const int height = std::atoi(argv[5]);
    const int width = std::atoi(argv[6]);
    const int threads = std::atoi(argv[9]);
    // ncnn 4D Mat 的构造顺序是 w,h,d,c；在本项目中固定解释为 W,H,T,C。
    ncnn::Mat input(width, height, frames, channels, 4u, 1);
    if (input.empty() || !read_raw(argv[7], input))
    {
        std::fprintf(stderr, "failed to read input tensor\n");
        return 3;
    }

    ncnn::Net net;
    net.opt.num_threads = threads;
    // 当前对齐链路使用 FP32 标量布局，避免 packing/fp16 引入额外误差来源。
    net.opt.use_packing_layout = false;
    net.opt.use_fp16_packed = false;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;
    // 这些 layer 是 pnnx 静态图不容易表达的 VAE 动态行为：
    // 逐帧 GroupNorm、逐帧空间注意力、以及根据运行时尺寸展开的时空 shuffle。
    net.register_custom_layer("DynamicFramewiseGroupNorm", DynamicFramewiseGroupNorm_layer_creator);
    net.register_custom_layer(
        "DynamicFramewiseSpatialAttention", DynamicFramewiseSpatialAttention_layer_creator);
    net.register_custom_layer("DynamicSpaceTimeShuffle", DynamicSpaceTimeShuffle_layer_creator);
    const int param_code = net.load_param(argv[1]);
    const int model_code = param_code == 0 ? net.load_model(argv[2]) : param_code;
    if (param_code != 0 || model_code != 0)
    {
        std::fprintf(stderr, "load failed: param=%d model=%d\n", param_code, model_code);
        return 4;
    }

    ncnn::Extractor extractor = net.create_extractor();
    if (extractor.input(net.input_names()[0], input) != 0)
        return 5;
    ncnn::Mat output;
    const int extract_code = extractor.extract(net.output_names()[0], output);
    if (extract_code != 0 || !write_raw(argv[8], output))
    {
        std::fprintf(stderr, "extract/write failed: %d\n", extract_code);
        return 6;
    }
    std::printf("output C,T,H,W = %d,%d,%d,%d\n", output.c, output.d, output.h, output.w);
    return 0;
}
