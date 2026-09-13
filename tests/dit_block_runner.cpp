// 本文件提供 SeedVR2 DiT 单个 Transformer block 的 NCNN 数值测试入口。
// 命令行输入为 block param/bin、FP32 vid[L_v,2560]、txt[L_t,2560]、
// timestep embedding[15360] 和视频 patch 形状 T/H/W；输出两个带 int32
// [rows,cols] 文件头的 FP32 矩阵。该 runner 只用于逐层对齐和定位误差，完整
// 产品接口由 SeedVR2Engine 负责 block 流式加载、CFG、sampler 与 VAE。可选的
// debug_prefix 会额外保存 QKV、QK Norm、RoPE/BF16 后 QKV、attention、projection
// 与最终输出，用于和 PyTorch 分阶段参考张量逐元素对齐。
#include "layers/seedvr2_dit_block.h"

#include <net.h>

#include <cstdint>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <utility>

#if NCNN_VULKAN
#include <gpu.h>
#endif

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
    if (argc != 13 && argc != 14)
    {
        std::fprintf(
            stderr,
            "usage: %s model.param model.bin vid.f32 vid_rows txt.f32 txt_rows "
            "emb.f32 T H W vid_out.f32 txt_out.f32 [debug_prefix]\n",
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

    const char* device = std::getenv("SEEDVR2_DEVICE");
    const bool use_vulkan = device && std::string(device) == "vulkan";
#if NCNN_VULKAN
    struct GpuInstanceGuard
    {
        explicit GpuInstanceGuard(bool enabled) : enabled(enabled)
        {
            if (enabled)
                ncnn::create_gpu_instance();
        }
        ~GpuInstanceGuard()
        {
            if (enabled)
                ncnn::destroy_gpu_instance();
        }
        bool enabled;
    } gpu_guard(use_vulkan);
#else
    if (use_vulkan)
    {
        std::fprintf(stderr, "runner was built without NCNN Vulkan support\n");
        return 4;
    }
#endif

    ncnn::Net net;
    net.opt.use_vulkan_compute = use_vulkan;
    net.opt.use_packing_layout = false;
    net.opt.use_fp16_packed = false;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;
    net.opt.num_threads = 8;
    net.register_custom_layer("SeedVR2DiTBlock", SeedVR2DiTBlock_layer_creator);
    if (net.load_param(argv[1]) != 0 || net.load_model(argv[2]) != 0)
    {
        std::fprintf(stderr, "failed to load block model\n");
        return 4;
    }
    SeedVR2DiTBlockDebugTensors debug_tensors;
    if (argc == 14)
    {
        if (use_vulkan)
        {
            std::fprintf(stderr, "intermediate capture currently supports the CPU path only\n");
            return 4;
        }
        auto* layer = dynamic_cast<SeedVR2DiTBlock*>(net.mutable_layers().back());
        if (!layer)
        {
            std::fprintf(stderr, "failed to locate SeedVR2DiTBlock for debug capture\n");
            return 4;
        }
        layer->set_debug_tensors(&debug_tensors);
    }
#if NCNN_VULKAN
    if (use_vulkan)
    {
        auto* layer = dynamic_cast<SeedVR2DiTBlock*>(net.mutable_layers().back());
        if (!layer)
            return 4;
        layer->set_runtime_shape(frames, height, width);
    }
#endif
    ncnn::Extractor extractor = net.create_extractor();
    if (extractor.input("vid", vid) != 0 || extractor.input("txt", txt) != 0
        || extractor.input("emb", embedding) != 0 || extractor.input("vid_shape", shape) != 0)
    {
        std::fprintf(stderr, "failed to bind block inputs\n");
        return 5;
    }
    ncnn::Mat vid_output;
    ncnn::Mat txt_output;
    const auto inference_start = std::chrono::steady_clock::now();
    const int vid_result = extractor.extract("vid_out", vid_output);
    const int txt_result = extractor.extract("txt_out", txt_output);
    const double inference_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - inference_start).count();
    if (vid_result != 0 || txt_result != 0)
    {
        std::fprintf(stderr, "block inference failed (vid=%d txt=%d)\n", vid_result, txt_result);
        return 6;
    }
    std::printf("block_forward_ms=%.3f\n", inference_ms);
    if (!write_matrix(argv[11], vid_output) || !write_matrix(argv[12], txt_output))
    {
        std::fprintf(stderr, "failed to write block outputs\n");
        return 7;
    }
    if (argc == 14)
    {
        const std::string prefix = argv[13];
        const std::pair<const char*, const ncnn::Mat*> captures[] = {
            {"qkv_vid", &debug_tensors.qkv_vid},
            {"qkv_txt", &debug_tensors.qkv_txt},
            {"norm_q_vid", &debug_tensors.norm_q_vid},
            {"norm_q_txt", &debug_tensors.norm_q_txt},
            {"norm_k_vid", &debug_tensors.norm_k_vid},
            {"norm_k_txt", &debug_tensors.norm_k_txt},
            {"attention_q", &debug_tensors.attention_q},
            {"attention_k", &debug_tensors.attention_k},
            {"attention_v", &debug_tensors.attention_v},
            {"attention_output", &debug_tensors.attention_output},
            {"projected_vid", &debug_tensors.projected_vid},
            {"projected_txt", &debug_tensors.projected_txt},
            {"output_vid", &debug_tensors.output_vid},
            {"output_txt", &debug_tensors.output_txt},
        };
        for (const auto& capture : captures)
        {
            if (!write_matrix(prefix + "_" + capture.first + ".f32", *capture.second))
            {
                std::fprintf(stderr, "failed to write debug capture %s\n", capture.first);
                return 8;
            }
        }
    }
    std::printf("vid=%dx%d txt=%dx%d\n", vid_output.h, vid_output.w, txt_output.h, txt_output.w);
    return 0;
}
