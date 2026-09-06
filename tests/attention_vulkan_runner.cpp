// 本文件是 DynamicFramewiseSpatialAttention 的 Vulkan 数值对齐测试 runner。
//
// 目标：证明 Vulkan 路径与 CPU 基线逐元素一致，重点验证：
//   - 逐帧独立注意力（不同帧的 QK^T/softmax 不串扰）；
//   - softmax 的数值稳定（max 减法防溢出）；
//   - GroupNorm 归约 + Q/K/V 投影 + 注意力 + 输出投影 + 残差的全链路。
//
// 做法：运行时生成一个最小单层 ncnn 图，分别用 CPU 与 Vulkan 后端各 forward
// 一次，比较输出张量的最大绝对误差。覆盖多组参数：
//   - channels = 32/64（必须能被 32 整除，因 GroupNorm 用 32 组）；
//   - tokens = W*H = 15/16/25/64（含非方阵、单帧、非 4 对齐）；
//   - 权重用小值，保证 QK^T 量级可控、softmax 不溢出。
#include "dynamic_framewise_spatial_attention.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "layer.h"
#include "mat.h"
#include "net.h"

namespace
{

struct TestCase
{
    int C;
    int T;
    int H;
    int W;
    const char* name;
};

bool write_param_bin(int C, const char* param_path, const char* bin_path)
{
    const int matrix = C * C;
    FILE* fp = fopen(param_path, "wb");
    if (!fp)
        return false;
    fprintf(fp, "7767517\n");
    fprintf(fp, "2 2\n");
    fprintf(fp, "Input                    in0                      0 1 in0\n");
    // load_param 只读 pd.get(10)（norm bias shape）恢复 channels。
    fprintf(fp, "DynamicFramewiseSpatialAttention attn         1 1 in0 out0 -23310=1,%d\n", C);
    fclose(fp);

    // bin：按 load_model 的固定顺序写入 10 个权重。
    auto fill = [](std::vector<float>& v, float base, float step) {
        for (size_t i = 0; i < v.size(); i++)
            v[i] = base + step * static_cast<float>((i % 7) - 3);
    };
    std::vector<float> norm_bias(C), norm_weight(C);
    std::vector<float> q_bias(C), q_weight(matrix);
    std::vector<float> k_bias(C), k_weight(matrix);
    std::vector<float> v_bias(C), v_weight(matrix);
    std::vector<float> out_bias(C), out_weight(matrix);
    fill(norm_bias, 0.f, 0.02f);
    fill(norm_weight, 1.f, 0.02f);      // norm 权重接近 1，避免放大
    fill(q_bias, 0.f, 0.01f);
    fill(q_weight, 0.f, 0.005f);        // 小权重，QK^T 量级可控
    fill(k_bias, 0.f, 0.01f);
    fill(k_weight, 0.f, 0.005f);
    fill(v_bias, 0.f, 0.01f);
    fill(v_weight, 0.f, 0.005f);
    fill(out_bias, 0.f, 0.01f);
    fill(out_weight, 0.f, 0.005f);

    fp = fopen(bin_path, "wb");
    if (!fp)
        return false;
    const std::vector<float>* order[] = {&norm_bias, &norm_weight, &k_bias, &k_weight,
                                         &out_bias, &out_weight, &q_bias, &q_weight,
                                         &v_bias, &v_weight};
    for (const std::vector<float>* v : order)
        fwrite(v->data(), sizeof(float), v->size(), fp);
    fclose(fp);
    return true;
}

ncnn::Mat make_input(int W, int H, int T, int C)
{
    ncnn::Mat in(W, H, T, C, 4u, 1);
    // 4D Mat 通道间存在 cstep 对齐填充，必须按通道逐个平面写入。
    size_t idx = 0;
    for (int c = 0; c < C; c++)
    {
        float* plane = in.channel(c);
        const size_t plane_size = static_cast<size_t>(W) * H * T;
        for (size_t i = 0; i < plane_size; i++)
            plane[i] = static_cast<float>(static_cast<int>(idx++ % 11) - 5) * 0.1f;
    }
    return in;
}

double max_abs_diff(const ncnn::Mat& a, const ncnn::Mat& b)
{
    if (a.dims != b.dims || a.w != b.w || a.h != b.h || a.d != b.d || a.c != b.c)
        return 1e30;
    double worst = 0.0;
    for (int c = 0; c < a.c; c++)
    {
        const float* pa = a.channel(c);
        const float* pb = b.channel(c);
        const size_t plane = static_cast<size_t>(a.w) * a.h * a.d;
        for (size_t i = 0; i < plane; i++)
        {
            const double d = static_cast<double>(pa[i]) - static_cast<double>(pb[i]);
            if (d > worst)
                worst = d;
            else if (-d > worst)
                worst = -d;
        }
    }
    return worst;
}

int run(bool use_vulkan, const TestCase& tc, ncnn::Mat& out)
{
    const char* param_path = "/tmp/attention_vulkan_probe.param";
    const char* bin_path = "/tmp/attention_vulkan_probe.bin";
    if (!write_param_bin(tc.C, param_path, bin_path))
        return -1;

    ncnn::Net net;
    net.register_custom_layer("DynamicFramewiseSpatialAttention", DynamicFramewiseSpatialAttention_layer_creator);
    net.opt.use_vulkan_compute = use_vulkan;
    net.opt.use_packing_layout = false;
    net.opt.use_fp16_packed = false;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;
    net.opt.num_threads = 1;

    if (net.load_param(param_path) != 0 || net.load_model(bin_path) != 0)
    {
        std::fprintf(stderr, "load failed (vulkan=%d)\n", use_vulkan ? 1 : 0);
        return -1;
    }

    ncnn::Mat in = make_input(tc.W, tc.H, tc.T, tc.C);
    ncnn::Extractor ex = net.create_extractor();
    if (ex.input("in0", in) != 0 || ex.extract("out0", out) != 0)
    {
        std::fprintf(stderr, "forward failed (vulkan=%d)\n", use_vulkan ? 1 : 0);
        return -1;
    }
    return 0;
}

} // namespace

int main()
{
    const TestCase cases[] = {
        {32, 2, 4, 4, "C=32 T=2 tokens=16"},
        {64, 3, 5, 5, "C=64 T=3 tokens=25"},
        {32, 1, 8, 8, "C=32 T=1 tokens=64(单帧)"},
        {64, 2, 3, 5, "C=64 T=2 tokens=15(非方阵,非4对齐)"},
    };
    const int ncase = static_cast<int>(sizeof(cases) / sizeof(cases[0]));

#if NCNN_VULKAN
    ncnn::create_gpu_instance();
#endif

    int pass = 0;
    int fail = 0;
    for (int i = 0; i < ncase; i++)
    {
        const TestCase& tc = cases[i];

        ncnn::Mat out_cpu;
        if (run(false, tc, out_cpu) != 0)
        {
            std::printf("[case %d] %s : CPU 运行失败\n", i, tc.name);
            fail++;
            continue;
        }

#if NCNN_VULKAN
        ncnn::Mat out_gpu;
        if (run(true, tc, out_gpu) != 0)
        {
            std::printf("[case %d] %s : Vulkan 运行失败\n", i, tc.name);
            fail++;
            continue;
        }
        const double diff = max_abs_diff(out_cpu, out_gpu);
        const bool ok = diff < 1e-4;
        std::printf("[case %d] %s : max|diff| = %.6f %s\n", i, tc.name, diff, ok ? "PASS" : "FAIL");
        ok ? pass++ : fail++;
#else
        std::printf("[case %d] %s : CPU shape (%d,%d,%d,%d) ok\n",
                    i, tc.name, out_cpu.w, out_cpu.h, out_cpu.d, out_cpu.c);
        pass++;
#endif
    }

    std::printf("======== 结果: %d pass, %d fail ========\n", pass, fail);
#if NCNN_VULKAN
    ncnn::destroy_gpu_instance();
#endif
    return fail == 0 ? 0 : 1;
}
