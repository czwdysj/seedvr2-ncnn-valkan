// 本文件是 DynamicFramewiseGroupNorm 的 Vulkan 数值对齐测试 runner。
//
// 目标：证明 Vulkan 路径与 CPU 基线（已对过 PyTorch reference）逐元素一致，
// 重点验证「逐帧统计」语义——每个帧独立计算分组均值/方差，不跨帧混合——
// 以及 cstep 对齐、group 划分、单帧边界等关键情况。
//
// 做法：运行时生成一个最小单层 ncnn 图，分别用 CPU 与 Vulkan 后端各 forward
// 一次，比较输出张量的最大绝对误差。覆盖多组参数以踩到关键边界：
//   - channels_per_group = 1/2/3（不同 group 划分，含非 2 次幂）
//   - T = 1（单帧）/ 2/3/4（多帧，验证逐帧不串扰）
//   - W = 5/7（非 4 对齐，覆盖 cstep 按 16 字节对齐后的地址计算）
//
// 注意：groups 在本层构造函数固定为 32，故 channels 必须是 32 的倍数。
#include "layers/dynamic_framewise_group_norm.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "layer.h"
#include "mat.h"
#include "net.h"

namespace
{

// 一组测试的参数：通道数 / 输入帧数 / 高 / 宽。
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
    FILE* fp = fopen(param_path, "wb");
    if (!fp)
        return false;
    fprintf(fp, "7767517\n");
    fprintf(fp, "2 2\n");
    fprintf(fp, "Input                    in0                      0 1 in0\n");
    // -23310=1,C -> pd id 10 的数组 [C]（bias shape，load_param 用它恢复 channels）
    fprintf(fp, "DynamicFramewiseGroupNorm norm                  1 1 in0 out0 -23310=1,%d\n", C);
    fclose(fp);

    // bin：先 bias[C] 再 weight[C]（与 load_model 读取顺序一致）。
    std::vector<float> bias(C);
    std::vector<float> weight(C);
    for (int i = 0; i < C; i++)
        bias[i] = 0.02f * static_cast<float>((i % 5) - 2);       // [-0.04, 0.04]
    for (int i = 0; i < C; i++)
        weight[i] = 0.5f + 0.05f * static_cast<float>(i % 11);    // [0.5, 1.0]

    fp = fopen(bin_path, "wb");
    if (!fp)
        return false;
    fwrite(bias.data(), sizeof(float), bias.size(), fp);
    fwrite(weight.data(), sizeof(float), weight.size(), fp);
    fclose(fp);
    return true;
}

ncnn::Mat make_input(int W, int H, int T, int C)
{
    ncnn::Mat in(W, H, T, C, 4u, 1);
    // 注意：4D Mat 通道间存在 cstep 对齐填充，必须按通道逐个平面写入。
    size_t idx = 0;
    for (int c = 0; c < C; c++)
    {
        float* plane = in.channel(c);
        const size_t plane_size = static_cast<size_t>(W) * H * T;
        for (size_t i = 0; i < plane_size; i++)
            // 显式转 int 再做减法，避免 size_t 无符号下溢成巨大值。
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
    const char* param_path = "/tmp/groupnorm_vulkan_probe.param";
    const char* bin_path = "/tmp/groupnorm_vulkan_probe.bin";
    if (!write_param_bin(tc.C, param_path, bin_path))
        return -1;

    ncnn::Net net;
    net.register_custom_layer("DynamicFramewiseGroupNorm", DynamicFramewiseGroupNorm_layer_creator);
    net.opt.use_vulkan_compute = use_vulkan;
    // 标量布局：与 RuntimeContext 契约一致，避免 packing 干扰数值比对。
    net.opt.use_packing_layout = false;
    // 三个 fp16 开关必须全部显式关闭，否则 GPU buffer 以半精度打包存储，
    // 与 fp32 CPU 基线比对会引入精度噪声（第一个算子踩坑的根因）。
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
        {32, 2, 3, 5, "C=32 cpg=1 T=2 W=5(非对齐)"},
        {64, 3, 4, 4, "C=64 cpg=2 T=3 W=4(对齐)"},
        {32, 1, 3, 5, "C=32 cpg=1 T=1(单帧)"},
        {96, 4, 2, 7, "C=96 cpg=3 T=4 W=7(非对齐,cpg非2幂)"},
        {512, 2, 16, 16, "C=512 cpg=16 T=2 W=16(count=4096>256,strided)"},
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
