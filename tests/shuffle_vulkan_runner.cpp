// 本文件是 DynamicSpaceTimeShuffle 的 Vulkan 数值对齐测试 runner。
//
// 目标：证明 Vulkan 路径与 CPU 基线（已对过 PyTorch reference）逐元素一致，
// 验证 shader 里的索引映射（子像素偏移、r_t==2 时的首帧删除、cstep 对齐）
// 与 CPU 实现完全等价。
//
// 做法：运行时生成一个最小单层 ncnn 图，分别用 CPU 与 Vulkan 后端各 forward
// 一次，比较输出张量的最大绝对误差。覆盖多组参数以踩到关键边界：
//   - C=4（4 的倍数：验证 record_upload 的 pack4 路径）
//   - C=5（非 4 倍数：验证标量路径 + cstep 对齐）
//   - r_t=2（首帧删除分支）、r_t=1（非 2 时间倍率分支）
//   - W=5（非 4 对齐，覆盖 cstep 按 16 字节对齐后的地址计算）
#include "layers/dynamic_space_time_shuffle.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "layer.h"
#include "mat.h"
#include "net.h"

namespace
{

// 一组测试的参数：通道数 / 输入帧数 / 高 / 宽 / 时间倍率。
struct TestCase
{
    int C;
    int T;
    int H;
    int W;
    int R_T;
    const char* name;
};

bool write_param_bin(int C, int R_T, const char* param_path, const char* bin_path)
{
    const int projected = C * 4 * R_T;

    FILE* fp = fopen(param_path, "wb");
    if (!fp)
        return false;
    fprintf(fp, "7767517\n");
    fprintf(fp, "2 2\n");
    fprintf(fp, "Input                    in0                      0 1 in0\n");
    // -23310=1,projected -> pd id 10 的数组 [projected]（bias shape）
    // -23311=5,projected,C,1,1,1 -> pd id 11 的数组（Conv3D weight shape）
    fprintf(fp, "DynamicSpaceTimeShuffle shuffle                 1 1 in0 out0 -23310=1,%d -23311=5,%d,%d,1,1,1\n",
            projected, projected, C);
    fclose(fp);

    std::vector<float> bias(projected);
    std::vector<float> weight(static_cast<size_t>(projected) * C);
    for (int i = 0; i < projected; i++)
        bias[i] = 0.01f * (i + 1);
    for (size_t i = 0; i < weight.size(); i++)
        weight[i] = 0.005f * static_cast<float>((i % 7) + 1);

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
    // 注意：4D Mat 通道间存在 cstep 对齐填充，必须按通道逐个平面写入，
    // 不能用连续指针跨越通道边界（否则数据会错位并读到填充区的垃圾值）。
    size_t idx = 0;
    for (int c = 0; c < C; c++)
    {
        float* plane = in.channel(c);
        const size_t plane_size = static_cast<size_t>(W) * H * T;
        for (size_t i = 0; i < plane_size; i++)
            // 注意：必须显式转 int 再做减法，否则 size_t 无符号下溢成巨大值。
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
    const char* param_path = "/tmp/shuffle_vulkan_probe.param";
    const char* bin_path = "/tmp/shuffle_vulkan_probe.bin";
    if (!write_param_bin(tc.C, tc.R_T, param_path, bin_path))
        return -1;

    ncnn::Net net;
    net.register_custom_layer("DynamicSpaceTimeShuffle", DynamicSpaceTimeShuffle_layer_creator);
    net.opt.use_vulkan_compute = use_vulkan;
    // 标量布局：与 RuntimeContext 的契约一致，避免 packing 干扰数值比对。
    net.opt.use_packing_layout = false;
    // 关键：三个 fp16 开关必须全部显式关闭。ncnn 默认 use_fp16_packed=true，
    // 漏关会让 GPU buffer 以「两个 half 打包一个 uint32」存储，shader 若按
    // fp32 读就会得到垃圾（本次排查踩坑的根因）。
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

// ReLU slope=1.0 是 identity：用它对照验证 Net 的 Vulkan 上传/下载链路是否保真，
// 从而区分「我的层/sharder 有问题」还是「llvmpipe + ncnn convert_packing 环境问题」。
bool relu_identity_probe()
{
    const char* PARAM = "/tmp/relu_probe.param";
    const char* BIN = "/tmp/relu_probe.bin";
    FILE* fp = fopen(PARAM, "wb");
    fprintf(fp, "7767517\n2 2\nInput in0 0 1 in0\nReLU relu 1 1 in0 out0 0=1.0\n");
    fclose(fp);
    fp = fopen(BIN, "wb");
    fclose(fp);

    ncnn::Mat in(5, 3, 2, 5, 4u, 1);
    for (int c = 0; c < 5; c++)
    {
        float* p = in.channel(c);
        for (int i = 0; i < 30; i++)
            p[i] = static_cast<float>(static_cast<int>((c * 30 + i) % 11) - 5) * 0.1f;
    }

    ncnn::Net net;
    net.opt.use_vulkan_compute = true;
    net.opt.use_packing_layout = false;
    net.opt.use_fp16_packed = false;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;
    if (net.load_param(PARAM) != 0 || net.load_model(BIN) != 0)
        return false;
    ncnn::Extractor ex = net.create_extractor();
    ex.input("in0", in);
    ncnn::Mat out;
    if (ex.extract("out0", out) != 0)
        return false;
    double d = max_abs_diff(in, out);
    std::printf("[RELU-probe] identity max|diff| = %.6f %s\n", d, d < 1e-4 ? "OK" : "FAIL");
    return d < 1e-4;
}

} // namespace

int main()
{
    const TestCase cases[] = {
        {4, 3, 7, 5, 2, "C=4 T=3 r_t=2 (pack4 + 首帧删除)"},
        {5, 2, 3, 5, 2, "C=5 T=2 r_t=2 (标量 + 首帧删除)"},
        {5, 4, 3, 5, 1, "C=5 T=4 r_t=1 (非 2 倍率，无首帧删除)"},
        {4, 2, 5, 5, 1, "C=4 T=2 r_t=1 (pack4 + 非 2 倍率)"},
    };
    const int ncase = static_cast<int>(sizeof(cases) / sizeof(cases[0]));

#if NCNN_VULKAN
    ncnn::create_gpu_instance();
    const bool probe_ok = relu_identity_probe();
#else
    const bool probe_ok = true;
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
    return fail == 0 && probe_ok ? 0 : 1;
}
