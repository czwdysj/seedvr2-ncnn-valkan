// 本文件是 Convolution3D（3D 卷积）Vulkan 后端的数值对齐测试 runner。
//
// 目标：证明新增的 Convolution3D_vulkan 与 CPU 基线逐元素一致，覆盖：
//   - kernel 3x3x3（VAE 时空卷积主体）、1x1x1（pointwise）；
//   - stride 1/2、dilation、显式 pad 与 SAME pad；
//   - 多通道、bias；
//   - 4D 输入 [w,h,d,c] 的 cstep 对齐。
//
// 做法：用小尺寸生成单层 ncnn 图，分别用 CPU 与 Vulkan 后端各 forward 一次，
// 比较输出。Convolution3D 为 ncnn 内置层，无需 register_custom_layer。
#include <cmath>
#include <cstdio>
#include <vector>

#include "layer.h"
#include "mat.h"
#include "net.h"

namespace
{

struct TestCase
{
    int w, h, d;          // 输入尺寸
    int channels;         // 输入通道
    int num_output;       // 输出通道
    int kernel_w, kernel_h, kernel_d;
    int stride_w, stride_h, stride_d;
    int dilation_w, dilation_h, dilation_d;
    int pad_left, pad_top, pad_front;   // -233 = SAME_UPPER
    bool bias;
    const char* name;
};

void fill4d(ncnn::Mat& m, int w, int h, int d, int c, int seed)
{
    m.create(w, h, d, c, static_cast<size_t>(4u));
    size_t idx = 0;
    for (int ch = 0; ch < c; ch++)
        for (int z = 0; z < d; z++)
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++, idx++)
                    m.channel(ch).depth(z).row(y)[x] =
                        static_cast<float>(static_cast<int>((idx + seed) % 11) - 5) * 0.1f;
}

bool write_param_bin(const TestCase& tc, const char* param_path, const char* bin_path)
{
    const int maxk = tc.kernel_w * tc.kernel_h * tc.kernel_d;
    const int weight_size = tc.num_output * tc.channels * maxk;

    FILE* fp = fopen(param_path, "wb");
    if (!fp)
        return false;
    fprintf(fp, "7767517\n");
    fprintf(fp, "2 2\n");
    fprintf(fp, "Input input 0 1 in\n");
    fprintf(fp, "Convolution3D conv3d 1 1 in out 0=%d 1=%d 11=%d 21=%d 2=%d 12=%d 22=%d 3=%d 13=%d 23=%d 4=%d 14=%d 24=%d 15=%d 16=%d 17=%d 5=%d 6=%d\n",
            tc.num_output, tc.kernel_w, tc.kernel_h, tc.kernel_d,
            tc.dilation_w, tc.dilation_h, tc.dilation_d,
            tc.stride_w, tc.stride_h, tc.stride_d,
            tc.pad_left, tc.pad_top, tc.pad_front,
            tc.pad_left, tc.pad_top, tc.pad_front,
            tc.bias ? 1 : 0, weight_size);
    fclose(fp);

    fp = fopen(bin_path, "wb");
    if (!fp)
        return false;
    // weight [num_output, channels, maxk]：type=0 权重前须写 4 字节 flag（fp32=0）。
    std::vector<float> w(static_cast<size_t>(weight_size));
    for (size_t i = 0; i < w.size(); i++)
        w[i] = 0.05f * static_cast<float>(static_cast<int>((i + 7) % 9) - 4);
    const uint32_t flag = 0u;
    fwrite(&flag, sizeof(flag), 1, fp);
    fwrite(w.data(), sizeof(float), w.size(), fp);
    if (tc.bias)
    {
        std::vector<float> b(tc.num_output);
        for (size_t i = 0; i < b.size(); i++)
            b[i] = 0.1f * static_cast<float>(static_cast<int>((i + 3) % 5) - 2);
        fwrite(b.data(), sizeof(float), b.size(), fp);
    }
    fclose(fp);
    return true;
}

double max_abs_diff(const ncnn::Mat& a, const ncnn::Mat& b)
{
    if (a.dims != b.dims || a.w != b.w || a.h != b.h || a.d != b.d || a.c != b.c)
        return 1e30;
    double worst = 0.0;
    // 4D Mat 的 buffer 是 cstep 对齐的（通道间有 padding），必须逐通道逐元素
    // 访问比较，不能用稠密 total() 遍历（会把 padding 间隙算进去）。
    for (int ch = 0; ch < a.c; ch++)
        for (int z = 0; z < a.d; z++)
            for (int y = 0; y < a.h; y++)
            {
                const float* ra = a.channel(ch).depth(z).row(y);
                const float* rb = b.channel(ch).depth(z).row(y);
                for (int x = 0; x < a.w; x++)
                {
                    const double dd = static_cast<double>(ra[x]) - static_cast<double>(rb[x]);
                    if (dd > worst) worst = dd;
                    else if (-dd > worst) worst = -dd;
                }
            }
    return worst;
}

int run(bool use_vulkan, const TestCase& tc, ncnn::Mat& out)
{
    const char* param_path = "/tmp/conv3d_probe.param";
    const char* bin_path = "/tmp/conv3d_probe.bin";
    if (!write_param_bin(tc, param_path, bin_path))
        return -1;

    ncnn::Mat input;
    fill4d(input, tc.w, tc.h, tc.d, tc.channels, 0);

    ncnn::Net net;
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
    ncnn::Extractor ex = net.create_extractor();
    if (ex.input("in", input) != 0)
        return -1;
    if (ex.extract("out", out) != 0)
    {
        std::fprintf(stderr, "extract failed (vulkan=%d)\n", use_vulkan ? 1 : 0);
        return -1;
    }
    return 0;
}

} // namespace

int main()
{
    const TestCase cases[] = {
        {4, 4, 4, 2, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, true, "3x3x3 s1 pad1 c2->3"},
        {3, 3, 3, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, false, "1x1x1 pointwise c4->2"},
        {5, 5, 5, 2, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, true, "3x3x3 s2 pad1 c2->4"},
        {3, 3, 3, 1, 1, 3, 3, 3, 1, 1, 1, 1, 1, 1, -233, -233, -233, true, "3x3x3 SAME c1->1"},
        {4, 3, 5, 3, 2, 3, 3, 3, 1, 1, 1, 2, 2, 2, 2, 1, 0, false, "3x3x3 dil2 c3->2 非对称"},
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
        double diff = max_abs_diff(out_cpu, out_gpu);
        const bool ok = diff < 1e-4;
        std::printf("[case %d] %s : max|diff| = %.6f %s\n", i, tc.name, diff, ok ? "PASS" : "FAIL");
        ok ? pass++ : fail++;
#else
        std::printf("[case %d] %s : CPU %dx%dx%dx%d ok\n", i, tc.name, out_cpu.w, out_cpu.h, out_cpu.d, out_cpu.c);
        pass++;
#endif
    }

    std::printf("======== 结果: %d pass, %d fail ========\n", pass, fail);
#if NCNN_VULKAN
    ncnn::destroy_gpu_instance();
#endif
    return fail == 0 ? 0 : 1;
}
