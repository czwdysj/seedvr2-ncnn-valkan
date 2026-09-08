// 本文件是 SeedVR2DiTOutput 的 Vulkan 数值对齐测试 runner。
//
// 目标：证明 Vulkan 路径与 CPU 基线逐元素一致，重点验证：
//   - 逐 token 平方和归约（norm_reduce，shared-memory 树形归约）；
//   - affine RMSNorm + output Ada 调制（norm_apply，pack1→pack4）；
//   - 2x2 unpatchify 空间反重排（projected pack4 → output pack1）；
//   - 复用原生 InnerProduct 的 Vulkan 实现（Layer_final 委托）。
//
// 做法：用小尺寸 dim=32/64、output_channels=4/5/8 生成单层 ncnn 图，分别用
// CPU 与 Vulkan 后端各 forward 一次，比较 2 个输出（浮点 output + int shape）。
#include "layers/seedvr2_dit_output.h"

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
    int dim;
    int output_channels;
    int T;
    int H;
    int W;
    const char* name;
};

// 填充一个 [rows, cols] 的 2D Mat（rows=1 时即 [cols,1]，w=cols）。
void fill(ncnn::Mat& m, int cols, int rows, int seed_offset)
{
    m.create(cols, rows, static_cast<size_t>(4u));
    size_t idx = 0;
    for (int r = 0; r < rows; r++)
    {
        // 注意：2D Mat 的行间存在 cstep 对齐填充，必须用 row() 获取行指针，
        // 不能用 m[r * cols + c] 直接跨行索引（非 4 倍数 cols 时会错位）。
        float* row = m.row(r);
        for (int c = 0; c < cols; c++)
            row[c] = static_cast<float>(static_cast<int>((idx++ + seed_offset) % 11) - 5) * 0.1f;
    }
}

bool write_param_bin(const TestCase& tc, const char* param_path, const char* bin_path)
{
    const int dim = tc.dim;
    const int oc = tc.output_channels;

    FILE* fp = fopen(param_path, "wb");
    if (!fp)
        return false;
    fprintf(fp, "7767517\n");
    fprintf(fp, "4 5\n");
    fprintf(fp, "Input input_vid 0 1 vid\n");
    fprintf(fp, "Input input_emb 0 1 emb\n");
    fprintf(fp, "Input input_vid_shape 0 1 vid_shape\n");
    fprintf(fp, "SeedVR2DiTOutput dit_output 3 2 vid emb vid_shape vid_out output_shape 0=%d 1=%d 2=1e-05\n",
            dim, oc);
    fclose(fp);

    // bin 顺序与 load_model 一致：norm_weight、output_shift、output_scale，
    // 然后是 projection 的 weight + bias（dim -> oc*4）。
    fp = fopen(bin_path, "wb");
    if (!fp)
        return false;

    auto write_vec = [&](FILE* f, int n, int seed) {
        std::vector<float> v(static_cast<size_t>(n));
        for (size_t i = 0; i < v.size(); i++)
            // 必须显式转 int 再减法，避免 size_t 无符号下溢。
            v[i] = 0.01f * static_cast<float>(static_cast<int>((i + seed) % 9) - 4);
        fwrite(v.data(), sizeof(float), v.size(), f);
    };
    auto write_ip = [&](FILE* f, int in, int out, int seed) {
        std::vector<float> w(static_cast<size_t>(in) * out);
        std::vector<float> b(out);
        for (size_t i = 0; i < w.size(); i++)
            w[i] = 0.005f * static_cast<float>(static_cast<int>((i + seed) % 7) - 3);
        for (size_t i = 0; i < b.size(); i++)
            b[i] = 0.01f * static_cast<float>(static_cast<int>((i + seed) % 5) - 2);
        fwrite(w.data(), sizeof(float), w.size(), f);
        fwrite(b.data(), sizeof(float), b.size(), f);
    };

    write_vec(fp, dim, 0);
    write_vec(fp, dim, 10);
    write_vec(fp, dim, 20);
    write_ip(fp, dim, oc * 4, 30);
    fclose(fp);
    return true;
}

double max_abs_diff(const ncnn::Mat& a, const ncnn::Mat& b)
{
    if (a.dims != b.dims || a.w != b.w || a.h != b.h)
        return 1e30;
    double worst = 0.0;
    const float* pa = a;
    const float* pb = b;
    const size_t total = a.total();
    for (size_t i = 0; i < total; i++)
    {
        const double d = static_cast<double>(pa[i]) - static_cast<double>(pb[i]);
        if (d > worst)
            worst = d;
        else if (-d > worst)
            worst = -d;
    }
    return worst;
}

int run(bool use_vulkan, const TestCase& tc, ncnn::Mat out[2])
{
    const char* param_path = "/tmp/dit_output_probe.param";
    const char* bin_path = "/tmp/dit_output_probe.bin";
    if (!write_param_bin(tc, param_path, bin_path))
        return -1;

    const int dim = tc.dim;
    const int tokens = tc.T * tc.H * tc.W;

    ncnn::Mat video, embedding, shape;
    fill(video, dim, tokens, 0);
    fill(embedding, dim * 6, 1, 100);
    shape.create(3, static_cast<size_t>(4u));
    int* s = shape;
    s[0] = tc.T;
    s[1] = tc.H;
    s[2] = tc.W;

    ncnn::Net net;
    net.register_custom_layer("SeedVR2DiTOutput", SeedVR2DiTOutput_layer_creator);
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
    const int r_in0 = ex.input("vid", video);
    const int r_in1 = ex.input("emb", embedding);
    const int r_in2 = ex.input("vid_shape", shape);
    if (r_in0 != 0 || r_in1 != 0 || r_in2 != 0)
    {
        std::fprintf(stderr, "input failed (vulkan=%d) r=%d,%d,%d\n",
                     use_vulkan ? 1 : 0, r_in0, r_in1, r_in2);
        return -1;
    }
    const int r_out0 = ex.extract("vid_out", out[0]);
    const int r_out1 = ex.extract("output_shape", out[1]);
    if (r_out0 != 0 || r_out1 != 0)
    {
        std::fprintf(stderr, "extract failed (vulkan=%d) r=%d,%d\n",
                     use_vulkan ? 1 : 0, r_out0, r_out1);
        return -1;
    }
    return 0;
}

} // namespace

int main()
{
    const TestCase cases[] = {
        {32, 4, 1, 2, 2, "dim=32 oc=4 T=1 H=2 W=2"},
        {64, 8, 2, 2, 3, "dim=64 oc=8 T=2 H=2 W=3(非对齐)"},
        {32, 5, 1, 1, 1, "dim=32 oc=5(非4倍数) T=1 单patch"},
        {64, 8, 2, 3, 3, "dim=64 oc=8 T=2 H=3 W=3(非方阵)"},
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

        ncnn::Mat out_cpu[2];
        if (run(false, tc, out_cpu) != 0)
        {
            std::printf("[case %d] %s : CPU 运行失败\n", i, tc.name);
            fail++;
            continue;
        }

#if NCNN_VULKAN
        ncnn::Mat out_gpu[2];
        if (run(true, tc, out_gpu) != 0)
        {
            std::printf("[case %d] %s : Vulkan 运行失败\n", i, tc.name);
            fail++;
            continue;
        }
        double diff = max_abs_diff(out_cpu[0], out_gpu[0]);
        const int* os_cpu = out_cpu[1];
        const int* os_gpu = out_gpu[1];
        const bool shape_ok = os_cpu[0] == os_gpu[0] && os_cpu[1] == os_gpu[1] && os_cpu[2] == os_gpu[2];
        const bool ok = diff < 1e-4 && shape_ok;
        std::printf("[case %d] %s : max|diff| = %.6f shape=%s %s\n",
                    i, tc.name, diff, shape_ok ? "ok" : "MISMATCH", ok ? "PASS" : "FAIL");
        ok ? pass++ : fail++;
#else
        std::printf("[case %d] %s : CPU out=%dx%d ok\n",
                    i, tc.name, out_cpu[0].h, out_cpu[0].w);
        pass++;
#endif
    }

    std::printf("======== 结果: %d pass, %d fail ========\n", pass, fail);
#if NCNN_VULKAN
    ncnn::destroy_gpu_instance();
#endif
    return fail == 0 ? 0 : 1;
}
