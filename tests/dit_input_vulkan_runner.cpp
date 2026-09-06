// 本文件是 SeedVR2DiTInput 的 Vulkan 数值对齐测试 runner。
//
// 目标：证明 Vulkan 路径与 CPU 基线逐元素一致，重点验证：
//   - patchify 的 2x2 空间重排（einops (dy,dx,c) 顺序）；
//   - 视频通道数 33（非 4 倍数）与 5 时的 cstep 对齐；
//   - sinusoidal 编码 + 时间 MLP（三层 InnerProduct + SiLU）；
//   - 复用原生 InnerProduct 的 Vulkan 实现（Layer_final 委托）。
//
// 做法：用小尺寸 dim=32/sinusoidal_dim=16/embedding_dim=192 生成单层 ncnn 图，
// 分别用 CPU 与 Vulkan 后端各 forward 一次，比较 4 个输出。
#include "seedvr2_dit_input.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "layer.h"
#include "mat.h"
#include "net.h"

namespace
{

const int DIM = 32;
const int SINUSOIDAL_DIM = 16;
const int EMBEDDING_DIM = DIM * 6; // 192

struct TestCase
{
    int video_channels;
    int text_channels;
    int T;
    int H;
    int W;
    int text_len;
    const char* name;
};

// 填充一个 [rows, cols] 的 2D Mat（或 1D）。
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
    const int vc = tc.video_channels;
    const int tc_ch = tc.text_channels;

    FILE* fp = fopen(param_path, "wb");
    if (!fp)
        return false;
    fprintf(fp, "7767517\n");
    fprintf(fp, "5 8\n");
    fprintf(fp, "Input input_vid 0 1 vid\n");
    fprintf(fp, "Input input_txt 0 1 txt\n");
    fprintf(fp, "Input input_timestep 0 1 timestep\n");
    fprintf(fp, "Input input_vid_shape 0 1 vid_shape\n");
    fprintf(fp, "SeedVR2DiTInput dit_input 4 4 vid txt timestep vid_shape vid_out txt_out emb patched_shape 0=%d 1=%d 2=%d 3=%d 4=%d\n",
            DIM, vc, tc_ch, SINUSOIDAL_DIM, EMBEDDING_DIM);
    fclose(fp);

    // bin：5 个 InnerProduct，每个按「weight 后 bias」顺序写入。
    // video_projection: vc*4 -> DIM
    // text_projection: tc -> DIM
    // time_projection_in: SINUSOIDAL_DIM -> DIM
    // time_projection_hidden: DIM -> DIM
    // time_projection_out: DIM -> EMBEDDING_DIM
    auto write_ip = [&](FILE* f, int in, int out, int seed) {
        std::vector<float> w(static_cast<size_t>(in) * out);
        std::vector<float> b(out);
        // 注意：必须显式转 int 再减法，(i + seed) % 7 是 size_t 无符号，减 3 为负时
        // 会无符号下溢成巨大值，污染 bin。
        for (size_t i = 0; i < w.size(); i++)
            w[i] = 0.005f * static_cast<float>(static_cast<int>((i + seed) % 7) - 3);
        for (size_t i = 0; i < b.size(); i++)
            b[i] = 0.01f * static_cast<float>(static_cast<int>((i + seed) % 5) - 2);
        fwrite(w.data(), sizeof(float), w.size(), f);
        fwrite(b.data(), sizeof(float), b.size(), f);
    };
    fp = fopen(bin_path, "wb");
    if (!fp)
        return false;
    write_ip(fp, vc * 4, DIM, 0);
    write_ip(fp, tc_ch, DIM, 10);
    write_ip(fp, SINUSOIDAL_DIM, DIM, 20);
    write_ip(fp, DIM, DIM, 30);
    write_ip(fp, DIM, EMBEDDING_DIM, 40);
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

int run(bool use_vulkan, const TestCase& tc, ncnn::Mat out[4])
{
    const char* param_path = "/tmp/dit_input_probe.param";
    const char* bin_path = "/tmp/dit_input_probe.bin";
    if (!write_param_bin(tc, param_path, bin_path))
        return -1;

    const int video_len = tc.T * tc.H * tc.W;

    ncnn::Mat video, text, timestep, shape;
    fill(video, tc.video_channels, video_len, 0);
    fill(text, tc.text_channels, tc.text_len, 100);
    timestep.create(1, static_cast<size_t>(4u));
    static_cast<float*>(timestep.data)[0] = 0.7f;
    shape.create(3, static_cast<size_t>(4u));
    int* s = shape;
    s[0] = tc.T;
    s[1] = tc.H;
    s[2] = tc.W;

    ncnn::Net net;
    net.register_custom_layer("SeedVR2DiTInput", SeedVR2DiTInput_layer_creator);
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
    const int r_in1 = ex.input("txt", text);
    const int r_in2 = ex.input("timestep", timestep);
    const int r_in3 = ex.input("vid_shape", shape);
    if (r_in0 != 0 || r_in1 != 0 || r_in2 != 0 || r_in3 != 0)
    {
        std::fprintf(stderr, "input failed (vulkan=%d) r=%d,%d,%d,%d\n",
                     use_vulkan ? 1 : 0, r_in0, r_in1, r_in2, r_in3);
        return -1;
    }
    const int r_out0 = ex.extract("vid_out", out[0]);
    const int r_out1 = ex.extract("txt_out", out[1]);
    const int r_out2 = ex.extract("emb", out[2]);
    const int r_out3 = ex.extract("patched_shape", out[3]);
    if (r_out0 != 0 || r_out1 != 0 || r_out2 != 0 || r_out3 != 0)
    {
        std::fprintf(stderr, "extract failed (vulkan=%d) r=%d,%d,%d,%d\n",
                     use_vulkan ? 1 : 0, r_out0, r_out1, r_out2, r_out3);
        return -1;
    }
    return 0;
}

} // namespace

int main()
{
    const TestCase cases[] = {
        {33, 16, 2, 4, 4, 4, "vc=33 T=2 H=4 W=4"},
        {33, 16, 1, 2, 6, 4, "vc=33 T=1 H=2 W=6(非4对齐)"},
        {5, 16, 2, 4, 4, 4, "vc=5(非4倍数,cstep) T=2 H=4 W=4"},
        {33, 8, 3, 2, 2, 6, "vc=33 tc=8 T=3 H=2 W=2"},
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

        ncnn::Mat out_cpu[4];
        if (run(false, tc, out_cpu) != 0)
        {
            std::printf("[case %d] %s : CPU 运行失败\n", i, tc.name);
            fail++;
            continue;
        }

#if NCNN_VULKAN
        ncnn::Mat out_gpu[4];
        if (run(true, tc, out_gpu) != 0)
        {
            std::printf("[case %d] %s : Vulkan 运行失败\n", i, tc.name);
            fail++;
            continue;
        }
        // 前 3 个输出是浮点，第 4 个 patched_shape 是 int 位模式。
        double diff = max_abs_diff(out_cpu[0], out_gpu[0]);
        diff = std::max(diff, max_abs_diff(out_cpu[1], out_gpu[1]));
        diff = std::max(diff, max_abs_diff(out_cpu[2], out_gpu[2]));
        const int* ps_cpu = out_cpu[3];
        const int* ps_gpu = out_gpu[3];
        const bool shape_ok = ps_cpu[0] == ps_gpu[0] && ps_cpu[1] == ps_gpu[1] && ps_cpu[2] == ps_gpu[2];
        const bool ok = diff < 1e-4 && shape_ok;
        std::printf("[case %d] %s : max|diff| = %.6f shape=%s %s\n",
                    i, tc.name, diff, shape_ok ? "ok" : "MISMATCH", ok ? "PASS" : "FAIL");
        ok ? pass++ : fail++;
#else
        std::printf("[case %d] %s : CPU vid=%dx%d txt=%dx%d emb=%d ok\n",
                    i, tc.name, out_cpu[0].h, out_cpu[0].w, out_cpu[1].h, out_cpu[1].w, out_cpu[2].w);
        pass++;
#endif
    }

    std::printf("======== 结果: %d pass, %d fail ========\n", pass, fail);
#if NCNN_VULKAN
    ncnn::destroy_gpu_instance();
#endif
    return fail == 0 ? 0 : 1;
}
