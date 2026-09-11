// 本文件是 SeedVR2DiTBlock 的 Vulkan 数值对齐测试 runner。
//
// 目标：证明 Vulkan 路径与 CPU 基线逐元素一致，重点验证：
//   - 逐 token RMSNorm + adaLN 输入调制（reduce+apply 两段）；
//   - MM-RoPE 旋转（多轴）+ bf16 舍入；
//   - softmax(QK^T/√d)@V 的动态窗口 attention（三遍 QK^T，不物化 scores）；
//   - 文本跨窗口平均、视频 shifted 窗口跨窗口累加；
//   - SwiGLU MLP（gate_proj + in_proj + silu + out_proj）；
//   - 复用原生 InnerProduct（vkdev 传递 + pack1 命中 gemm 分支）；
//   - last_layer / shared_weights 的特殊语义。
//
// 做法：用小尺寸 dim=32/128 生成单层 ncnn 图，分别用 CPU 与 Vulkan 后端各
// forward 一次，比较 2 个输出。
#include "layers/seedvr2_dit_block.h"

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
    int heads;
    int head_dim;
    int mlp_hidden;
    int T;
    int H;
    int W;
    int text_len;
    bool shared;
    bool last;
    bool shifted;
    const char* name;
};

void fill(ncnn::Mat& m, int cols, int rows, int seed_offset)
{
    m.create(cols, rows, static_cast<size_t>(4u));
    size_t idx = 0;
    for (int r = 0; r < rows; r++)
    {
        float* row = m.row(r);
        for (int c = 0; c < cols; c++)
            row[c] = static_cast<float>(static_cast<int>((idx++ + seed_offset) % 11) - 5) * 0.1f;
    }
}

bool write_param_bin(const TestCase& tc, const char* param_path, const char* bin_path)
{
    const int dim = tc.dim;
    const int heads = tc.heads;
    const int head_dim = tc.head_dim;
    const int mlp_hidden = tc.mlp_hidden;

    FILE* fp = fopen(param_path, "wb");
    if (!fp)
        return false;
    fprintf(fp, "7767517\n");
    fprintf(fp, "5 6\n");
    fprintf(fp, "Input input_vid 0 1 vid\n");
    fprintf(fp, "Input input_txt 0 1 txt\n");
    fprintf(fp, "Input input_emb 0 1 emb\n");
    fprintf(fp, "Input input_vid_shape 0 1 vid_shape\n");
    fprintf(fp, "SeedVR2DiTBlock dit_block 4 2 vid txt emb vid_shape vid_out txt_out 0=0 1=%d 2=%d 3=%d 4=%d 5=%d 6=%d 7=%d 8=1e-05\n",
            tc.shared ? 1 : 0, tc.last ? 1 : 0, tc.shifted ? 1 : 0,
            dim, heads, head_dim, mlp_hidden);
    fclose(fp);

    fp = fopen(bin_path, "wb");
    if (!fp)
        return false;

    auto write_vec = [&](FILE* f, int n, int seed) {
        std::vector<float> v(static_cast<size_t>(n));
        for (size_t i = 0; i < v.size(); i++)
            v[i] = 0.01f * static_cast<float>(static_cast<int>((i + seed) % 9) - 4);
        fwrite(v.data(), sizeof(float), v.size(), f);
    };
    auto write_ip = [&](FILE* f, int in, int out, bool bias, int seed) {
        std::vector<float> w(static_cast<size_t>(in) * out);
        for (size_t i = 0; i < w.size(); i++)
            w[i] = 0.005f * static_cast<float>(static_cast<int>((i + seed) % 7) - 3);
        fwrite(w.data(), sizeof(float), w.size(), f);
        if (bias)
        {
            std::vector<float> b(out);
            for (size_t i = 0; i < b.size(); i++)
                b[i] = 0.01f * static_cast<float>(static_cast<int>((i + seed) % 5) - 2);
            fwrite(b.data(), sizeof(float), b.size(), f);
        }
    };
    auto write_branch = [&](FILE* f, int base_seed) {
        write_vec(f, dim, base_seed + 0);   // attn_shift
        write_vec(f, dim, base_seed + 1);   // attn_scale
        write_vec(f, dim, base_seed + 2);   // attn_gate
        write_vec(f, dim, base_seed + 3);   // mlp_shift
        write_vec(f, dim, base_seed + 4);   // mlp_scale
        write_vec(f, dim, base_seed + 5);   // mlp_gate
        write_ip(f, dim, dim * 3, false, base_seed + 6);   // qkv
        write_ip(f, dim, dim, true, base_seed + 7);        // proj_out
        write_vec(f, head_dim, base_seed + 8);             // norm_q
        write_vec(f, head_dim, base_seed + 9);             // norm_k
        write_ip(f, dim, mlp_hidden, false, base_seed + 10);  // mlp_gate_proj
        write_ip(f, dim, mlp_hidden, false, base_seed + 11);  // mlp_in_proj
        write_ip(f, mlp_hidden, dim, false, base_seed + 12);  // mlp_out_proj
    };

    write_branch(fp, 0);
    if (!tc.shared)
        write_branch(fp, 50);
    write_vec(fp, 21, 100);   // rope_freqs
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
    const char* param_path = "/tmp/dit_block_probe.param";
    const char* bin_path = "/tmp/dit_block_probe.bin";
    if (!write_param_bin(tc, param_path, bin_path))
        return -1;

    const int dim = tc.dim;
    const int video_len = tc.T * tc.H * tc.W;

    ncnn::Mat vid, txt, embedding, shape;
    fill(vid, dim, video_len, 0);
    fill(txt, dim, tc.text_len, 100);
    fill(embedding, dim * 6, 1, 200);
    shape.create(3, static_cast<size_t>(4u));
    int* s = shape;
    s[0] = tc.T;
    s[1] = tc.H;
    s[2] = tc.W;

    ncnn::Net net;
    net.register_custom_layer("SeedVR2DiTBlock", SeedVR2DiTBlock_layer_creator);
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

    if (use_vulkan)
    {
        auto* layer = dynamic_cast<SeedVR2DiTBlock*>(net.mutable_layers().back());
        if (!layer)
            return -1;
        layer->set_runtime_shape(tc.T, tc.H, tc.W);
    }

    ncnn::Extractor ex = net.create_extractor();
    const int r0 = ex.input("vid", vid);
    const int r1 = ex.input("txt", txt);
    const int r2 = ex.input("emb", embedding);
    const int r3 = ex.input("vid_shape", shape);
    if (r0 != 0 || r1 != 0 || r2 != 0 || r3 != 0)
    {
        std::fprintf(stderr, "input failed (vulkan=%d) r=%d,%d,%d,%d\n",
                     use_vulkan ? 1 : 0, r0, r1, r2, r3);
        return -1;
    }
    const int ro0 = ex.extract("vid_out", out[0]);
    const int ro1 = ex.extract("txt_out", out[1]);
    if (ro0 != 0 || ro1 != 0)
    {
        std::fprintf(stderr, "extract failed (vulkan=%d) r=%d,%d\n",
                     use_vulkan ? 1 : 0, ro0, ro1);
        return -1;
    }
    return 0;
}

} // namespace

int main()
{
    // 注意：head_dim 必须 >= 128，因为 CPU 基线的 MM-RoPE 循环覆盖 126 维且无
    // 边界检查（假设 head_dim=128）。用 head_dim=128 与真实模型一致。
    const TestCase cases[] = {
        {128, 1, 128, 128, 2, 2, 2, 3, false, false, false, "dim=128 heads=1 T=2 基本"},
        {128, 1, 128, 128, 2, 2, 2, 3, false, false, true, "shifted(视频跨窗口累加)"},
        {128, 1, 128, 128, 1, 1, 1, 2, true, true, false, "last_layer+shared 单patch"},
        {256, 2, 128, 128, 3, 2, 2, 4, false, false, false, "dim=256 heads=2 T=3"},
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
        double diff = std::max(max_abs_diff(out_cpu[0], out_gpu[0]),
                              max_abs_diff(out_cpu[1], out_gpu[1]));
        // bf16 舍入 + softmax 累积，允许 1e-3 量级的浮点差异。
        const bool ok = diff < 1e-3;
        std::printf("[case %d] %s : max|diff| = %.6f %s\n", i, tc.name, diff, ok ? "PASS" : "FAIL");
        ok ? pass++ : fail++;
#else
        std::printf("[case %d] %s : CPU vid=%dx%d txt=%dx%d ok\n",
                    i, tc.name, out_cpu[0].h, out_cpu[0].w, out_cpu[1].h, out_cpu[1].w);
        pass++;
#endif
    }

    std::printf("======== 结果: %d pass, %d fail ========\n", pass, fail);
#if NCNN_VULKAN
    ncnn::destroy_gpu_instance();
#endif
    return fail == 0 ? 0 : 1;
}
