// perf_bench.cpp —— 各 Vulkan 算子的真实规模性能基准（CPU vs Vulkan 延迟对比）。
//
// 目标：在真实规模（接近实际模型）下，测量每个算子在 CPU 与 Vulkan 后端的
// 单次 forward 延迟，输出加速比。与数值对齐 runner 的区别：本程序用「真实量级
// 尺寸 + 多次迭代计时」，而不是「小尺寸 + 一次性数值比对」。
//
// 计时方法：先 warmup 一次（触发 shader 编译 / 缓存预热），再对 CPU 与 Vulkan
// 各测 N 次取中位数。ncnn 的 Extractor::extract 在 Vulkan 模式下内部会
// submit_and_wait，返回时 GPU 计算已完成，故计时是「端到端延迟」。
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "layer.h"
#include "mat.h"
#include "net.h"

#include "dynamic_framewise_group_norm.h"
#include "dynamic_framewise_spatial_attention.h"
#include "dynamic_space_time_shuffle.h"
#include "seedvr2_dit_block.h"
#include "seedvr2_dit_input.h"
#include "seedvr2_dit_output.h"

namespace
{

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

double median(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

void set_opts(ncnn::Net& net, bool use_vulkan)
{
    net.opt.use_vulkan_compute = use_vulkan;
    net.opt.use_packing_layout = false;
    net.opt.use_fp16_packed = false;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;
    net.opt.num_threads = 8;   // CPU 用 8 线程（合理多核基线）
}

struct BenchResult
{
    const char* name;
    double cpu_ms;
    double gpu_ms;
    bool ok;
};

void report(const BenchResult& r)
{
    const double speedup = (r.gpu_ms > 0.0) ? r.cpu_ms / r.gpu_ms : 0.0;
    std::printf("%-14s  CPU %10.3f ms   Vulkan %10.3f ms   speedup %7.2fx  %s\n",
                r.name, r.cpu_ms, r.gpu_ms, speedup, r.ok ? "" : "(数值不匹配!)");
}

// ============================================================================
// 通用输入填充
// ============================================================================
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

void fill2d(ncnn::Mat& m, int cols, int rows, int seed)
{
    m.create(cols, rows, static_cast<size_t>(4u));
    size_t idx = 0;
    for (int r = 0; r < rows; r++)
    {
        float* row = m.row(r);
        for (int c = 0; c < cols; c++)
            row[c] = static_cast<float>(static_cast<int>((idx++ + seed) % 11) - 5) * 0.1f;
    }
}

double max_abs_diff(const ncnn::Mat& a, const ncnn::Mat& b)
{
    if (a.dims != b.dims || a.w != b.w || a.h != b.h || a.d != b.d || a.c != b.c)
        return 1e30;
    double worst = 0.0;
    for (int ch = 0; ch < a.c; ch++)
    {
        const float* pa = a.channel(ch);
        const float* pb = b.channel(ch);
        const size_t plane = static_cast<size_t>(a.w) * a.h * a.d;
        for (size_t i = 0; i < plane; i++)
        {
            const double dd = static_cast<double>(pa[i]) - static_cast<double>(pb[i]);
            if (dd > worst) worst = dd;
            else if (-dd > worst) worst = -dd;
        }
    }
    return worst;
}

// ============================================================================
// 1. DynamicSpaceTimeShuffle
// ============================================================================
BenchResult bench_shuffle(int C, int T, int H, int W, int R_T)
{
    const int projected = C * 4 * R_T;
    FILE* fp = fopen("/tmp/perf_shuffle.param", "wb");
    fprintf(fp, "7767517\n2 2\nInput in0 0 1 in0\n");
    fprintf(fp, "DynamicSpaceTimeShuffle shuffle 1 1 in0 out0 -23310=1,%d -23311=5,%d,%d,1,1,1\n",
            projected, projected, C);
    fclose(fp);
    std::vector<float> bias(projected, 0.01f);
    std::vector<float> weight(static_cast<size_t>(projected) * C, 0.005f);
    fp = fopen("/tmp/perf_shuffle.bin", "wb");
    fwrite(bias.data(), sizeof(float), bias.size(), fp);
    fwrite(weight.data(), sizeof(float), weight.size(), fp);
    fclose(fp);

    ncnn::Net net_cpu, net_gpu;
    set_opts(net_cpu, false);
    set_opts(net_gpu, true);
    net_cpu.register_custom_layer("DynamicSpaceTimeShuffle", DynamicSpaceTimeShuffle_layer_creator);
    net_gpu.register_custom_layer("DynamicSpaceTimeShuffle", DynamicSpaceTimeShuffle_layer_creator);
    if (net_cpu.load_param("/tmp/perf_shuffle.param") || net_cpu.load_model("/tmp/perf_shuffle.bin")
        || net_gpu.load_param("/tmp/perf_shuffle.param") || net_gpu.load_model("/tmp/perf_shuffle.bin"))
        return {"shuffle", 0, 0, false};

    ncnn::Mat in;
    fill4d(in, W, H, T, C, 0);

    { ncnn::Mat o; ncnn::Extractor e = net_cpu.create_extractor(); e.input("in0", in); e.extract("out0", o); }
    { ncnn::Mat o; ncnn::Extractor e = net_gpu.create_extractor(); e.input("in0", in); e.extract("out0", o); }

    std::vector<double> cpu_t, gpu_t;
    ncnn::Mat ref;
    for (int i = 0; i < 5; i++)
    {
        auto t0 = Clock::now();
        ncnn::Mat o; ncnn::Extractor e = net_cpu.create_extractor(); e.input("in0", in); e.extract("out0", o);
        cpu_t.push_back(ms_since(t0));
        if (i == 0) ref = o;
    }
    double worst = 0.0;
    for (int i = 0; i < 5; i++)
    {
        auto t0 = Clock::now();
        ncnn::Mat o; ncnn::Extractor e = net_gpu.create_extractor(); e.input("in0", in); e.extract("out0", o);
        gpu_t.push_back(ms_since(t0));
        worst = std::max(worst, max_abs_diff(ref, o));
    }
    return {"shuffle", median(cpu_t), median(gpu_t), worst < 1e-3};
}

// ============================================================================
// 2. DynamicFramewiseGroupNorm
// ============================================================================
BenchResult bench_groupnorm(int C, int T, int H, int W)
{
    FILE* fp = fopen("/tmp/perf_gn.param", "wb");
    fprintf(fp, "7767517\n2 2\nInput in0 0 1 in0\n");
    fprintf(fp, "DynamicFramewiseGroupNorm norm 1 1 in0 out0 -23310=1,%d\n", C);
    fclose(fp);
    std::vector<float> b(C, 0.02f), w(C, 0.5f);
    fp = fopen("/tmp/perf_gn.bin", "wb");
    fwrite(b.data(), sizeof(float), b.size(), fp);
    fwrite(w.data(), sizeof(float), w.size(), fp);
    fclose(fp);

    ncnn::Net net_cpu, net_gpu;
    set_opts(net_cpu, false);
    set_opts(net_gpu, true);
    net_cpu.register_custom_layer("DynamicFramewiseGroupNorm", DynamicFramewiseGroupNorm_layer_creator);
    net_gpu.register_custom_layer("DynamicFramewiseGroupNorm", DynamicFramewiseGroupNorm_layer_creator);
    if (net_cpu.load_param("/tmp/perf_gn.param") || net_cpu.load_model("/tmp/perf_gn.bin")
        || net_gpu.load_param("/tmp/perf_gn.param") || net_gpu.load_model("/tmp/perf_gn.bin"))
        return {"groupnorm", 0, 0, false};

    ncnn::Mat in;
    fill4d(in, W, H, T, C, 0);
    { ncnn::Mat o; ncnn::Extractor e = net_cpu.create_extractor(); e.input("in0", in); e.extract("out0", o); }
    { ncnn::Mat o; ncnn::Extractor e = net_gpu.create_extractor(); e.input("in0", in); e.extract("out0", o); }

    std::vector<double> cpu_t, gpu_t;
    ncnn::Mat ref;
    for (int i = 0; i < 5; i++)
    {
        auto t0 = Clock::now();
        ncnn::Mat o; ncnn::Extractor e = net_cpu.create_extractor(); e.input("in0", in); e.extract("out0", o);
        cpu_t.push_back(ms_since(t0));
        if (i == 0) ref = o;
    }
    double worst = 0.0;
    for (int i = 0; i < 5; i++)
    {
        auto t0 = Clock::now();
        ncnn::Mat o; ncnn::Extractor e = net_gpu.create_extractor(); e.input("in0", in); e.extract("out0", o);
        gpu_t.push_back(ms_since(t0));
        worst = std::max(worst, max_abs_diff(ref, o));
    }
    return {"groupnorm", median(cpu_t), median(gpu_t), worst < 1e-3};
}

// ============================================================================
// 3. DynamicFramewiseSpatialAttention
// ============================================================================
BenchResult bench_attention(int C, int T, int H, int W)
{
    const int matrix = C * C;
    FILE* fp = fopen("/tmp/perf_attn.param", "wb");
    fprintf(fp, "7767517\n2 2\nInput in0 0 1 in0\n");
    fprintf(fp, "DynamicFramewiseSpatialAttention attn 1 1 in0 out0 -23310=1,%d\n", C);
    fclose(fp);
    auto fill = [](std::vector<float>& v, float base, float step) {
        for (size_t i = 0; i < v.size(); i++)
            v[i] = base + step * static_cast<float>(static_cast<int>((i % 7) - 3));
    };
    std::vector<float> nb(C), nw(C), qb(C), qw(matrix), kb(C), kw(matrix), vb(C), vw(matrix), ob(C), ow(matrix);
    fill(nb, 0.f, 0.02f); fill(nw, 1.f, 0.02f);
    fill(qb, 0.f, 0.01f); fill(qw, 0.f, 0.005f);
    fill(kb, 0.f, 0.01f); fill(kw, 0.f, 0.005f);
    fill(vb, 0.f, 0.01f); fill(vw, 0.f, 0.005f);
    fill(ob, 0.f, 0.01f); fill(ow, 0.f, 0.005f);
    fp = fopen("/tmp/perf_attn.bin", "wb");
    const std::vector<float>* order[] = {&nb, &nw, &kb, &kw, &ob, &ow, &qb, &qw, &vb, &vw};
    for (auto v : order) fwrite(v->data(), sizeof(float), v->size(), fp);
    fclose(fp);

    ncnn::Net net_cpu, net_gpu;
    set_opts(net_cpu, false);
    set_opts(net_gpu, true);
    net_cpu.register_custom_layer("DynamicFramewiseSpatialAttention", DynamicFramewiseSpatialAttention_layer_creator);
    net_gpu.register_custom_layer("DynamicFramewiseSpatialAttention", DynamicFramewiseSpatialAttention_layer_creator);
    if (net_cpu.load_param("/tmp/perf_attn.param") || net_cpu.load_model("/tmp/perf_attn.bin")
        || net_gpu.load_param("/tmp/perf_attn.param") || net_gpu.load_model("/tmp/perf_attn.bin"))
        return {"attention", 0, 0, false};

    ncnn::Mat in;
    fill4d(in, W, H, T, C, 0);
    { ncnn::Mat o; ncnn::Extractor e = net_cpu.create_extractor(); e.input("in0", in); e.extract("out0", o); }
    { ncnn::Mat o; ncnn::Extractor e = net_gpu.create_extractor(); e.input("in0", in); e.extract("out0", o); }

    std::vector<double> cpu_t, gpu_t;
    ncnn::Mat ref;
    for (int i = 0; i < 3; i++)
    {
        auto t0 = Clock::now();
        ncnn::Mat o; ncnn::Extractor e = net_cpu.create_extractor(); e.input("in0", in); e.extract("out0", o);
        cpu_t.push_back(ms_since(t0));
        if (i == 0) ref = o;
    }
    double worst = 0.0;
    for (int i = 0; i < 3; i++)
    {
        auto t0 = Clock::now();
        ncnn::Mat o; ncnn::Extractor e = net_gpu.create_extractor(); e.input("in0", in); e.extract("out0", o);
        gpu_t.push_back(ms_since(t0));
        worst = std::max(worst, max_abs_diff(ref, o));
    }
    return {"attention", median(cpu_t), median(gpu_t), worst < 1e-3};
}

// ============================================================================
// 4. Convolution3D（ncnn 内置层，Vulkan 后端为本项目新增）
// ============================================================================
BenchResult bench_conv3d(int w, int h, int d, int c, int out)
{
    const int weight_size = out * c * 27;
    FILE* fp = fopen("/tmp/perf_conv3d.param", "wb");
    fprintf(fp, "7767517\n2 2\nInput input 0 1 in\n");
    fprintf(fp, "Convolution3D conv3d 1 1 in out 0=%d 1=3 11=3 21=3 2=1 12=1 22=1 3=1 13=1 23=1 4=1 14=1 24=1 15=1 16=1 17=1 5=1 6=%d\n",
            out, weight_size);
    fclose(fp);
    std::vector<float> wt(weight_size, 0.05f);
    std::vector<float> bs(out, 0.1f);
    fp = fopen("/tmp/perf_conv3d.bin", "wb");
    uint32_t flag = 0u;
    fwrite(&flag, sizeof(flag), 1, fp);
    fwrite(wt.data(), sizeof(float), wt.size(), fp);
    fwrite(bs.data(), sizeof(float), bs.size(), fp);
    fclose(fp);

    ncnn::Net net_cpu, net_gpu;
    set_opts(net_cpu, false);
    set_opts(net_gpu, true);
    if (net_cpu.load_param("/tmp/perf_conv3d.param") || net_cpu.load_model("/tmp/perf_conv3d.bin")
        || net_gpu.load_param("/tmp/perf_conv3d.param") || net_gpu.load_model("/tmp/perf_conv3d.bin"))
        return {"conv3d", 0, 0, false};

    ncnn::Mat in;
    fill4d(in, w, h, d, c, 0);
    { ncnn::Mat o; ncnn::Extractor e = net_cpu.create_extractor(); e.input("in", in); e.extract("out", o); }
    { ncnn::Mat o; ncnn::Extractor e = net_gpu.create_extractor(); e.input("in", in); e.extract("out", o); }

    std::vector<double> cpu_t, gpu_t;
    ncnn::Mat ref;
    for (int i = 0; i < 3; i++)
    {
        auto t0 = Clock::now();
        ncnn::Mat o; ncnn::Extractor e = net_cpu.create_extractor(); e.input("in", in); e.extract("out", o);
        cpu_t.push_back(ms_since(t0));
        if (i == 0) ref = o;
    }
    double worst = 0.0;
    for (int i = 0; i < 3; i++)
    {
        auto t0 = Clock::now();
        ncnn::Mat o; ncnn::Extractor e = net_gpu.create_extractor(); e.input("in", in); e.extract("out", o);
        gpu_t.push_back(ms_since(t0));
        worst = std::max(worst, max_abs_diff(ref, o));
    }
    return {"conv3d", median(cpu_t), median(gpu_t), worst < 1e-2};
}

// ============================================================================
// 5. SeedVR2DiTInput（复用 5 个 InnerProduct）
// ============================================================================
BenchResult bench_dit_input(int dim, int vc, int tc, int T, int H, int W)
{
    const int sinusoidal = 512;
    const int embed = dim * 6;
    FILE* fp = fopen("/tmp/perf_ditin.param", "wb");
    fprintf(fp, "7767517\n5 8\n");
    fprintf(fp, "Input input_vid 0 1 vid\nInput input_txt 0 1 txt\nInput input_timestep 0 1 timestep\nInput input_vid_shape 0 1 vid_shape\n");
    fprintf(fp, "SeedVR2DiTInput dit_input 4 4 vid txt timestep vid_shape vid_out txt_out emb patched_shape 0=%d 1=%d 2=%d 3=%d 4=%d\n",
            dim, vc, tc, sinusoidal, embed);
    fclose(fp);
    fp = fopen("/tmp/perf_ditin.bin", "wb");
    auto write_ip = [&](FILE* f, int in, int out, int seed) {
        std::vector<float> w(static_cast<size_t>(in) * out), b(out);
        for (size_t i = 0; i < w.size(); i++) w[i] = 0.005f * static_cast<float>(static_cast<int>((i + seed) % 7) - 3);
        for (size_t i = 0; i < b.size(); i++) b[i] = 0.01f * static_cast<float>(static_cast<int>((i + seed) % 5) - 2);
        fwrite(w.data(), sizeof(float), w.size(), f);
        fwrite(b.data(), sizeof(float), b.size(), f);
    };
    write_ip(fp, vc * 4, dim, 0);
    write_ip(fp, tc, dim, 10);
    write_ip(fp, sinusoidal, dim, 20);
    write_ip(fp, dim, dim, 30);
    write_ip(fp, dim, embed, 40);
    fclose(fp);

    const int video_len = T * H * W;
    ncnn::Mat video, text, timestep, shape;
    fill2d(video, vc, video_len, 0);
    fill2d(text, tc, 1024, 100);
    timestep.create(1, static_cast<size_t>(4u));
    static_cast<float*>(timestep.data)[0] = 0.7f;
    shape.create(3, static_cast<size_t>(4u));
    int* s = shape; s[0] = T; s[1] = H; s[2] = W;

    ncnn::Net net_cpu, net_gpu;
    set_opts(net_cpu, false);
    set_opts(net_gpu, true);
    net_cpu.register_custom_layer("SeedVR2DiTInput", SeedVR2DiTInput_layer_creator);
    net_gpu.register_custom_layer("SeedVR2DiTInput", SeedVR2DiTInput_layer_creator);
    if (net_cpu.load_param("/tmp/perf_ditin.param") || net_cpu.load_model("/tmp/perf_ditin.bin")
        || net_gpu.load_param("/tmp/perf_ditin.param") || net_gpu.load_model("/tmp/perf_ditin.bin"))
        return {"dit_input", 0, 0, false};

    { ncnn::Mat o[4]; ncnn::Extractor e = net_cpu.create_extractor(); e.input("vid", video); e.input("txt", text); e.input("timestep", timestep); e.input("vid_shape", shape); e.extract("vid_out", o[0]); e.extract("txt_out", o[1]); e.extract("emb", o[2]); e.extract("patched_shape", o[3]); }
    { ncnn::Mat o[4]; ncnn::Extractor e = net_gpu.create_extractor(); e.input("vid", video); e.input("txt", text); e.input("timestep", timestep); e.input("vid_shape", shape); e.extract("vid_out", o[0]); e.extract("txt_out", o[1]); e.extract("emb", o[2]); e.extract("patched_shape", o[3]); }

    std::vector<double> cpu_t, gpu_t;
    ncnn::Mat ref;
    for (int i = 0; i < 5; i++)
    {
        auto t0 = Clock::now();
        ncnn::Mat o[4]; ncnn::Extractor e = net_cpu.create_extractor(); e.input("vid", video); e.input("txt", text); e.input("timestep", timestep); e.input("vid_shape", shape); e.extract("vid_out", o[0]); e.extract("txt_out", o[1]); e.extract("emb", o[2]); e.extract("patched_shape", o[3]);
        cpu_t.push_back(ms_since(t0));
        if (i == 0) ref = o[0];
    }
    double worst = 0.0;
    for (int i = 0; i < 5; i++)
    {
        auto t0 = Clock::now();
        ncnn::Mat o[4]; ncnn::Extractor e = net_gpu.create_extractor(); e.input("vid", video); e.input("txt", text); e.input("timestep", timestep); e.input("vid_shape", shape); e.extract("vid_out", o[0]); e.extract("txt_out", o[1]); e.extract("emb", o[2]); e.extract("patched_shape", o[3]);
        gpu_t.push_back(ms_since(t0));
        worst = std::max(worst, max_abs_diff(ref, o[0]));
    }
    return {"dit_input", median(cpu_t), median(gpu_t), worst < 1e-2};
}

// ============================================================================
// 6. SeedVR2DiTOutput（复用 1 个 InnerProduct）
// ============================================================================
BenchResult bench_dit_output(int dim, int oc, int T, int H, int W)
{
    FILE* fp = fopen("/tmp/perf_ditout.param", "wb");
    fprintf(fp, "7767517\n4 5\n");
    fprintf(fp, "Input input_vid 0 1 vid\nInput input_emb 0 1 emb\nInput input_vid_shape 0 1 vid_shape\n");
    fprintf(fp, "SeedVR2DiTOutput dit_output 3 2 vid emb vid_shape vid_out output_shape 0=%d 1=%d 2=1e-05\n", dim, oc);
    fclose(fp);
    fp = fopen("/tmp/perf_ditout.bin", "wb");
    auto write_vec = [&](FILE* f, int n, int seed) {
        std::vector<float> v(n);
        for (size_t i = 0; i < v.size(); i++) v[i] = 0.01f * static_cast<float>(static_cast<int>((i + seed) % 9) - 4);
        fwrite(v.data(), sizeof(float), v.size(), f);
    };
    auto write_ip = [&](FILE* f, int in, int out, int seed) {
        std::vector<float> w(static_cast<size_t>(in) * out), b(out);
        for (size_t i = 0; i < w.size(); i++) w[i] = 0.005f * static_cast<float>(static_cast<int>((i + seed) % 7) - 3);
        for (size_t i = 0; i < b.size(); i++) b[i] = 0.01f * static_cast<float>(static_cast<int>((i + seed) % 5) - 2);
        fwrite(w.data(), sizeof(float), w.size(), f);
        fwrite(b.data(), sizeof(float), b.size(), f);
    };
    write_vec(fp, dim, 0);   // norm_weight
    write_vec(fp, dim, 10);  // output_shift
    write_vec(fp, dim, 20);  // output_scale
    write_ip(fp, dim, oc * 4, 30);  // projection
    fclose(fp);

    const int tokens = T * H * W;
    ncnn::Mat video, embedding, shape;
    fill2d(video, dim, tokens, 0);
    fill2d(embedding, dim * 6, 1, 100);
    shape.create(3, static_cast<size_t>(4u));
    int* s = shape; s[0] = T; s[1] = H; s[2] = W;

    ncnn::Net net_cpu, net_gpu;
    set_opts(net_cpu, false);
    set_opts(net_gpu, true);
    net_cpu.register_custom_layer("SeedVR2DiTOutput", SeedVR2DiTOutput_layer_creator);
    net_gpu.register_custom_layer("SeedVR2DiTOutput", SeedVR2DiTOutput_layer_creator);
    if (net_cpu.load_param("/tmp/perf_ditout.param") || net_cpu.load_model("/tmp/perf_ditout.bin")
        || net_gpu.load_param("/tmp/perf_ditout.param") || net_gpu.load_model("/tmp/perf_ditout.bin"))
        return {"dit_output", 0, 0, false};

    { ncnn::Mat o[2]; ncnn::Extractor e = net_cpu.create_extractor(); e.input("vid", video); e.input("emb", embedding); e.input("vid_shape", shape); e.extract("vid_out", o[0]); e.extract("output_shape", o[1]); }
    { ncnn::Mat o[2]; ncnn::Extractor e = net_gpu.create_extractor(); e.input("vid", video); e.input("emb", embedding); e.input("vid_shape", shape); e.extract("vid_out", o[0]); e.extract("output_shape", o[1]); }

    std::vector<double> cpu_t, gpu_t;
    ncnn::Mat ref;
    for (int i = 0; i < 5; i++)
    {
        auto t0 = Clock::now();
        ncnn::Mat o[2]; ncnn::Extractor e = net_cpu.create_extractor(); e.input("vid", video); e.input("emb", embedding); e.input("vid_shape", shape); e.extract("vid_out", o[0]); e.extract("output_shape", o[1]);
        cpu_t.push_back(ms_since(t0));
        if (i == 0) ref = o[0];
    }
    double worst = 0.0;
    for (int i = 0; i < 5; i++)
    {
        auto t0 = Clock::now();
        ncnn::Mat o[2]; ncnn::Extractor e = net_gpu.create_extractor(); e.input("vid", video); e.input("emb", embedding); e.input("vid_shape", shape); e.extract("vid_out", o[0]); e.extract("output_shape", o[1]);
        gpu_t.push_back(ms_since(t0));
        worst = std::max(worst, max_abs_diff(ref, o[0]));
    }
    return {"dit_output", median(cpu_t), median(gpu_t), worst < 1e-2};
}

// ============================================================================
// 7. SeedVR2DiTBlock（复用 5 个 InnerProduct + 7 个自定义 shader）
// ============================================================================
BenchResult bench_dit_block(int dim, int heads, int head_dim, int mlp_hidden, int T, int H, int W)
{
    FILE* fp = fopen("/tmp/perf_ditb.param", "wb");
    fprintf(fp, "7767517\n5 6\n");
    fprintf(fp, "Input input_vid 0 1 vid\nInput input_txt 0 1 txt\nInput input_emb 0 1 emb\nInput input_vid_shape 0 1 vid_shape\n");
    fprintf(fp, "SeedVR2DiTBlock dit_block 4 2 vid txt emb vid_shape vid_out txt_out 0=0 1=0 2=0 3=0 4=%d 5=%d 6=%d 7=%d 8=1e-05\n",
            dim, heads, head_dim, mlp_hidden);
    fclose(fp);
    fp = fopen("/tmp/perf_ditb.bin", "wb");
    auto write_vec = [&](FILE* f, int n, int seed) {
        std::vector<float> v(n);
        for (size_t i = 0; i < v.size(); i++) v[i] = 0.01f * static_cast<float>(static_cast<int>((i + seed) % 9) - 4);
        fwrite(v.data(), sizeof(float), v.size(), f);
    };
    auto write_ip = [&](FILE* f, int in, int out, bool bias, int seed) {
        std::vector<float> w(static_cast<size_t>(in) * out);
        for (size_t i = 0; i < w.size(); i++) w[i] = 0.005f * static_cast<float>(static_cast<int>((i + seed) % 7) - 3);
        fwrite(w.data(), sizeof(float), w.size(), f);
        if (bias) { std::vector<float> b(out); for (size_t i = 0; i < b.size(); i++) b[i] = 0.01f * static_cast<float>(static_cast<int>((i + seed) % 5) - 2); fwrite(b.data(), sizeof(float), b.size(), f); }
    };
    auto write_branch = [&](FILE* f, int base) {
        write_vec(f, dim, base + 0); write_vec(f, dim, base + 1); write_vec(f, dim, base + 2);
        write_vec(f, dim, base + 3); write_vec(f, dim, base + 4); write_vec(f, dim, base + 5);
        write_ip(f, dim, dim * 3, false, base + 6);
        write_ip(f, dim, dim, true, base + 7);
        write_vec(f, head_dim, base + 8); write_vec(f, head_dim, base + 9);
        write_ip(f, dim, mlp_hidden, false, base + 10);
        write_ip(f, dim, mlp_hidden, false, base + 11);
        write_ip(f, mlp_hidden, dim, false, base + 12);
    };
    write_branch(fp, 0);
    write_branch(fp, 50);
    write_vec(fp, 21, 100);
    fclose(fp);

    const int video_len = T * H * W;
    ncnn::Mat vid, txt, embedding, shape;
    fill2d(vid, dim, video_len, 0);
    fill2d(txt, dim, 1024, 100);
    fill2d(embedding, dim * 6, 1, 200);
    shape.create(3, static_cast<size_t>(4u));
    int* s = shape; s[0] = T; s[1] = H; s[2] = W;

    ncnn::Net net_cpu, net_gpu;
    set_opts(net_cpu, false);
    set_opts(net_gpu, true);
    net_cpu.register_custom_layer("SeedVR2DiTBlock", SeedVR2DiTBlock_layer_creator);
    net_gpu.register_custom_layer("SeedVR2DiTBlock", SeedVR2DiTBlock_layer_creator);
    if (net_cpu.load_param("/tmp/perf_ditb.param") || net_cpu.load_model("/tmp/perf_ditb.bin")
        || net_gpu.load_param("/tmp/perf_ditb.param") || net_gpu.load_model("/tmp/perf_ditb.bin"))
        return {"dit_block", 0, 0, false};

    { ncnn::Mat o[2]; ncnn::Extractor e = net_cpu.create_extractor(); e.input("vid", vid); e.input("txt", txt); e.input("emb", embedding); e.input("vid_shape", shape); e.extract("vid_out", o[0]); e.extract("txt_out", o[1]); }
    { ncnn::Mat o[2]; ncnn::Extractor e = net_gpu.create_extractor(); e.input("vid", vid); e.input("txt", txt); e.input("emb", embedding); e.input("vid_shape", shape); e.extract("vid_out", o[0]); e.extract("txt_out", o[1]); }

    std::vector<double> cpu_t, gpu_t;
    ncnn::Mat ref;
    for (int i = 0; i < 3; i++)
    {
        auto t0 = Clock::now();
        ncnn::Mat o[2]; ncnn::Extractor e = net_cpu.create_extractor(); e.input("vid", vid); e.input("txt", txt); e.input("emb", embedding); e.input("vid_shape", shape); e.extract("vid_out", o[0]); e.extract("txt_out", o[1]);
        cpu_t.push_back(ms_since(t0));
        if (i == 0) ref = o[0];
    }
    double worst = 0.0;
    for (int i = 0; i < 3; i++)
    {
        auto t0 = Clock::now();
        ncnn::Mat o[2]; ncnn::Extractor e = net_gpu.create_extractor(); e.input("vid", vid); e.input("txt", txt); e.input("emb", embedding); e.input("vid_shape", shape); e.extract("vid_out", o[0]); e.extract("txt_out", o[1]);
        gpu_t.push_back(ms_since(t0));
        worst = std::max(worst, max_abs_diff(ref, o[0]));
    }
    return {"dit_block", median(cpu_t), median(gpu_t), worst < 1e-2};
}

} // namespace

int main()
{
#if NCNN_VULKAN
    ncnn::create_gpu_instance();
    const int gpu_count = ncnn::get_gpu_count();
    std::printf("Vulkan GPU count = %d\n", gpu_count);
#endif

    std::printf("%-14s  %-20s %-20s %s\n", "算子", "CPU(ms)", "Vulkan(ms)", "加速比");
    std::printf("----------------------------------------------------------------\n");

    // 真实规模（接近实际模型，CPU 可跑的量级）
    report(bench_shuffle(512, 5, 16, 16, 2));          // VAE decoder 上采样
    report(bench_groupnorm(512, 16, 16, 16));          // VAE 残差块
    report(bench_attention(512, 8, 16, 16));           // VAE bottleneck 单头注意力
    report(bench_conv3d(16, 16, 8, 128, 128));         // VAE 3x3x3 时空卷积
    report(bench_dit_input(1024, 33, 1024, 8, 8, 8));  // DiT 输入（patchify + emb MLP）
    report(bench_dit_output(1024, 16, 8, 8, 8));       // DiT 输出（unpatchify + 投影）
    report(bench_dit_block(1024, 8, 128, 4096, 8, 8, 8)); // DiT transformer block

#if NCNN_VULKAN
    ncnn::destroy_gpu_instance();
#endif
    return 0;
}
