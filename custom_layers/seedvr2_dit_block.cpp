// 本文件实现 SeedVR2 3B 动态多模态 Transformer block 的 FP32 CPU 路径。
// 大矩阵乘法由 NCNN 原生 InnerProduct 完成；这里实现 batch=1 下按运行时
// T/H/W 生成普通或 shifted 窗口、视频/文本联合 MM-RoPE attention、AdaSingle
// 调制、残差和 SwiGLU 调度。该实现首先用于逐层数值基线，Vulkan 路径会复用
// 完全相同的窗口索引和权重布局，避免 CPU/GPU 两套语义发生偏差。
#include "seedvr2_dit_block.h"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

namespace
{
ncnn::Layer* load_inner_product(
    const ncnn::ModelBin& mb,
    int input_size,
    int output_size,
    bool bias)
{
    ncnn::Layer* layer = ncnn::create_layer("InnerProduct");
    if (!layer)
        return nullptr;
    ncnn::ParamDict pd;
    pd.set(0, output_size);
    pd.set(1, bias ? 1 : 0);
    pd.set(2, input_size * output_size);
    if (layer->load_param(pd) != 0 || layer->load_model(mb) != 0)
    {
        delete layer;
        return nullptr;
    }
    return layer;
}

inline float silu(float value)
{
    return value / (1.f + std::exp(-value));
}

inline int ceil_div(int value, int divisor)
{
    return (value + divisor - 1) / divisor;
}

inline float round_to_bfloat16(float value)
{
    // PyTorch 在调用 SDPA 前显式执行 .bfloat16()。这里采用 round-to-nearest
    // ties-to-even 后再以 FP32 保存，既复现数值又保持 NCNN Mat 的 FP32 ABI。
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t least_significant = (bits >> 16) & 1u;
    bits += 0x7fffu + least_significant;
    bits &= 0xffff0000u;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}
} // namespace

SeedVR2DiTBlock::SeedVR2DiTBlock()
    : block_index(0),
      shared_weights(false),
      last_layer(false),
      shifted_window(false),
      dim(2560),
      heads(20),
      head_dim(128),
      mlp_hidden(6912),
      norm_eps(1e-5f)
{
    one_blob_only = false;
    support_inplace = false;
    support_packing = false;
}

SeedVR2DiTBlock::~SeedVR2DiTBlock()
{
    destroy_branch(vid_weights);
    if (!shared_weights)
        destroy_branch(txt_weights);
}

void SeedVR2DiTBlock::destroy_branch(BranchWeights& branch)
{
    delete branch.qkv;
    delete branch.proj_out;
    delete branch.mlp_gate_proj;
    delete branch.mlp_in_proj;
    delete branch.mlp_out_proj;
    branch.qkv = nullptr;
    branch.proj_out = nullptr;
    branch.mlp_gate_proj = nullptr;
    branch.mlp_in_proj = nullptr;
    branch.mlp_out_proj = nullptr;
}

int SeedVR2DiTBlock::load_param(const ncnn::ParamDict& pd)
{
    block_index = pd.get(0, 0);
    shared_weights = pd.get(1, 0) != 0;
    last_layer = pd.get(2, 0) != 0;
    shifted_window = pd.get(3, 0) != 0;
    dim = pd.get(4, 2560);
    heads = pd.get(5, 20);
    head_dim = pd.get(6, 128);
    mlp_hidden = pd.get(7, 6912);
    norm_eps = pd.get(8, 1e-5f);
    return dim == heads * head_dim ? 0 : -1;
}

int SeedVR2DiTBlock::load_branch(const ncnn::ModelBin& mb, BranchWeights& branch)
{
    branch.attn_shift = mb.load(dim, 1);
    branch.attn_scale = mb.load(dim, 1);
    branch.attn_gate = mb.load(dim, 1);
    branch.mlp_shift = mb.load(dim, 1);
    branch.mlp_scale = mb.load(dim, 1);
    branch.mlp_gate = mb.load(dim, 1);
    if (branch.attn_shift.empty() || branch.attn_scale.empty() || branch.attn_gate.empty()
        || branch.mlp_shift.empty() || branch.mlp_scale.empty() || branch.mlp_gate.empty())
        return -100;

    branch.qkv = load_inner_product(mb, dim, dim * 3, false);
    branch.proj_out = load_inner_product(mb, dim, dim, true);
    branch.norm_q = mb.load(head_dim, 1);
    branch.norm_k = mb.load(head_dim, 1);
    branch.mlp_gate_proj = load_inner_product(mb, dim, mlp_hidden, false);
    branch.mlp_in_proj = load_inner_product(mb, dim, mlp_hidden, false);
    branch.mlp_out_proj = load_inner_product(mb, mlp_hidden, dim, false);
    if (!branch.qkv || !branch.proj_out || branch.norm_q.empty() || branch.norm_k.empty()
        || !branch.mlp_gate_proj || !branch.mlp_in_proj || !branch.mlp_out_proj)
        return -100;
    return 0;
}

int SeedVR2DiTBlock::load_model(const ncnn::ModelBin& mb)
{
    if (load_branch(mb, vid_weights) != 0)
        return -100;
    if (!shared_weights && load_branch(mb, txt_weights) != 0)
        return -100;
    rope_freqs = mb.load(21, 1);
    return rope_freqs.empty() ? -100 : 0;
}

int SeedVR2DiTBlock::create_pipeline(const ncnn::Option& opt)
{
    auto create_branch = [&](BranchWeights& branch) {
        for (ncnn::Layer* layer : {branch.qkv, branch.proj_out, branch.mlp_gate_proj,
                                  branch.mlp_in_proj, branch.mlp_out_proj})
        {
            if (layer && layer->create_pipeline(opt) != 0)
                return -1;
        }
        return 0;
    };
    if (create_branch(vid_weights) != 0)
        return -1;
    return shared_weights ? 0 : create_branch(txt_weights);
}

int SeedVR2DiTBlock::destroy_pipeline(const ncnn::Option& opt)
{
    auto destroy = [&](BranchWeights& branch) {
        for (ncnn::Layer* layer : {branch.qkv, branch.proj_out, branch.mlp_gate_proj,
                                  branch.mlp_in_proj, branch.mlp_out_proj})
            if (layer)
                layer->destroy_pipeline(opt);
    };
    destroy(vid_weights);
    if (!shared_weights)
        destroy(txt_weights);
    return 0;
}

void SeedVR2DiTBlock::apply_rmsnorm(ncnn::Mat& value) const
{
    for (int row = 0; row < value.h; row++)
    {
        float* data = value.row(row);
        double square_sum = 0.0;
        for (int channel = 0; channel < dim; channel++)
            square_sum += static_cast<double>(data[channel]) * data[channel];
        const float inverse_rms = 1.f / std::sqrt(static_cast<float>(square_sum / dim) + norm_eps);
        for (int channel = 0; channel < dim; channel++)
            data[channel] *= inverse_rms;
    }
}

void SeedVR2DiTBlock::apply_ada_input(
    ncnn::Mat& value,
    const ncnn::Mat& embedding,
    const ncnn::Mat& shift,
    const ncnn::Mat& scale,
    int layer_index) const
{
    const float* emb = embedding;
    const float* shift_b = shift;
    const float* scale_b = scale;
    const int slot = layer_index * 3;
#pragma omp parallel for
    for (int row = 0; row < value.h; row++)
    {
        float* data = value.row(row);
        for (int channel = 0; channel < dim; channel++)
        {
            const int offset = channel * 6 + slot;
            data[channel] = data[channel] * (emb[offset + 1] + scale_b[channel])
                + emb[offset] + shift_b[channel];
        }
    }
}

void SeedVR2DiTBlock::apply_ada_output_and_residual(
    ncnn::Mat& value,
    const ncnn::Mat& residual,
    const ncnn::Mat& embedding,
    const ncnn::Mat& gate,
    int layer_index) const
{
    const float* emb = embedding;
    const float* gate_b = gate;
    const int slot = layer_index * 3 + 2;
#pragma omp parallel for
    for (int row = 0; row < value.h; row++)
    {
        float* data = value.row(row);
        const float* skip = residual.row(row);
        for (int channel = 0; channel < dim; channel++)
            data[channel] = data[channel] * (emb[channel * 6 + slot] + gate_b[channel])
                + skip[channel];
    }
}

std::vector<SeedVR2DiTBlock::Window> SeedVR2DiTBlock::make_windows(
    int frames,
    int height,
    int width) const
{
    const double scale = std::sqrt(3600.0 / static_cast<double>(height * width));
    // Python round 使用 ties-to-even；nearbyint 在默认舍入模式下保持相同规则。
    const int resized_height = static_cast<int>(std::nearbyint(height * scale));
    const int resized_width = static_cast<int>(std::nearbyint(width * scale));
    const int window_height = ceil_div(resized_height, 3);
    const int window_width = ceil_div(resized_width, 3);
    const int window_frames = ceil_div(std::min(frames, 30), 4);

    const double shift_t = shifted_window && window_frames < frames ? 0.5 : 0.0;
    const double shift_h = shifted_window && window_height < height ? 0.5 : 0.0;
    const double shift_w = shifted_window && window_width < width ? 0.5 : 0.0;
    int count_t = shifted_window
        ? (shift_t > 0.0 ? static_cast<int>(std::ceil((frames - shift_t) / window_frames)) + 1 : 1)
        : ceil_div(frames, window_frames);
    int count_h = shifted_window
        ? (shift_h > 0.0 ? static_cast<int>(std::ceil((height - shift_h) / window_height)) + 1 : 1)
        : ceil_div(height, window_height);
    int count_w = shifted_window
        ? (shift_w > 0.0 ? static_cast<int>(std::ceil((width - shift_w) / window_width)) + 1 : 1)
        : ceil_div(width, window_width);

    std::vector<Window> windows;
    for (int iw = 0; iw < count_w; iw++)
        for (int ih = 0; ih < count_h; ih++)
            for (int it = 0; it < count_t; it++)
            {
                const int t0 = std::max(static_cast<int>((it - shift_t) * window_frames), 0);
                const int t1 = std::min(static_cast<int>((it - shift_t + 1.0) * window_frames), frames);
                const int h0 = std::max(static_cast<int>((ih - shift_h) * window_height), 0);
                const int h1 = std::min(static_cast<int>((ih - shift_h + 1.0) * window_height), height);
                const int w0 = std::max(static_cast<int>((iw - shift_w) * window_width), 0);
                const int w1 = std::min(static_cast<int>((iw - shift_w + 1.0) * window_width), width);
                if (t1 > t0 && h1 > h0 && w1 > w0)
                    windows.push_back({t0, t1, h0, h1, w0, w1});
            }
    return windows;
}

int SeedVR2DiTBlock::attention(
    const ncnn::Mat& vid_qkv,
    const ncnn::Mat& txt_qkv,
    const ncnn::Mat& vid_shape,
    const BranchWeights& vid_branch,
    const BranchWeights& txt_branch,
    ncnn::Mat& vid_output,
    ncnn::Mat& txt_output,
    const ncnn::Option& opt) const
{
    const int frames = static_cast<const int*>(vid_shape)[0];
    const int height = static_cast<const int*>(vid_shape)[1];
    const int width = static_cast<const int*>(vid_shape)[2];
    if (frames * height * width != vid_qkv.h)
        return -1;
    const int text_length = txt_qkv.h;
    const std::vector<Window> windows = make_windows(frames, height, width);
    if (windows.empty())
        return -1;

    vid_output.create(dim, vid_qkv.h, 4u, opt.blob_allocator);
    txt_output.create(dim, text_length, 4u, opt.blob_allocator);
    if (vid_output.empty() || txt_output.empty())
        return -100;
    vid_output.fill(0.f);
    txt_output.fill(0.f);
    const float attention_scale = 1.f / std::sqrt(static_cast<float>(head_dim));
    const float* frequencies = rope_freqs;

    // 每个 head 独占自己的输出通道，窗口在同一 head 内顺序执行，避免文本
    // 汇聚产生写冲突；视频窗口本身是分区，每个 token 只写一次。
#pragma omp parallel for num_threads(opt.num_threads)
    for (int head = 0; head < heads; head++)
    {
        std::vector<float> text_accumulator(static_cast<size_t>(text_length) * head_dim, 0.f);
        for (const Window& window : windows)
        {
            const int local_t = window.t1 - window.t0;
            const int local_h = window.h1 - window.h0;
            const int local_w = window.w1 - window.w0;
            const int video_length = local_t * local_h * local_w;
            const int sequence_length = video_length + text_length;
            std::vector<int> video_indices;
            video_indices.reserve(video_length);
            std::vector<float> q(static_cast<size_t>(sequence_length) * head_dim);
            std::vector<float> k(static_cast<size_t>(sequence_length) * head_dim);
            std::vector<float> v(static_cast<size_t>(sequence_length) * head_dim);
            std::vector<float> scores(sequence_length);

            auto normalize_and_rotate = [&](const float* source, const float* gamma,
                                            float* destination, int position_t,
                                            int position_h, int position_w) {
                double square_sum = 0.0;
                for (int d = 0; d < head_dim; d++)
                    square_sum += static_cast<double>(source[d]) * source[d];
                const float inverse_rms = 1.f / std::sqrt(static_cast<float>(square_sum / head_dim) + norm_eps);
                for (int d = 0; d < head_dim; d++)
                    destination[d] = source[d] * inverse_rms * gamma[d];
                const int positions[3] = {position_t, position_h, position_w};
                for (int axis = 0; axis < 3; axis++)
                    for (int pair = 0; pair < 21; pair++)
                    {
                        const int d = axis * 42 + pair * 2;
                        const float angle = positions[axis] * frequencies[pair];
                        const float cosine = std::cos(angle);
                        const float sine = std::sin(angle);
                        const float first = destination[d];
                        const float second = destination[d + 1];
                        destination[d] = first * cosine - second * sine;
                        destination[d + 1] = second * cosine + first * sine;
                    }
                for (int d = 0; d < head_dim; d++)
                    destination[d] = round_to_bfloat16(destination[d]);
            };

            int token = 0;
            for (int t = window.t0; t < window.t1; t++)
                for (int y = window.h0; y < window.h1; y++)
                    for (int x = window.w0; x < window.w1; x++, token++)
                    {
                        const int index = (t * height + y) * width + x;
                        video_indices.push_back(index);
                        const float* row = vid_qkv.row(index);
                        normalize_and_rotate(
                            row + head * head_dim,
                            vid_branch.norm_q,
                            q.data() + static_cast<size_t>(token) * head_dim,
                            text_length + (t - window.t0), y - window.h0, x - window.w0);
                        normalize_and_rotate(
                            row + dim + head * head_dim,
                            vid_branch.norm_k,
                            k.data() + static_cast<size_t>(token) * head_dim,
                            text_length + (t - window.t0), y - window.h0, x - window.w0);
                        std::memcpy(
                            v.data() + static_cast<size_t>(token) * head_dim,
                            row + dim * 2 + head * head_dim,
                            static_cast<size_t>(head_dim) * sizeof(float));
                        float* value = v.data() + static_cast<size_t>(token) * head_dim;
                        for (int d = 0; d < head_dim; d++)
                            value[d] = round_to_bfloat16(value[d]);
                    }
            for (int text = 0; text < text_length; text++, token++)
            {
                const float* row = txt_qkv.row(text);
                normalize_and_rotate(
                    row + head * head_dim,
                    txt_branch.norm_q,
                    q.data() + static_cast<size_t>(token) * head_dim,
                    text, text, text);
                normalize_and_rotate(
                    row + dim + head * head_dim,
                    txt_branch.norm_k,
                    k.data() + static_cast<size_t>(token) * head_dim,
                    text, text, text);
                std::memcpy(
                    v.data() + static_cast<size_t>(token) * head_dim,
                    row + dim * 2 + head * head_dim,
                    static_cast<size_t>(head_dim) * sizeof(float));
                float* value = v.data() + static_cast<size_t>(token) * head_dim;
                for (int d = 0; d < head_dim; d++)
                    value[d] = round_to_bfloat16(value[d]);
            }

            for (int query = 0; query < sequence_length; query++)
            {
                const float* query_data = q.data() + static_cast<size_t>(query) * head_dim;
                float maximum = -std::numeric_limits<float>::infinity();
                for (int key = 0; key < sequence_length; key++)
                {
                    const float* key_data = k.data() + static_cast<size_t>(key) * head_dim;
                    double dot = 0.0;
                    for (int d = 0; d < head_dim; d++)
                        dot += static_cast<double>(query_data[d]) * key_data[d];
                    scores[key] = static_cast<float>(dot) * attention_scale;
                    maximum = std::max(maximum, scores[key]);
                }
                float denominator = 0.f;
                for (float& score : scores)
                {
                    score = std::exp(score - maximum);
                    denominator += score;
                }
                float* destination = nullptr;
                if (query < video_length)
                    destination = vid_output.row(video_indices[query]) + head * head_dim;
                else
                    destination = text_accumulator.data()
                        + static_cast<size_t>(query - video_length) * head_dim;
                for (int d = 0; d < head_dim; d++)
                {
                    double sum = 0.0;
                    for (int key = 0; key < sequence_length; key++)
                        sum += static_cast<double>(scores[key] / denominator)
                            * v[static_cast<size_t>(key) * head_dim + d];
                    destination[d] += round_to_bfloat16(static_cast<float>(sum));
                }
            }
        }
        const float inverse_window_count = 1.f / windows.size();
        for (int text = 0; text < text_length; text++)
        {
            float* destination = txt_output.row(text) + head * head_dim;
            const float* source = text_accumulator.data() + static_cast<size_t>(text) * head_dim;
            for (int d = 0; d < head_dim; d++)
                destination[d] = source[d] * inverse_window_count;
        }
    }
    return 0;
}

int SeedVR2DiTBlock::apply_mlp(
    const ncnn::Mat& input,
    const BranchWeights& branch,
    ncnn::Mat& output,
    const ncnn::Option& opt) const
{
    ncnn::Mat gate;
    ncnn::Mat value;
    if (branch.mlp_gate_proj->forward(input, gate, opt) != 0
        || branch.mlp_in_proj->forward(input, value, opt) != 0)
        return -100;
#pragma omp parallel for num_threads(opt.num_threads)
    for (int row = 0; row < gate.h; row++)
    {
        float* gate_data = gate.row(row);
        const float* value_data = value.row(row);
        for (int channel = 0; channel < mlp_hidden; channel++)
            gate_data[channel] = silu(gate_data[channel]) * value_data[channel];
    }
    return branch.mlp_out_proj->forward(gate, output, opt);
}

int SeedVR2DiTBlock::forward(
    const std::vector<ncnn::Mat>& bottom_blobs,
    std::vector<ncnn::Mat>& top_blobs,
    const ncnn::Option& opt) const
{
    if (bottom_blobs.size() != 4 || top_blobs.size() != 2)
    {
        std::fprintf(stderr, "SeedVR2DiTBlock[%d]: blob count mismatch (bottom=%zu top=%zu)\n",
                     block_index, bottom_blobs.size(), top_blobs.size());
        return -1;
    }
    const ncnn::Mat& vid = bottom_blobs[0];
    const ncnn::Mat& txt = bottom_blobs[1];
    const ncnn::Mat& embedding = bottom_blobs[2];
    const ncnn::Mat& vid_shape = bottom_blobs[3];
    if (vid.dims != 2 || txt.dims != 2 || vid.w != dim || txt.w != dim
        || embedding.w != dim * 6 || vid_shape.dims != 1 || vid_shape.w != 3)
    {
        std::fprintf(
            stderr,
            "SeedVR2DiTBlock[%d]: invalid input vid=(dims=%d w=%d h=%d) "
            "txt=(dims=%d w=%d h=%d) emb=%zu shape=%zu expected_dim=%d\n",
            block_index, vid.dims, vid.w, vid.h, txt.dims, txt.w, txt.h,
            embedding.total(), vid_shape.total(), dim);
        return -1;
    }
    const BranchWeights& text_branch = shared_weights ? vid_weights : txt_weights;

    ncnn::Mat vid_attention_input = vid.clone(opt.blob_allocator);
    ncnn::Mat txt_attention_input = txt.clone(opt.blob_allocator);
    if (vid_attention_input.empty() || txt_attention_input.empty())
        return -100;
    apply_rmsnorm(vid_attention_input);
    apply_rmsnorm(txt_attention_input);
    apply_ada_input(vid_attention_input, embedding, vid_weights.attn_shift, vid_weights.attn_scale, 0);
    // 最后一个 PyTorch block 的 Ada MMModule 配置为 vid_only，文本仅经过
    // RMSNorm，不应用 attention shift/scale。
    if (!last_layer)
        apply_ada_input(txt_attention_input, embedding, text_branch.attn_shift, text_branch.attn_scale, 0);

    ncnn::Mat vid_qkv;
    ncnn::Mat txt_qkv;
    const int vid_qkv_result = vid_weights.qkv->forward(vid_attention_input, vid_qkv, opt);
    const int txt_qkv_result = text_branch.qkv->forward(txt_attention_input, txt_qkv, opt);
    if (vid_qkv_result != 0 || txt_qkv_result != 0)
    {
        std::fprintf(stderr, "SeedVR2DiTBlock[%d]: qkv failed (vid=%d txt=%d)\n",
                     block_index, vid_qkv_result, txt_qkv_result);
        return -100;
    }
    ncnn::Mat vid_attention;
    ncnn::Mat txt_attention;
    const int attention_result = attention(vid_qkv, txt_qkv, vid_shape, vid_weights,
                                           text_branch, vid_attention, txt_attention, opt);
    if (attention_result != 0)
    {
        std::fprintf(stderr, "SeedVR2DiTBlock[%d]: attention failed (%d)\n",
                     block_index, attention_result);
        return -100;
    }
    ncnn::Mat vid_projected;
    ncnn::Mat txt_projected;
    const int vid_proj_result = vid_weights.proj_out->forward(vid_attention, vid_projected, opt);
    const int txt_proj_result = text_branch.proj_out->forward(txt_attention, txt_projected, opt);
    if (vid_proj_result != 0 || txt_proj_result != 0)
    {
        std::fprintf(stderr, "SeedVR2DiTBlock[%d]: attention projection failed (vid=%d txt=%d)\n",
                     block_index, vid_proj_result, txt_proj_result);
        return -100;
    }
    apply_ada_output_and_residual(vid_projected, vid, embedding, vid_weights.attn_gate, 0);
    if (!last_layer)
    {
        apply_ada_output_and_residual(txt_projected, txt, embedding, text_branch.attn_gate, 0);
    }
    else
    {
        // vid_only 同样跳过文本 attention gate，但 block 外层普通残差仍存在。
#pragma omp parallel for num_threads(opt.num_threads)
        for (int row = 0; row < txt_projected.h; row++)
        {
            float* data = txt_projected.row(row);
            const float* skip = txt.row(row);
            for (int channel = 0; channel < dim; channel++)
                data[channel] += skip[channel];
        }
    }

    ncnn::Mat vid_mlp_input = vid_projected.clone(opt.blob_allocator);
    apply_rmsnorm(vid_mlp_input);
    apply_ada_input(vid_mlp_input, embedding, vid_weights.mlp_shift, vid_weights.mlp_scale, 1);
    ncnn::Mat vid_mlp;
    const int vid_mlp_result = apply_mlp(vid_mlp_input, vid_weights, vid_mlp, opt);
    if (vid_mlp_result != 0)
    {
        std::fprintf(stderr, "SeedVR2DiTBlock[%d]: video MLP failed (%d)\n",
                     block_index, vid_mlp_result);
        return -100;
    }
    apply_ada_output_and_residual(vid_mlp, vid_projected, embedding, vid_weights.mlp_gate, 1);
    top_blobs[0] = vid_mlp;

    if (last_layer)
    {
        // PyTorch 的 MMModule(vid_only=True) 会原样返回 txt，随后 block 仍然
        // 执行 txt_mlp + txt_attn，因此最后一个 block 的文本结果是两倍
        // txt_attn。这里保留该看似多余但对模型数值兼容必需的行为。
        ncnn::Mat txt_last = txt_projected.clone(opt.blob_allocator);
        if (txt_last.empty())
            return -100;
#pragma omp parallel for num_threads(opt.num_threads)
        for (int row = 0; row < txt_last.h; row++)
        {
            float* data = txt_last.row(row);
            for (int channel = 0; channel < dim; channel++)
                data[channel] *= 2.f;
        }
        top_blobs[1] = txt_last;
        return 0;
    }
    ncnn::Mat txt_mlp_input = txt_projected.clone(opt.blob_allocator);
    apply_rmsnorm(txt_mlp_input);
    apply_ada_input(txt_mlp_input, embedding, text_branch.mlp_shift, text_branch.mlp_scale, 1);
    ncnn::Mat txt_mlp;
    const int txt_mlp_result = apply_mlp(txt_mlp_input, text_branch, txt_mlp, opt);
    if (txt_mlp_result != 0)
    {
        std::fprintf(stderr, "SeedVR2DiTBlock[%d]: text MLP failed (%d)\n",
                     block_index, txt_mlp_result);
        return -100;
    }
    apply_ada_output_and_residual(txt_mlp, txt_projected, embedding, text_branch.mlp_gate, 1);
    top_blobs[1] = txt_mlp;
    return 0;
}

DEFINE_LAYER_CREATOR(SeedVR2DiTBlock)
