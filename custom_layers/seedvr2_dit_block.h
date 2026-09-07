// 本文件声明 SeedVR2 3B 的动态多模态 Transformer block 自定义层。
// 输入为 vid[L_v,2560]、txt[L_t,2560]、timestep embedding[15360] 和
// patch 后视频形状[T,H,W]，输出保持 vid/txt token 数不变。该层内部复用
// NCNN 原生 InnerProduct 执行大矩阵乘法，只负责 Ada、动态窗口、MM-RoPE、
// 变长 attention 和共享权重调度；batch 固定为 1，T/H/W 与文本长度动态。
//
// Vulkan 路径复用同一份权重顺序与动态形状约束：5 个矩阵乘投影（qkv/proj_out/
// mlp_gate_proj/mlp_in_proj/mlp_out_proj）复用 InnerProduct 的 Vulkan 实现，
// 自定义代码负责 6 个原生算子表达不了的 shader——rmsnorm_ada（逐 token RMSNorm
// + adaLN 调制，reduce+apply 两段）、qkv_prepare（MM-RoPE 旋转 + bf16 舍入）、
// attention（softmax 三遍 QK^T，不物化 scores）、ada_output_residual（门控残差）、
// silu（SwiGLU 门控）、text_normalize（文本跨窗口平均）。
#pragma once

#include <layer.h>

class SeedVR2DiTBlock : public ncnn::Layer
{
public:
    SeedVR2DiTBlock();
    ~SeedVR2DiTBlock() override;

    int load_param(const ncnn::ParamDict& pd) override;
    int load_model(const ncnn::ModelBin& mb) override;
    int create_pipeline(const ncnn::Option& opt) override;
    int destroy_pipeline(const ncnn::Option& opt) override;
    int forward(
        const std::vector<ncnn::Mat>& bottom_blobs,
        std::vector<ncnn::Mat>& top_blobs,
        const ncnn::Option& opt) const override;

#if NCNN_VULKAN
    int upload_model(ncnn::VkTransfer& cmd, const ncnn::Option& opt) override;
    int forward(
        const std::vector<ncnn::VkMat>& bottom_blobs,
        std::vector<ncnn::VkMat>& top_blobs,
        ncnn::VkCompute& cmd,
        const ncnn::Option& opt) const override;
#endif

private:
    struct BranchWeights
    {
        ncnn::Mat attn_shift;
        ncnn::Mat attn_scale;
        ncnn::Mat attn_gate;
        ncnn::Mat mlp_shift;
        ncnn::Mat mlp_scale;
        ncnn::Mat mlp_gate;
        ncnn::Mat norm_q;
        ncnn::Mat norm_k;
        ncnn::Layer* qkv = nullptr;
        ncnn::Layer* proj_out = nullptr;
        ncnn::Layer* mlp_gate_proj = nullptr;
        ncnn::Layer* mlp_in_proj = nullptr;
        ncnn::Layer* mlp_out_proj = nullptr;
#if NCNN_VULKAN
        // 6 个调制权重 + norm_q/norm_k 的 GPU 缓冲（upload_model 一次性上传）。
        ncnn::VkMat attn_shift_gpu;
        ncnn::VkMat attn_scale_gpu;
        ncnn::VkMat attn_gate_gpu;
        ncnn::VkMat mlp_shift_gpu;
        ncnn::VkMat mlp_scale_gpu;
        ncnn::VkMat mlp_gate_gpu;
        ncnn::VkMat norm_q_gpu;
        ncnn::VkMat norm_k_gpu;
#endif
    };

    struct Window
    {
        int t0;
        int t1;
        int h0;
        int h1;
        int w0;
        int w1;
    };

    int load_branch(const ncnn::ModelBin& mb, BranchWeights& branch);
    int attention(
        const ncnn::Mat& vid_qkv,
        const ncnn::Mat& txt_qkv,
        const ncnn::Mat& vid_shape,
        const BranchWeights& vid_branch,
        const BranchWeights& txt_branch,
        ncnn::Mat& vid_output,
        ncnn::Mat& txt_output,
        const ncnn::Option& opt) const;
    int apply_mlp(
        const ncnn::Mat& input,
        const BranchWeights& branch,
        ncnn::Mat& output,
        const ncnn::Option& opt) const;
    void apply_rmsnorm(ncnn::Mat& value) const;
    void apply_ada_input(
        ncnn::Mat& value,
        const ncnn::Mat& embedding,
        const ncnn::Mat& shift,
        const ncnn::Mat& scale,
        int layer_index) const;
    void apply_ada_output_and_residual(
        ncnn::Mat& value,
        const ncnn::Mat& residual,
        const ncnn::Mat& embedding,
        const ncnn::Mat& gate,
        int layer_index) const;
    std::vector<Window> make_windows(int frames, int height, int width) const;
    void destroy_branch(BranchWeights& branch);

private:
    int block_index;
    bool shared_weights;
    bool last_layer;
    bool shifted_window;
    int dim;
    int heads;
    int head_dim;
    int mlp_hidden;
    float norm_eps;
    BranchWeights vid_weights;
    BranchWeights txt_weights;
    ncnn::Mat rope_freqs;

#if NCNN_VULKAN
    ncnn::VkMat rope_freqs_gpu;
    // 6 个自定义 compute shader 的管线（矩阵乘走原生 InnerProduct 的 Vulkan 实现）。
    ncnn::Pipeline* pipeline_rmsnorm_reduce;
    ncnn::Pipeline* pipeline_rmsnorm_apply;
    ncnn::Pipeline* pipeline_qkv_prepare;
    ncnn::Pipeline* pipeline_attention;
    ncnn::Pipeline* pipeline_ada_residual;
    ncnn::Pipeline* pipeline_silu;
    ncnn::Pipeline* pipeline_text_normalize;
#endif
};

ncnn::Layer* SeedVR2DiTBlock_layer_creator(void* userdata);
