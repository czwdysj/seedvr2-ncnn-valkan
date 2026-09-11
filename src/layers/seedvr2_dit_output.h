// 本文件声明 SeedVR2 3B DiT 的动态输出层。
// 输入为最后一个 block 的视频 token、15360 维 timestep embedding 和 patch
// 形状[T,H,W]；输出恢复为 [T,H*2,W*2,16] 的扁平 token 及对应动态形状。
//
// 本层是 DiTInput 的对称逆操作：affine RMSNorm + output Ada 调制 + 投影 +
// 2x2 unpatchify（把 4 通道拆回 2x2 空间位置）。投影复用 ncnn 原生
// InnerProduct（create_layer 返回 Layer_final，自动获得 CPU + Vulkan 双实现），
// 自定义代码负责三个原生算子表达不了的 shader：逐 token 平方和归约
// （norm_reduce）、RMSNorm + Ada 调制（norm_apply）、2x2 空间反重排（unpatchify）。
#pragma once

#include <layer.h>

class SeedVR2DiTOutput : public ncnn::Layer
{
public:
    SeedVR2DiTOutput();
    ~SeedVR2DiTOutput() override;

    int load_param(const ncnn::ParamDict& pd) override;
    int load_model(const ncnn::ModelBin& mb) override;
    int create_pipeline(const ncnn::Option& opt) override;
    int destroy_pipeline(const ncnn::Option& opt) override;
    int forward(const std::vector<ncnn::Mat>& bottom_blobs,
                std::vector<ncnn::Mat>& top_blobs,
                const ncnn::Option& opt) const override;

    // unpatchify 的输出尺寸由 DiT 调度器直接注入，Vulkan forward 不再下载
    // vid_shape，也不会在输出头之前强制提交整条命令队列。
    void set_runtime_shape(int frames, int height, int width);

#if NCNN_VULKAN
    int upload_model(ncnn::VkTransfer& cmd, const ncnn::Option& opt) override;
    int forward(const std::vector<ncnn::VkMat>& bottom_blobs,
                std::vector<ncnn::VkMat>& top_blobs,
                ncnn::VkCompute& cmd,
                const ncnn::Option& opt) const override;
#endif

private:
    int dim;
    int output_channels;
    float norm_eps;
    ncnn::Mat norm_weight;
    ncnn::Mat output_shift;
    ncnn::Mat output_scale;
    ncnn::Layer* projection;
    int runtime_frames;
    int runtime_height;
    int runtime_width;
    bool runtime_shape_valid;

#if NCNN_VULKAN
    // 三个调制权重的 GPU 缓冲（upload_model 一次性上传，forward 零搬运）。
    ncnn::VkMat norm_weight_gpu;
    ncnn::VkMat output_shift_gpu;
    ncnn::VkMat output_scale_gpu;
    // 三个自定义 compute shader 的管线（投影走原生 InnerProduct 的 Vulkan 实现）。
    ncnn::Pipeline* pipeline_norm_reduce;
    ncnn::Pipeline* pipeline_norm_apply;
    ncnn::Pipeline* pipeline_unpatchify;
#endif
};

ncnn::Layer* SeedVR2DiTOutput_layer_creator(void* userdata);
