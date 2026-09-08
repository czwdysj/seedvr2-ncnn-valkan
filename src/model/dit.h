// 本文件声明 SeedVR2 3B DiT 的正式流式运行时。
// forward 接收 C,T,H,W 的 33 通道 latent 和 [tokens,5120] 文本 embedding，
// 内部运行输入头、32 个 Transformer block 和输出头，返回 16 通道 latent。
// 每次只加载一个 block 的 param/bin，保证 3B 权重不会同时驻留内存。
#pragma once

#include <net.h>

#include <memory>
#include <string>
#include <vector>

#include "core/runtime_context.h"

namespace seedvr2
{
class SeedVR2DiT
{
public:
    SeedVR2DiT();
    ~SeedVR2DiT();

    int load(const std::string& model_dir, const RuntimeContext& context);
    int forward(const ncnn::Mat& latent,
                const ncnn::Mat& text,
                float timestep,
                ncnn::Mat& output);

    const std::string& last_error() const noexcept { return last_error_; }

private:
    int run_block(int index,
                  const ncnn::Mat& video,
                  const ncnn::Mat& text,
                  const ncnn::Mat& embedding,
                  const ncnn::Mat& shape,
                  ncnn::Mat& video_output,
                  ncnn::Mat& text_output);

    const RuntimeContext* context_ = nullptr;
    std::string model_dir_;
    std::unique_ptr<ncnn::Net> input_;
    std::unique_ptr<ncnn::Net> output_;
    // 常驻模式：load 时一次性把 32 个 block 的 Net 全部载入；流式模式保持为空，
    // forward 时由 run_block 逐个临时加载。resident_ 由 RuntimeOptions.dit_resident 决定。
    std::vector<std::unique_ptr<ncnn::Net>> blocks_;
    bool resident_ = false;
    std::string last_error_;
};
} // namespace seedvr2
