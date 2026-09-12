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
struct DiTVulkanTransferStats
{
    int entry_upload_commands = 0;
    int final_download_commands = 0;
    int intermediate_download_commands = 0;
    int queue_submissions = 0;
};

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

#if NCNN_VULKAN
    // 单次 DiT 的 Vulkan 边界接口：入口集中上传，input/32 blocks/output 全程
    // 传递 VkMat，最后只下载一次输出。公开 Engine API 仍保持不变。
    int forward_vulkan(const ncnn::Mat& latent,
                       const ncnn::Mat& text,
                       float timestep,
                       ncnn::Mat& output);

    // Engine 内部零大张量传输接口。latent/output 使用 token-major 的二维布局
    // [T*H*W,C]（ncnn: w=C,h=T*H*W），text 为 [tokens,5120]。
    // shape/timestep 仍由 CPU 元数据控制，但不会下载任何 feature tensor。
    int forward_vkmat(const ncnn::VkMat& latent,
                      const ncnn::VkMat& text,
                      int frames,
                      int height,
                      int width,
                      float timestep,
                      VulkanExecutionContext& execution,
                      ncnn::VkMat& output);
#endif

    const std::string& last_error() const noexcept { return last_error_; }
    const DiTVulkanTransferStats& last_vulkan_transfer_stats() const noexcept
    {
        return last_vulkan_transfer_stats_;
    }

private:
    int run_block(int index,
                  const ncnn::Mat& video,
                  const ncnn::Mat& text,
                  const ncnn::Mat& embedding,
                  const ncnn::Mat& shape,
                  ncnn::Mat& video_output,
                  ncnn::Mat& text_output);

#if NCNN_VULKAN
    int run_block_vulkan(int index,
                         const ncnn::VkMat& video,
                         const ncnn::VkMat& text,
                         const ncnn::VkMat& embedding,
                         const ncnn::VkMat& shape,
                         int frames,
                         int height,
                         int width,
                         ncnn::VkAllocator* blob_allocator,
                         ncnn::VkAllocator* staging_allocator,
                         ncnn::VkCompute& command,
                         ncnn::VkMat& video_output,
                         ncnn::VkMat& text_output);
#endif

    const RuntimeContext* context_ = nullptr;
    std::string model_dir_;
    std::unique_ptr<ncnn::Net> input_;
    std::unique_ptr<ncnn::Net> output_;
    // 常驻模式：load 时一次性把 32 个 block 的 Net 全部载入；流式模式保持为空，
    // forward 时由 run_block 逐个临时加载。resident_ 由 RuntimeOptions.dit_resident 决定。
    std::vector<std::unique_ptr<ncnn::Net>> blocks_;
    bool resident_ = false;
    std::string last_error_;
    DiTVulkanTransferStats last_vulkan_transfer_stats_;
};
} // namespace seedvr2
