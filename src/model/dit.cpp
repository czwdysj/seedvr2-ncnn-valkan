// 本文件实现完整 DiT 的 CPU 与 Vulkan 调度。CPU 路径使用 Mat；Vulkan 路径在
// 入口集中上传后，以共享 allocator 和 VkCompute 串联输入头、32 个 block 和
// 输出头，最后只下载一次结果。流式模式允许逐 block GPU 同步以安全释放权重，
// 但中间 vid/txt/embedding 始终保留为 VkMat，不发生设备到主机的数据往返。
#include "model/dit.h"

#include <cstdio>
#include <filesystem>

#include "layers/seedvr2_dit_block.h"
#include "layers/seedvr2_dit_input.h"
#include "layers/seedvr2_dit_output.h"
#include "seedvr2/engine.h"

#if NCNN_VULKAN
#include <command.h>
#include <gpu.h>
#endif

namespace seedvr2
{
namespace
{
int load_io_net(const std::filesystem::path& param,
                const std::filesystem::path& bin,
                const RuntimeContext& context,
                const char* layer_name,
                ncnn::layer_creator_func creator,
                std::unique_ptr<ncnn::Net>& output,
                std::string& error)
{
    if (!std::filesystem::is_regular_file(param) || !std::filesystem::is_regular_file(bin))
    {
        error = "DiT model file is missing: " + param.string();
        return static_cast<int>(Status::ModelNotFound);
    }
    auto net = std::make_unique<ncnn::Net>();
    context.configure(*net);
    net->register_custom_layer(layer_name, creator);
    if (net->load_param(param.string().c_str()) != 0 || net->load_model(bin.string().c_str()) != 0)
    {
        error = "failed to load DiT model: " + param.string();
        return static_cast<int>(Status::ModelLoadFailed);
    }
    output = std::move(net);
    return static_cast<int>(Status::Ok);
}

// 常驻模式辅助：加载单个 block 的 Net（供 load 阶段一次性预加载 32 个 block）。
std::unique_ptr<ncnn::Net> load_block_net(int index,
                                          const std::filesystem::path& root,
                                          const RuntimeContext& context,
                                          std::string& error)
{
    char stem[64];
    std::snprintf(stem, sizeof(stem), "seedvr2_dit_block_%02d.ncnn", index);
    auto net = std::make_unique<ncnn::Net>();
    context.configure(*net);
    net->register_custom_layer("SeedVR2DiTBlock", SeedVR2DiTBlock_layer_creator);
    const std::string param = (root / (std::string(stem) + ".param")).string();
    const std::string bin = (root / (std::string(stem) + ".bin")).string();
    if (net->load_param(param.c_str()) != 0 || net->load_model(bin.c_str()) != 0)
    {
        error = "failed to load DiT block " + std::to_string(index);
        return nullptr;
    }
    return net;
}

ncnn::Mat flatten_latent(const ncnn::Mat& value)
{
    ncnn::Mat result(value.c,
                     value.d * value.h * value.w,
                     static_cast<size_t>(4u),
                     1);
    int token = 0;
    for (int frame = 0; frame < value.d; ++frame)
        for (int y = 0; y < value.h; ++y)
            for (int x = 0; x < value.w; ++x, ++token)
            {
                float* row = result.row(token);
                for (int channel = 0; channel < value.c; ++channel)
                    row[channel] = value.channel(channel).depth(frame).row(y)[x];
            }
    return result;
}

ncnn::Mat unflatten_latent(const ncnn::Mat& value, int frames, int height, int width)
{
    ncnn::Mat result(width, height, frames, value.w, 4u, 1);
    int token = 0;
    for (int frame = 0; frame < frames; ++frame)
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x, ++token)
            {
                const float* row = value.row(token);
                for (int channel = 0; channel < value.w; ++channel)
                    result.channel(channel).depth(frame).row(y)[x] = row[channel];
            }
    return result;
}

template <typename LayerType>
LayerType* find_custom_layer(ncnn::Net& net)
{
    for (ncnn::Layer* layer : net.mutable_layers())
    {
        if (auto* custom = dynamic_cast<LayerType*>(layer))
            return custom;
    }
    return nullptr;
}

#if NCNN_VULKAN
class VulkanExecutionScope
{
public:
    explicit VulkanExecutionScope(int device_index)
        : device_(ncnn::get_gpu_device(device_index)),
          blob_allocator_(device_ ? device_->acquire_blob_allocator() : nullptr),
          staging_allocator_(device_ ? device_->acquire_staging_allocator() : nullptr)
    {
    }

    ~VulkanExecutionScope()
    {
        if (device_ && staging_allocator_)
            device_->reclaim_staging_allocator(staging_allocator_);
        if (device_ && blob_allocator_)
            device_->reclaim_blob_allocator(blob_allocator_);
    }

    bool valid() const noexcept
    {
        return device_ && blob_allocator_ && staging_allocator_;
    }

    const ncnn::VulkanDevice* device() const noexcept { return device_; }
    ncnn::VkAllocator* blob_allocator() const noexcept { return blob_allocator_; }
    ncnn::VkAllocator* staging_allocator() const noexcept { return staging_allocator_; }

    ncnn::Option option(const ncnn::Option& base) const
    {
        ncnn::Option result = base;
        result.blob_vkallocator = blob_allocator_;
        result.workspace_vkallocator = blob_allocator_;
        result.staging_vkallocator = staging_allocator_;
        return result;
    }

    void configure(ncnn::Extractor& extractor) const
    {
        extractor.set_blob_vkallocator(blob_allocator_);
        extractor.set_workspace_vkallocator(blob_allocator_);
        extractor.set_staging_vkallocator(staging_allocator_);
    }

private:
    const ncnn::VulkanDevice* device_;
    ncnn::VkAllocator* blob_allocator_;
    ncnn::VkAllocator* staging_allocator_;
};
#endif
} // namespace

SeedVR2DiT::SeedVR2DiT() = default;
SeedVR2DiT::~SeedVR2DiT() = default;

int SeedVR2DiT::load(const std::string& model_dir, const RuntimeContext& context)
{
    const std::filesystem::path root(model_dir);
    // load 阶段先验证 32 对 param/bin，避免推理到中途才发现单个 block 缺失。
    for (int index = 0; index < 32; ++index)
    {
        char stem[64];
        std::snprintf(stem, sizeof(stem), "seedvr2_dit_block_%02d.ncnn", index);
        if (!std::filesystem::is_regular_file(root / (std::string(stem) + ".param"))
            || !std::filesystem::is_regular_file(root / (std::string(stem) + ".bin")))
        {
            last_error_ = "DiT block model is missing: " + std::string(stem);
            return static_cast<int>(Status::ModelNotFound);
        }
    }

    const int input_result = load_io_net(root / "seedvr2_dit_input.ncnn.param",
                                         root / "seedvr2_dit_input.ncnn.bin",
                                         context,
                                         "SeedVR2DiTInput",
                                         SeedVR2DiTInput_layer_creator,
                                         input_,
                                         last_error_);
    if (input_result != 0)
        return input_result;
    const int output_result = load_io_net(root / "seedvr2_dit_output.ncnn.param",
                                          root / "seedvr2_dit_output.ncnn.bin",
                                          context,
                                          "SeedVR2DiTOutput",
                                          SeedVR2DiTOutput_layer_creator,
                                          output_,
                                          last_error_);
    if (output_result != 0)
    {
        input_.reset();
        return output_result;
    }

    // 常驻模式：一次性把 32 个 block 的权重全 load 进内存（RAM + 后续 forward 上传显存）。
    // 流式模式（默认）保持 blocks_ 为空，forward 时由 run_block 逐个临时加载。
    resident_ = context.options().dit_resident;
    if (resident_)
    {
        blocks_.clear();
        blocks_.reserve(32);
        for (int index = 0; index < 32; ++index)
        {
            auto net = load_block_net(index, root, context, last_error_);
            if (!net)
            {
                blocks_.clear();
                input_.reset();
                output_.reset();
                return static_cast<int>(Status::ModelLoadFailed);
            }
            blocks_.push_back(std::move(net));
        }
    }

    context_ = &context;
    model_dir_ = model_dir;
    last_error_.clear();
    return static_cast<int>(Status::Ok);
}

int SeedVR2DiT::run_block(int index,
                          const ncnn::Mat& video,
                          const ncnn::Mat& text,
                          const ncnn::Mat& embedding,
                          const ncnn::Mat& shape,
                          ncnn::Mat& video_output,
                          ncnn::Mat& text_output)
{
    char stem[64];
    std::snprintf(stem, sizeof(stem), "seedvr2_dit_block_%02d.ncnn", index);
    ncnn::Net net;
    context_->configure(net);
    net.register_custom_layer("SeedVR2DiTBlock", SeedVR2DiTBlock_layer_creator);
    const std::string param = model_dir_ + "/" + stem + ".param";
    const std::string bin = model_dir_ + "/" + stem + ".bin";
    if (net.load_param(param.c_str()) != 0 || net.load_model(bin.c_str()) != 0)
    {
        last_error_ = "failed to stream-load DiT block " + std::to_string(index);
        return static_cast<int>(Status::ModelLoadFailed);
    }
    ncnn::Extractor extractor = net.create_extractor();
    if (extractor.input("vid", video) != 0 || extractor.input("txt", text) != 0
        || extractor.input("emb", embedding) != 0 || extractor.input("vid_shape", shape) != 0
        || extractor.extract("vid_out", video_output) != 0
        || extractor.extract("txt_out", text_output) != 0)
    {
        last_error_ = "DiT block inference failed at index " + std::to_string(index);
        return static_cast<int>(Status::InferenceFailed);
    }
    return static_cast<int>(Status::Ok);
}

#if NCNN_VULKAN
int SeedVR2DiT::run_block_vulkan(int index,
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
                                 ncnn::VkMat& text_output)
{
    char stem[64];
    std::snprintf(stem, sizeof(stem), "seedvr2_dit_block_%02d.ncnn", index);
    ncnn::Net net;
    context_->configure(net);
    net.register_custom_layer("SeedVR2DiTBlock", SeedVR2DiTBlock_layer_creator);
    const std::string param = model_dir_ + "/" + stem + ".param";
    const std::string bin = model_dir_ + "/" + stem + ".bin";
    if (net.load_param(param.c_str()) != 0 || net.load_model(bin.c_str()) != 0)
    {
        last_error_ = "failed to stream-load DiT block " + std::to_string(index);
        return static_cast<int>(Status::ModelLoadFailed);
    }

    SeedVR2DiTBlock* layer = find_custom_layer<SeedVR2DiTBlock>(net);
    if (!layer)
    {
        last_error_ = "DiT block custom layer is missing at index " + std::to_string(index);
        return static_cast<int>(Status::ModelLoadFailed);
    }
    layer->set_runtime_shape(frames, height, width);

    ncnn::Extractor extractor = net.create_extractor();
    extractor.set_blob_vkallocator(blob_allocator);
    extractor.set_workspace_vkallocator(blob_allocator);
    extractor.set_staging_vkallocator(staging_allocator);
    if (extractor.input("vid", video) != 0 || extractor.input("txt", text) != 0
        || extractor.input("emb", embedding) != 0 || extractor.input("vid_shape", shape) != 0
        || extractor.extract("vid_out", video_output, command) != 0
        || extractor.extract("txt_out", text_output, command) != 0)
    {
        last_error_ = "DiT Vulkan block inference failed at index " + std::to_string(index);
        return static_cast<int>(Status::InferenceFailed);
    }

    // 流式 Net 离开作用域后会销毁该 block 的 pipeline 和 GPU 权重，因此必须先
    // 等待本 block 完成。这里只同步命令，不下载 video/text 中间张量。
    if (command.submit_and_wait() != 0 || command.reset() != 0)
    {
        last_error_ = "DiT Vulkan block submission failed at index " + std::to_string(index);
        return static_cast<int>(Status::InferenceFailed);
    }
    ++last_vulkan_transfer_stats_.queue_submissions;
    return static_cast<int>(Status::Ok);
}
#endif

int SeedVR2DiT::forward(const ncnn::Mat& latent,
                        const ncnn::Mat& text,
                        float timestep,
                        ncnn::Mat& output)
{
#if NCNN_VULKAN
    if (context_ && context_->options().device == DeviceType::Vulkan)
        return forward_vulkan(latent, text, timestep, output);
#endif
    if (!input_ || !output_ || !context_)
    {
        last_error_ = "DiT is not loaded";
        return static_cast<int>(Status::NotLoaded);
    }
    if (latent.dims != 4 || latent.c != 33 || latent.elemsize != 4u || latent.elempack != 1
        || text.dims != 2 || text.w != 5120 || text.elemsize != 4u || text.elempack != 1)
    {
        last_error_ = "DiT expects latent [33,T,H,W] and text [tokens,5120]";
        return static_cast<int>(Status::InvalidArgument);
    }

    ncnn::Mat flat = flatten_latent(latent);
    if (flat.empty())
        return static_cast<int>(Status::OutOfMemory);
    ncnn::Mat shape(3, static_cast<size_t>(4u), 1);
    int* shape_data = shape;
    shape_data[0] = latent.d;
    shape_data[1] = latent.h;
    shape_data[2] = latent.w;
    ncnn::Mat timestep_mat(1, static_cast<size_t>(4u), 1);
    static_cast<float*>(timestep_mat.data)[0] = timestep;

    ncnn::Extractor input_extractor = input_->create_extractor();
    ncnn::Mat video_tokens;
    ncnn::Mat text_tokens;
    ncnn::Mat embedding;
    ncnn::Mat patched_shape;
    if (input_extractor.input("vid", flat) != 0 || input_extractor.input("txt", text) != 0
        || input_extractor.input("timestep", timestep_mat) != 0
        || input_extractor.input("vid_shape", shape) != 0
        || input_extractor.extract("vid_out", video_tokens) != 0
        || input_extractor.extract("txt_out", text_tokens) != 0
        || input_extractor.extract("emb", embedding) != 0
        || input_extractor.extract("patched_shape", patched_shape) != 0)
    {
        last_error_ = "DiT input projection failed";
        return static_cast<int>(Status::InferenceFailed);
    }

    if (resident_ && blocks_.size() == 32)
    {
        for (int index = 0; index < 32; ++index)
        {
            ncnn::Mat next_video;
            ncnn::Mat next_text;
            ncnn::Extractor extractor = blocks_[static_cast<std::size_t>(index)]->create_extractor();
            if (extractor.input("vid", video_tokens) != 0 || extractor.input("txt", text_tokens) != 0
                || extractor.input("emb", embedding) != 0 || extractor.input("vid_shape", patched_shape) != 0
                || extractor.extract("vid_out", next_video) != 0
                || extractor.extract("txt_out", next_text) != 0)
            {
                last_error_ = "DiT block inference failed at index " + std::to_string(index);
                return static_cast<int>(Status::InferenceFailed);
            }
            video_tokens = next_video;
            text_tokens = next_text;
        }
    }
    else
    {
        for (int index = 0; index < 32; ++index)
        {
            ncnn::Mat next_video;
            ncnn::Mat next_text;
            const int result = run_block(index,
                                         video_tokens,
                                         text_tokens,
                                         embedding,
                                         patched_shape,
                                         next_video,
                                         next_text);
            if (result != 0)
                return result;
            video_tokens = next_video;
            text_tokens = next_text;
        }
    }

    ncnn::Extractor output_extractor = output_->create_extractor();
    ncnn::Mat flat_output;
    ncnn::Mat output_shape;
    if (output_extractor.input("vid", video_tokens) != 0
        || output_extractor.input("emb", embedding) != 0
        || output_extractor.input("vid_shape", patched_shape) != 0
        || output_extractor.extract("vid_out", flat_output) != 0
        || output_extractor.extract("output_shape", output_shape) != 0
        || flat_output.dims != 2 || flat_output.w != 16
        || flat_output.h != latent.d * latent.h * latent.w)
    {
        last_error_ = "DiT output projection returned an invalid shape";
        return static_cast<int>(Status::InferenceFailed);
    }
    output = unflatten_latent(flat_output, latent.d, latent.h, latent.w);
    if (output.empty())
        return static_cast<int>(Status::OutOfMemory);
    return static_cast<int>(Status::Ok);
}

#if NCNN_VULKAN
int SeedVR2DiT::forward_vulkan(const ncnn::Mat& latent,
                               const ncnn::Mat& text,
                               float timestep,
                               ncnn::Mat& output)
{
    last_vulkan_transfer_stats_ = {};
    if (!input_ || !output_ || !context_)
    {
        last_error_ = "DiT is not loaded";
        return static_cast<int>(Status::NotLoaded);
    }
    if (context_->options().device != DeviceType::Vulkan)
    {
        last_error_ = "DiT Vulkan forward requires a Vulkan RuntimeContext";
        return static_cast<int>(Status::UnsupportedBackend);
    }
    if (latent.dims != 4 || latent.c != 33 || latent.elemsize != 4u || latent.elempack != 1
        || text.dims != 2 || text.w != 5120 || text.elemsize != 4u || text.elempack != 1
        || latent.h % 2 != 0 || latent.w % 2 != 0)
    {
        last_error_ = "DiT Vulkan expects latent [33,T,H,W] with even H/W and text [tokens,5120]";
        return static_cast<int>(Status::InvalidArgument);
    }

    ncnn::Mat flat = flatten_latent(latent);
    ncnn::Mat shape(3, static_cast<size_t>(4u), 1);
    ncnn::Mat timestep_mat(1, static_cast<size_t>(4u), 1);
    if (flat.empty() || shape.empty() || timestep_mat.empty())
        return static_cast<int>(Status::OutOfMemory);
    int* shape_data = shape;
    shape_data[0] = latent.d;
    shape_data[1] = latent.h;
    shape_data[2] = latent.w;
    static_cast<float*>(timestep_mat.data)[0] = timestep;

    VulkanExecutionScope execution(context_->options().vulkan_device_index);
    if (!execution.valid())
    {
        last_error_ = "failed to acquire Vulkan allocators for DiT";
        return static_cast<int>(Status::OutOfMemory);
    }
    ncnn::Option option = execution.option(context_->ncnn_option());
    ncnn::VkCompute command(execution.device());

    // 四个入口张量集中记录到同一命令流。后续所有子网复用这些 allocator，避免
    // extractor 析构时回收中间 VkMat 所属的临时 allocator。
    ncnn::VkMat flat_gpu;
    ncnn::VkMat text_gpu;
    ncnn::VkMat timestep_gpu;
    ncnn::VkMat shape_gpu;
    command.record_upload(flat, flat_gpu, option);
    command.record_upload(text, text_gpu, option);
    command.record_upload(timestep_mat, timestep_gpu, option);
    command.record_upload(shape, shape_gpu, option);
    last_vulkan_transfer_stats_.entry_upload_commands = 1;

    SeedVR2DiTInput* input_layer = find_custom_layer<SeedVR2DiTInput>(*input_);
    if (!input_layer)
    {
        last_error_ = "DiT input custom layer is missing";
        return static_cast<int>(Status::ModelLoadFailed);
    }
    input_layer->set_runtime_metadata(latent.d, latent.h, latent.w, timestep);

    ncnn::Extractor input_extractor = input_->create_extractor();
    execution.configure(input_extractor);
    ncnn::VkMat video_tokens;
    ncnn::VkMat text_tokens;
    ncnn::VkMat embedding;
    ncnn::VkMat patched_shape;
    if (input_extractor.input("vid", flat_gpu) != 0
        || input_extractor.input("txt", text_gpu) != 0
        || input_extractor.input("timestep", timestep_gpu) != 0
        || input_extractor.input("vid_shape", shape_gpu) != 0
        || input_extractor.extract("vid_out", video_tokens, command) != 0
        || input_extractor.extract("txt_out", text_tokens, command) != 0
        || input_extractor.extract("emb", embedding, command) != 0
        || input_extractor.extract("patched_shape", patched_shape, command) != 0)
    {
        last_error_ = "DiT Vulkan input projection failed";
        return static_cast<int>(Status::InferenceFailed);
    }

    const int patched_height = latent.h / 2;
    const int patched_width = latent.w / 2;
    if (resident_ && blocks_.size() == 32)
    {
        for (int index = 0; index < 32; ++index)
        {
            ncnn::Net& net = *blocks_[static_cast<std::size_t>(index)];
            SeedVR2DiTBlock* layer = find_custom_layer<SeedVR2DiTBlock>(net);
            if (!layer)
            {
                last_error_ = "DiT block custom layer is missing at index " + std::to_string(index);
                return static_cast<int>(Status::ModelLoadFailed);
            }
            layer->set_runtime_shape(latent.d, patched_height, patched_width);

            ncnn::Extractor extractor = net.create_extractor();
            execution.configure(extractor);
            ncnn::VkMat next_video;
            ncnn::VkMat next_text;
            if (extractor.input("vid", video_tokens) != 0
                || extractor.input("txt", text_tokens) != 0
                || extractor.input("emb", embedding) != 0
                || extractor.input("vid_shape", patched_shape) != 0
                || extractor.extract("vid_out", next_video, command) != 0
                || extractor.extract("txt_out", next_text, command) != 0)
            {
                last_error_ = "DiT Vulkan block inference failed at index " + std::to_string(index);
                return static_cast<int>(Status::InferenceFailed);
            }

            // 当前 block 内部会创建大量短生命周期 workspace。跨 32 个 Net 一次性
            // 延迟提交会使 allocator 过早复用这些区域，因此 resident 也在 block
            // 边界完成 GPU 命令。video/text 仍是 VkMat，不发生主机数据传输。
            if (command.submit_and_wait() != 0 || command.reset() != 0)
            {
                last_error_ = "DiT resident Vulkan submission failed at index "
                    + std::to_string(index);
                return static_cast<int>(Status::InferenceFailed);
            }
            ++last_vulkan_transfer_stats_.queue_submissions;
            video_tokens = next_video;
            text_tokens = next_text;
        }
    }
    else
    {
        for (int index = 0; index < 32; ++index)
        {
            ncnn::VkMat next_video;
            ncnn::VkMat next_text;
            const int result = run_block_vulkan(index,
                                                video_tokens,
                                                text_tokens,
                                                embedding,
                                                patched_shape,
                                                latent.d,
                                                patched_height,
                                                patched_width,
                                                execution.blob_allocator(),
                                                execution.staging_allocator(),
                                                command,
                                                next_video,
                                                next_text);
            if (result != 0)
                return result;
            video_tokens = next_video;
            text_tokens = next_text;
        }
    }

    SeedVR2DiTOutput* output_layer = find_custom_layer<SeedVR2DiTOutput>(*output_);
    if (!output_layer)
    {
        last_error_ = "DiT output custom layer is missing";
        return static_cast<int>(Status::ModelLoadFailed);
    }
    output_layer->set_runtime_shape(latent.d, patched_height, patched_width);

    ncnn::Extractor output_extractor = output_->create_extractor();
    execution.configure(output_extractor);
    ncnn::VkMat flat_output_gpu;
    if (output_extractor.input("vid", video_tokens) != 0
        || output_extractor.input("emb", embedding) != 0
        || output_extractor.input("vid_shape", patched_shape) != 0
        || output_extractor.extract("vid_out", flat_output_gpu, command) != 0)
    {
        last_error_ = "DiT Vulkan output projection failed";
        return static_cast<int>(Status::InferenceFailed);
    }

    ncnn::Mat flat_output;
    command.record_download(flat_output_gpu, flat_output, option);
    last_vulkan_transfer_stats_.final_download_commands = 1;
    if (command.submit_and_wait() != 0)
    {
        last_error_ = "DiT Vulkan final submission failed";
        return static_cast<int>(Status::InferenceFailed);
    }
    ++last_vulkan_transfer_stats_.queue_submissions;

    if (flat_output.dims != 2 || flat_output.w != 16
        || flat_output.h != latent.d * latent.h * latent.w)
    {
        last_error_ = "DiT Vulkan output projection returned an invalid shape";
        return static_cast<int>(Status::InferenceFailed);
    }
    output = unflatten_latent(flat_output, latent.d, latent.h, latent.w);
    if (output.empty())
        return static_cast<int>(Status::OutOfMemory);
    last_error_.clear();
    return static_cast<int>(Status::Ok);
}
#endif
} // namespace seedvr2
