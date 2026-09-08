// 本文件实现完整 DiT 的流式调度，是原 dit_full_runner 中真实运行逻辑的库化版本。
// 输入/输出头常驻，32 个大 block 逐层 load-forward-destroy；块间只传递 vid/txt
// token、时间 embedding 和动态 patch shape，因此峰值权重内存约为一个最大 block。
#include "model/dit.h"

#include <cstdio>
#include <filesystem>

#include "layers/seedvr2_dit_block.h"
#include "layers/seedvr2_dit_input.h"
#include "layers/seedvr2_dit_output.h"
#include "seedvr2/engine.h"

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

int SeedVR2DiT::forward(const ncnn::Mat& latent,
                        const ncnn::Mat& text,
                        float timestep,
                        ncnn::Mat& output)
{
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
} // namespace seedvr2
