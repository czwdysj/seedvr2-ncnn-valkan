// 本文件实现动态 VAE 的模型加载、posterior 采样与缩放。
// Encoder NCNN 图输出 [32,T,H,W] moments，前 16 通道为 mean、后 16 通道为
// logvar；Decoder 输入为逆 scaling 后的 [16,T,H,W]。三种动态层仍复用已验证实现。
#include "vae.h"

#include <algorithm>
#include <cmath>
#include <filesystem>

#include "dynamic_framewise_group_norm.h"
#include "dynamic_framewise_spatial_attention.h"
#include "dynamic_space_time_shuffle.h"
#include "seedvr2/engine.h"

namespace seedvr2
{
namespace
{
void register_vae_layers(ncnn::Net& net)
{
    net.register_custom_layer("DynamicFramewiseGroupNorm",
                              DynamicFramewiseGroupNorm_layer_creator);
    net.register_custom_layer("DynamicFramewiseSpatialAttention",
                              DynamicFramewiseSpatialAttention_layer_creator);
    net.register_custom_layer("DynamicSpaceTimeShuffle",
                              DynamicSpaceTimeShuffle_layer_creator);
}

int load_net(const std::filesystem::path& param,
             const std::filesystem::path& bin,
             const RuntimeContext& context,
             std::unique_ptr<ncnn::Net>& output,
             std::string& error)
{
    if (!std::filesystem::is_regular_file(param) || !std::filesystem::is_regular_file(bin))
    {
        error = "VAE model file is missing: " + param.string();
        return static_cast<int>(Status::ModelNotFound);
    }
    auto net = std::make_unique<ncnn::Net>();
    context.configure(*net);
    register_vae_layers(*net);
    const int param_result = net->load_param(param.string().c_str());
    const int model_result = param_result == 0 ? net->load_model(bin.string().c_str()) : param_result;
    if (param_result != 0 || model_result != 0)
    {
        error = "failed to load VAE model: " + param.string();
        return static_cast<int>(Status::ModelLoadFailed);
    }
    output = std::move(net);
    return static_cast<int>(Status::Ok);
}
} // namespace

SeedVR2VAE::SeedVR2VAE() = default;
SeedVR2VAE::~SeedVR2VAE() = default;

int SeedVR2VAE::load(const std::string& model_dir, const RuntimeContext& context)
{
    const std::filesystem::path root(model_dir);
    const int encoder_result = load_net(root / "seedvr2_vae_encoder_dynamic.ncnn.param",
                                        root / "seedvr2_vae_encoder_dynamic.ncnn.bin",
                                        context,
                                        encoder_,
                                        last_error_);
    if (encoder_result != 0)
        return encoder_result;
    const int decoder_result = load_net(root / "seedvr2_vae_decoder_dynamic.ncnn.param",
                                        root / "seedvr2_vae_decoder_dynamic.ncnn.bin",
                                        context,
                                        decoder_,
                                        last_error_);
    if (decoder_result != 0)
    {
        encoder_.reset();
        return decoder_result;
    }
    last_error_.clear();
    return static_cast<int>(Status::Ok);
}

int SeedVR2VAE::encode(const ncnn::Mat& video,
                       std::mt19937_64& random,
                       bool stochastic,
                       float scaling_factor,
                       ncnn::Mat& latent)
{
    if (!encoder_)
    {
        last_error_ = "VAE encoder is not loaded";
        return static_cast<int>(Status::NotLoaded);
    }
    if (video.dims != 4 || video.c != 3 || video.elemsize != 4u || video.elempack != 1)
    {
        last_error_ = "VAE encoder expects FP32 C=3,T,H,W input";
        return static_cast<int>(Status::InvalidArgument);
    }

    ncnn::Extractor extractor = encoder_->create_extractor();
    if (extractor.input(encoder_->input_names()[0], video) != 0)
    {
        last_error_ = "failed to bind VAE encoder input";
        return static_cast<int>(Status::InferenceFailed);
    }
    ncnn::Mat moments;
    if (extractor.extract(encoder_->output_names()[0], moments) != 0 || moments.dims != 4
        || moments.c != 32 || moments.elemsize != 4u || moments.elempack != 1)
    {
        last_error_ = "VAE encoder did not return [32,T,H,W] FP32 moments";
        return static_cast<int>(Status::InferenceFailed);
    }

    latent.create(moments.w, moments.h, moments.d, 16, 4u, 1);
    if (latent.empty())
        return static_cast<int>(Status::OutOfMemory);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    for (int channel = 0; channel < 16; ++channel)
        for (int frame = 0; frame < moments.d; ++frame)
            for (int y = 0; y < moments.h; ++y)
            {
                const float* mean = moments.channel(channel).depth(frame).row(y);
                const float* logvar = moments.channel(channel + 16).depth(frame).row(y);
                float* destination = latent.channel(channel).depth(frame).row(y);
                for (int x = 0; x < moments.w; ++x)
                {
                    float value = mean[x];
                    if (stochastic)
                    {
                        const float clamped = std::clamp(logvar[x], -30.0f, 20.0f);
                        value += std::exp(0.5f * clamped) * normal(random);
                    }
                    destination[x] = value * scaling_factor;
                }
            }
    return static_cast<int>(Status::Ok);
}

int SeedVR2VAE::decode(const ncnn::Mat& latent, float scaling_factor, ncnn::Mat& video)
{
    if (!decoder_)
    {
        last_error_ = "VAE decoder is not loaded";
        return static_cast<int>(Status::NotLoaded);
    }
    if (latent.dims != 4 || latent.c != 16 || latent.elemsize != 4u || latent.elempack != 1)
    {
        last_error_ = "VAE decoder expects FP32 C=16,T,H,W latent";
        return static_cast<int>(Status::InvalidArgument);
    }

    ncnn::Mat unscaled(latent.w, latent.h, latent.d, latent.c, 4u, 1);
    if (unscaled.empty())
        return static_cast<int>(Status::OutOfMemory);
    for (int channel = 0; channel < latent.c; ++channel)
        for (int frame = 0; frame < latent.d; ++frame)
            for (int y = 0; y < latent.h; ++y)
            {
                const float* source = latent.channel(channel).depth(frame).row(y);
                float* destination = unscaled.channel(channel).depth(frame).row(y);
                for (int x = 0; x < latent.w; ++x)
                    destination[x] = source[x] / scaling_factor;
            }

    ncnn::Extractor extractor = decoder_->create_extractor();
    if (extractor.input(decoder_->input_names()[0], unscaled) != 0
        || extractor.extract(decoder_->output_names()[0], video) != 0)
    {
        last_error_ = "VAE decoder inference failed";
        return static_cast<int>(Status::InferenceFailed);
    }
    return static_cast<int>(Status::Ok);
}
} // namespace seedvr2
