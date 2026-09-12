// 本文件实现动态 VAE 的模型加载、posterior 采样与缩放。
// Encoder NCNN 图输出 [32,T,H,W] moments，前 16 通道为 mean、后 16 通道为
// logvar；Decoder 输入为逆 scaling 后的 [16,T,H,W]。三种动态层仍复用已验证实现。
#include "model/vae.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <vector>

#if NCNN_VULKAN
#include <gpu.h>
#endif

#include "layers/dynamic_framewise_group_norm.h"
#include "layers/dynamic_framewise_spatial_attention.h"
#include "layers/dynamic_space_time_shuffle.h"
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

#if NCNN_VULKAN
static const char* posterior_shader = R"VKGLSL(
#version 450
layout(binding=0, std430) readonly buffer moments_block { sfp moments[]; };
layout(binding=1, std430) readonly buffer noise_block { sfp noise[]; };
layout(binding=2, std430) writeonly buffer output_block { sfp output_data[]; };
layout(push_constant) uniform parameter {
    uint tokens; uint cstep; uint total; uint stochastic; float scaling;
} p;
void main() {
    uint i=gl_GlobalInvocationID.x; if(i>=p.total) return;
    uint token=i/16u; uint channel=i-token*16u;
    afp value=buffer_ld1(moments,channel*p.cstep+token);
    if(p.stochastic!=0u) {
        afp logvar=clamp(buffer_ld1(moments,(channel+16u)*p.cstep+token),-30.0,20.0);
        value += exp(0.5*logvar)*buffer_ld1(noise,i);
    }
    buffer_st1(output_data,i,value*p.scaling);
}
)VKGLSL";

static const char* decode_layout_shader = R"VKGLSL(
#version 450
layout(binding=0, std430) readonly buffer latent_block { sfp latent[]; };
layout(binding=1, std430) writeonly buffer output_block { sfp output_data[]; };
layout(push_constant) uniform parameter { uint tokens; uint cstep; uint total; float inverse_scaling; } p;
void main() {
    uint i=gl_GlobalInvocationID.x; if(i>=p.total) return;
    uint channel=i/p.tokens; uint token=i-channel*p.tokens;
    buffer_st1(output_data,channel*p.cstep+token,
               buffer_ld1(latent,token*16u+channel)*p.inverse_scaling);
}
)VKGLSL";

int compile_vae_pipeline(const ncnn::VulkanDevice* device,
                         const ncnn::Option& option,
                         const char* source,
                         ncnn::Pipeline*& pipeline)
{
    std::vector<uint32_t> spirv;
    const int result=ncnn::compile_spirv_module(source,option,spirv);
    if(result!=0) return result;
    pipeline=new ncnn::Pipeline(device);
    pipeline->set_optimal_local_size_xyz(256,1,1);
    const std::vector<ncnn::vk_specialization_type> specializations;
    return pipeline->create(spirv.data(),spirv.size()*sizeof(uint32_t),specializations);
}
#endif
} // namespace

SeedVR2VAE::SeedVR2VAE() = default;
SeedVR2VAE::~SeedVR2VAE()
{
#if NCNN_VULKAN
    delete pipeline_posterior_;
    delete pipeline_decode_layout_;
#endif
}

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
#if NCNN_VULKAN
    if (context.options().device == DeviceType::Vulkan)
    {
        const ncnn::VulkanDevice* device =
            ncnn::get_gpu_device(context.options().vulkan_device_index);
        if (!device
            || compile_vae_pipeline(device, context.ncnn_option(), posterior_shader,
                                    pipeline_posterior_) != 0
            || compile_vae_pipeline(device, context.ncnn_option(), decode_layout_shader,
                                    pipeline_decode_layout_) != 0)
        {
            last_error_ = "failed to compile VAE boundary Vulkan pipelines";
            return static_cast<int>(Status::ModelLoadFailed);
        }
    }
#endif
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
        char detail[128];
        if (moments.empty())
            std::snprintf(detail, sizeof(detail), "got empty output");
        else
            std::snprintf(detail, sizeof(detail),
                          "got [c=%d d=%d h=%d w=%d] dims=%d elemsize=%zu elempack=%d",
                          moments.c, moments.d, moments.h, moments.w, moments.dims,
                          moments.elemsize, moments.elempack);
        last_error_ = std::string("VAE encoder did not return [32,T,H,W] FP32 moments: ")
            + detail;
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

#if NCNN_VULKAN
int SeedVR2VAE::encode_vkmat(const ncnn::VkMat& video,
                             const ncnn::VkMat& posterior_noise,
                             bool stochastic,
                             float scaling_factor,
                             VulkanExecutionContext& execution,
                             ncnn::VkMat& latent)
{
    if (!encoder_ || !pipeline_posterior_)
    {
        last_error_="VAE Vulkan encoder is not loaded";
        return static_cast<int>(Status::NotLoaded);
    }
    if(video.dims!=4 || video.c*video.elempack!=3 || posterior_noise.dims!=2
        || posterior_noise.w!=16 || posterior_noise.elempack!=1)
    {
        last_error_="VAE Vulkan encoder expects video C=3,T,H,W and pack1 posterior noise";
        return static_cast<int>(Status::InvalidArgument);
    }

    ncnn::VkCompute command(execution.device());
    ncnn::Extractor extractor=encoder_->create_extractor();
    execution.configure(extractor);
    ncnn::VkMat moments_packed;
    if(extractor.input(encoder_->input_names()[0],video)!=0
        || extractor.extract(encoder_->output_names()[0],moments_packed,command)!=0)
    {
        last_error_="VAE Vulkan encoder inference failed";
        return static_cast<int>(Status::InferenceFailed);
    }
    ncnn::VkMat moments;
    execution.device()->convert_packing(moments_packed,moments,1,command,execution.option());
    const int tokens=posterior_noise.h;
    latent.create(16,tokens,posterior_noise.elemsize,1,execution.blob_allocator());
    if(latent.empty()) return static_cast<int>(Status::OutOfMemory);
    const uint32_t total=static_cast<uint32_t>(tokens*16);
    std::vector<ncnn::VkMat> bindings={moments,posterior_noise,latent};
    std::vector<ncnn::vk_constant_type> constants(5);
    constants[0].u32=tokens; constants[1].u32=static_cast<uint32_t>(moments.cstep);
    constants[2].u32=total; constants[3].u32=stochastic?1u:0u; constants[4].f=scaling_factor;
    ncnn::VkMat dispatcher; dispatcher.w=total; dispatcher.h=1; dispatcher.c=1;
    command.record_pipeline(pipeline_posterior_,bindings,constants,dispatcher);
    if(command.submit_and_wait()!=0)
    {
        last_error_="VAE Vulkan posterior dispatch failed";
        return static_cast<int>(Status::InferenceFailed);
    }
    last_error_.clear();
    return static_cast<int>(Status::Ok);
}

int SeedVR2VAE::decode_vkmat(const ncnn::VkMat& latent,
                             int frames,
                             int height,
                             int width,
                             float scaling_factor,
                             VulkanExecutionContext& execution,
                             ncnn::VkMat& video)
{
    if(!decoder_ || !pipeline_decode_layout_)
    {
        last_error_="VAE Vulkan decoder is not loaded";
        return static_cast<int>(Status::NotLoaded);
    }
    const int tokens=frames*height*width;
    if(latent.dims!=2 || latent.w!=16 || latent.h!=tokens || latent.elempack!=1
        || scaling_factor<=0.0f)
    {
        last_error_="VAE Vulkan decoder expects pack1 token-major [T*H*W,16] latent";
        return static_cast<int>(Status::InvalidArgument);
    }

    ncnn::VkCompute command(execution.device());
    ncnn::VkMat unscaled;
    unscaled.create(width,height,frames,16,latent.elemsize,1,execution.blob_allocator());
    if(unscaled.empty()) return static_cast<int>(Status::OutOfMemory);
    const uint32_t total=static_cast<uint32_t>(tokens*16);
    std::vector<ncnn::VkMat> bindings={latent,unscaled};
    std::vector<ncnn::vk_constant_type> constants(4);
    constants[0].u32=tokens; constants[1].u32=static_cast<uint32_t>(unscaled.cstep);
    constants[2].u32=total; constants[3].f=1.0f/scaling_factor;
    ncnn::VkMat dispatcher; dispatcher.w=total; dispatcher.h=1; dispatcher.c=1;
    command.record_pipeline(pipeline_decode_layout_,bindings,constants,dispatcher);

    ncnn::Extractor extractor=decoder_->create_extractor();
    execution.configure(extractor);
    if(extractor.input(decoder_->input_names()[0],unscaled)!=0
        || extractor.extract(decoder_->output_names()[0],video,command)!=0
        || command.submit_and_wait()!=0)
    {
        last_error_="VAE Vulkan decoder inference failed";
        return static_cast<int>(Status::InferenceFailed);
    }
    last_error_.clear();
    return static_cast<int>(Status::Ok);
}
#endif
} // namespace seedvr2
