// 本文件实现与 SeedVR2 PyTorch 基准一致的 CFG 和 Euler/v_lerp 更新。
// lerp schedule 满足 A(t)=1-t/T、B(t)=t/T；最终 endpoint 直接得到
// x0=x_t-B(t)*prediction。全部计算使用 FP32，便于先完成 CPU 数值基线。
#include "model/sampler.h"

#include <algorithm>
#include <cmath>

#if NCNN_VULKAN
#include <gpu.h>
#endif

#include "seedvr2/engine.h"

namespace seedvr2
{
namespace
{
bool same_shape(const ncnn::Mat& a, const ncnn::Mat& b)
{
    return a.dims == 4 && b.dims == 4 && a.w == b.w && a.h == b.h && a.d == b.d
        && a.c == b.c && a.elemsize == 4u && b.elemsize == 4u && a.elempack == 1
        && b.elempack == 1;
}

template <typename Function>
void transform_4d(const ncnn::Mat& a, const ncnn::Mat& b, ncnn::Mat& output, Function function)
{
    for (int channel = 0; channel < a.c; ++channel)
        for (int frame = 0; frame < a.d; ++frame)
            for (int y = 0; y < a.h; ++y)
            {
                const float* pa = a.channel(channel).depth(frame).row(y);
                const float* pb = b.channel(channel).depth(frame).row(y);
                float* po = output.channel(channel).depth(frame).row(y);
                for (int x = 0; x < a.w; ++x)
                    po[x] = function(pa[x], pb[x]);
            }
}

double standard_deviation(const ncnn::Mat& value)
{
    const std::size_t count =
        static_cast<std::size_t>(value.w) * value.h * value.d * value.c;
    double sum = 0.0;
    double square_sum = 0.0;
    for (int channel = 0; channel < value.c; ++channel)
        for (int frame = 0; frame < value.d; ++frame)
            for (int y = 0; y < value.h; ++y)
            {
                const float* row = value.channel(channel).depth(frame).row(y);
                for (int x = 0; x < value.w; ++x)
                {
                    sum += row[x];
                    square_sum += static_cast<double>(row[x]) * row[x];
                }
            }
    if (count <= 1)
        return 0.0;
    const double variance = std::max(0.0, (square_sum - sum * sum / count) / (count - 1));
    return std::sqrt(variance);
}

#if NCNN_VULKAN
bool same_vk_shape(const ncnn::VkMat& a, const ncnn::VkMat& b)
{
    return a.dims == 2 && b.dims == 2 && a.w == b.w && a.h == b.h
        && a.elempack == 1 && b.elempack == 1;
}

static const char* condition_shader = R"VKGLSL(
#version 450
layout(binding=0, std430) readonly buffer a_data { sfp a[]; };
layout(binding=1, std430) readonly buffer b_data { sfp b[]; };
layout(binding=2, std430) writeonly buffer output_block { sfp output_data[]; };
layout(push_constant) uniform parameter { uint total; float ratio; } p;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i < p.total) buffer_st1(output_data, i, (1.0-p.ratio)*buffer_ld1(a,i)+p.ratio*buffer_ld1(b,i));
}
)VKGLSL";

static const char* dit_input_shader = R"VKGLSL(
#version 450
layout(binding=0, std430) readonly buffer latent_data { sfp latent[]; };
layout(binding=1, std430) readonly buffer condition_data { sfp condition[]; };
layout(binding=2, std430) writeonly buffer output_block { sfp output_data[]; };
layout(push_constant) uniform parameter { uint tokens; uint total; } p;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= p.total) return;
    uint token = i / 33u;
    uint channel = i - token*33u;
    afp value = channel < 16u ? buffer_ld1(latent, token*16u+channel)
              : channel < 32u ? buffer_ld1(condition, token*16u+channel-16u) : 1.0;
    buffer_st1(output_data, i, value);
}
)VKGLSL";

static const char* cfg_shader = R"VKGLSL(
#version 450
layout(binding=0, std430) readonly buffer positive_data { sfp positive[]; };
layout(binding=1, std430) readonly buffer negative_data { sfp negative[]; };
layout(binding=2, std430) writeonly buffer output_block { sfp output_data[]; };
layout(push_constant) uniform parameter { uint total; float scale; } p;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i < p.total) {
        afp pos = buffer_ld1(positive,i); afp neg = buffer_ld1(negative,i);
        buffer_st1(output_data,i,neg+p.scale*(pos-neg));
    }
}
)VKGLSL";

static const char* cfg_stats_shader = R"VKGLSL(
#version 450
layout(binding=0, std430) readonly buffer positive_data { sfp positive[]; };
layout(binding=1, std430) readonly buffer cfg_data { sfp cfg[]; };
layout(binding=2, std430) writeonly buffer stats_data { float stats[]; };
layout(push_constant) uniform parameter { uint total; } p;
shared float s0[256]; shared float s1[256]; shared float s2[256]; shared float s3[256];
void main() {
    uint tid=gl_LocalInvocationID.x; uint n=gl_WorkGroupSize.x;
    float a=0.0,b=0.0,c=0.0,d=0.0;
    for(uint i=tid;i<p.total;i+=n){ float x=buffer_ld1(positive,i); float y=buffer_ld1(cfg,i); a+=x;b+=x*x;c+=y;d+=y*y; }
    s0[tid]=a;s1[tid]=b;s2[tid]=c;s3[tid]=d; barrier();
    for(uint step=n/2u;step>0u;step>>=1u){ if(tid<step){s0[tid]+=s0[tid+step];s1[tid]+=s1[tid+step];s2[tid]+=s2[tid+step];s3[tid]+=s3[tid+step];} barrier(); }
    if(tid==0u){stats[0]=s0[0];stats[1]=s1[0];stats[2]=s2[0];stats[3]=s3[0];}
}
)VKGLSL";

static const char* cfg_rescale_shader = R"VKGLSL(
#version 450
layout(binding=0, std430) buffer cfg_data { sfp cfg[]; };
layout(binding=1, std430) readonly buffer stats_data { float stats[]; };
layout(push_constant) uniform parameter { uint total; float rescale; } p;
void main() {
    uint i=gl_GlobalInvocationID.x; if(i>=p.total) return;
    float n=float(p.total); float denom=max(n-1.0,1.0);
    float var_pos=max((stats[1]-stats[0]*stats[0]/n)/denom,0.0);
    float var_cfg=max((stats[3]-stats[2]*stats[2]/n)/denom,0.0);
    float factor=var_cfg>0.0 ? p.rescale*sqrt(var_pos/var_cfg)+(1.0-p.rescale) : 1.0;
    buffer_st1(cfg,i,buffer_ld1(cfg,i)*factor);
}
)VKGLSL";

static const char* euler_shader = R"VKGLSL(
#version 450
layout(binding=0, std430) readonly buffer prediction_data { sfp prediction[]; };
layout(binding=1, std430) readonly buffer current_data { sfp current[]; };
layout(binding=2, std430) writeonly buffer output_block { sfp output_data[]; };
layout(push_constant) uniform parameter { uint total; float delta; } p;
void main(){ uint i=gl_GlobalInvocationID.x; if(i<p.total) buffer_st1(output_data,i,buffer_ld1(current,i)+p.delta*buffer_ld1(prediction,i)); }
)VKGLSL";

int compile_pipeline(const ncnn::VulkanDevice* device,
                     const ncnn::Option& option,
                     const char* source,
                     int local_size,
                     ncnn::Pipeline*& pipeline)
{
    std::vector<uint32_t> spirv;
    const int result = ncnn::compile_spirv_module(source, option, spirv);
    if (result != 0)
        return result;
    pipeline = new ncnn::Pipeline(device);
    pipeline->set_optimal_local_size_xyz(local_size, 1, 1);
    const std::vector<ncnn::vk_specialization_type> specializations;
    return pipeline->create(spirv.data(), spirv.size()*sizeof(uint32_t), specializations);
}

void set_dispatcher(ncnn::VkMat& dispatcher, uint32_t total)
{
    dispatcher.w = static_cast<int>(total);
    dispatcher.h = 1;
    dispatcher.c = 1;
}
#endif
} // namespace

EulerSampler::EulerSampler(int steps)
{
    steps = std::max(1, steps);
    timesteps_.reserve(steps);
    for (int index = 0; index < steps; ++index)
        timesteps_.push_back(1000.0f * (1.0f - static_cast<float>(index) / steps));
}

EulerSampler::~EulerSampler()
{
#if NCNN_VULKAN
    delete pipeline_condition_;
    delete pipeline_dit_input_;
    delete pipeline_cfg_;
    delete pipeline_cfg_stats_;
    delete pipeline_cfg_rescale_;
    delete pipeline_euler_;
#endif
}

int EulerSampler::apply_cfg(const ncnn::Mat& positive,
                            const ncnn::Mat& negative,
                            float scale,
                            float rescale,
                            ncnn::Mat& output,
                            std::string& error) const
{
    if (!same_shape(positive, negative))
    {
        error = "CFG inputs must have identical FP32 C,T,H,W shapes";
        return static_cast<int>(Status::InvalidArgument);
    }
    output.create(positive.w, positive.h, positive.d, positive.c, 4u, 1);
    if (output.empty())
        return static_cast<int>(Status::OutOfMemory);

    transform_4d(positive, negative, output,
                 [scale](float pos, float neg) { return neg + scale * (pos - neg); });
    if (rescale != 0.0f)
    {
        const double cfg_std = standard_deviation(output);
        if (cfg_std > 0.0)
        {
            const float factor = static_cast<float>(rescale * standard_deviation(positive) / cfg_std
                                                    + (1.0f - rescale));
            for (int channel = 0; channel < output.c; ++channel)
                for (int frame = 0; frame < output.d; ++frame)
                    for (int y = 0; y < output.h; ++y)
                    {
                        float* row = output.channel(channel).depth(frame).row(y);
                        for (int x = 0; x < output.w; ++x)
                            row[x] *= factor;
                    }
        }
    }
    return static_cast<int>(Status::Ok);
}

int EulerSampler::step_to(const ncnn::Mat& prediction,
                          const ncnn::Mat& current,
                          float timestep,
                          float next_timestep,
                          ncnn::Mat& output,
                          std::string& error) const
{
    if (!same_shape(prediction, current))
    {
        error = "Euler prediction and current latent shapes differ";
        return static_cast<int>(Status::InvalidArgument);
    }
    output.create(current.w, current.h, current.d, current.c, 4u, 1);
    if (output.empty())
        return static_cast<int>(Status::OutOfMemory);

    const float bt = timestep / 1000.0f;
    const float at = 1.0f - bt;
    const float bs = std::clamp(next_timestep / 1000.0f, 0.0f, 1.0f);
    const float as = 1.0f - bs;
    transform_4d(current, prediction, output, [=](float x, float pred) {
        const float x0 = x - bt * pred;
        const float xT = x + at * pred;
        return as * x0 + bs * xT;
    });
    return static_cast<int>(Status::Ok);
}

int EulerSampler::endpoint(const ncnn::Mat& prediction,
                           const ncnn::Mat& current,
                           float timestep,
                           ncnn::Mat& output,
                           std::string& error) const
{
    if (!same_shape(prediction, current))
    {
        error = "Euler endpoint inputs have different shapes";
        return static_cast<int>(Status::InvalidArgument);
    }
    output.create(current.w, current.h, current.d, current.c, 4u, 1);
    if (output.empty())
        return static_cast<int>(Status::OutOfMemory);
    const float bt = timestep / 1000.0f;
    transform_4d(current, prediction, output,
                 [bt](float x, float pred) { return x - bt * pred; });
    return static_cast<int>(Status::Ok);
}

#if NCNN_VULKAN
int EulerSampler::initialize_vulkan(const RuntimeContext& context, std::string& error)
{
    const ncnn::VulkanDevice* device = ncnn::get_gpu_device(context.options().vulkan_device_index);
    if (!device)
    {
        error = "sampler Vulkan device is unavailable";
        return static_cast<int>(Status::UnsupportedBackend);
    }
    const ncnn::Option& option = context.ncnn_option();
    struct Entry { const char* source; int local_size; ncnn::Pipeline** output; };
    const Entry entries[] = {
        {condition_shader, 256, &pipeline_condition_}, {dit_input_shader, 256, &pipeline_dit_input_},
        {cfg_shader, 256, &pipeline_cfg_}, {cfg_stats_shader, 256, &pipeline_cfg_stats_},
        {cfg_rescale_shader, 256, &pipeline_cfg_rescale_}, {euler_shader, 256, &pipeline_euler_}};
    for (const Entry& entry : entries)
    {
        const int result = compile_pipeline(device, option, entry.source, entry.local_size, *entry.output);
        if (result != 0)
        {
            error = "failed to compile sampler Vulkan pipeline";
            return static_cast<int>(Status::ModelLoadFailed);
        }
    }
    return static_cast<int>(Status::Ok);
}

int EulerSampler::make_condition_vulkan(const ncnn::VkMat& encoded,
                                         const ncnn::VkMat& augment_noise,
                                         float ratio,
                                         VulkanExecutionContext& execution,
                                         ncnn::VkCompute& command,
                                         ncnn::VkMat& output,
                                         std::string& error) const
{
    if (!pipeline_condition_ || !same_vk_shape(encoded, augment_noise) || encoded.w != 16)
    {
        error = "Vulkan condition inputs must be pack1 [tokens,16] tensors";
        return static_cast<int>(Status::InvalidArgument);
    }
    output.create(encoded.w, encoded.h, encoded.elemsize, 1, execution.blob_allocator());
    if (output.empty()) return static_cast<int>(Status::OutOfMemory);
    const uint32_t total = static_cast<uint32_t>(encoded.w*encoded.h);
    std::vector<ncnn::VkMat> bindings = {encoded, augment_noise, output};
    std::vector<ncnn::vk_constant_type> constants(2); constants[0].u32=total; constants[1].f=ratio;
    ncnn::VkMat dispatcher; set_dispatcher(dispatcher,total);
    command.record_pipeline(pipeline_condition_,bindings,constants,dispatcher);
    return static_cast<int>(Status::Ok);
}

int EulerSampler::make_dit_input_vulkan(const ncnn::VkMat& latent,
                                         const ncnn::VkMat& condition,
                                         VulkanExecutionContext& execution,
                                         ncnn::VkCompute& command,
                                         ncnn::VkMat& output,
                                         std::string& error) const
{
    if (!pipeline_dit_input_ || !same_vk_shape(latent,condition) || latent.w!=16)
    {
        error="Vulkan DiT input sources must be pack1 [tokens,16] tensors";
        return static_cast<int>(Status::InvalidArgument);
    }
    output.create(33,latent.h,latent.elemsize,1,execution.blob_allocator());
    if(output.empty()) return static_cast<int>(Status::OutOfMemory);
    const uint32_t total=static_cast<uint32_t>(latent.h*33);
    std::vector<ncnn::VkMat> bindings={latent,condition,output};
    std::vector<ncnn::vk_constant_type> constants(2); constants[0].u32=latent.h; constants[1].u32=total;
    ncnn::VkMat dispatcher; set_dispatcher(dispatcher,total);
    command.record_pipeline(pipeline_dit_input_,bindings,constants,dispatcher);
    return static_cast<int>(Status::Ok);
}

int EulerSampler::apply_cfg_vulkan(const ncnn::VkMat& positive,
                                    const ncnn::VkMat& negative,
                                    float scale,
                                    float rescale,
                                    VulkanExecutionContext& execution,
                                    ncnn::VkCompute& command,
                                    ncnn::VkMat& output,
                                    std::string& error) const
{
    if (!pipeline_cfg_ || !same_vk_shape(positive,negative) || positive.w!=16)
    {
        error="Vulkan CFG inputs must be pack1 [tokens,16] tensors";
        return static_cast<int>(Status::InvalidArgument);
    }
    output.create(positive.w,positive.h,positive.elemsize,1,execution.blob_allocator());
    if(output.empty()) return static_cast<int>(Status::OutOfMemory);
    const uint32_t total=static_cast<uint32_t>(positive.w*positive.h);
    std::vector<ncnn::VkMat> bindings={positive,negative,output};
    std::vector<ncnn::vk_constant_type> constants(2); constants[0].u32=total; constants[1].f=scale;
    ncnn::VkMat dispatcher; set_dispatcher(dispatcher,total);
    command.record_pipeline(pipeline_cfg_,bindings,constants,dispatcher);
    if(rescale!=0.0f)
    {
        ncnn::VkMat stats; stats.create(4,4u,1,execution.blob_allocator());
        if(stats.empty()) return static_cast<int>(Status::OutOfMemory);
        bindings={positive,output,stats}; constants.resize(1); constants[0].u32=total;
        dispatcher.w=pipeline_cfg_stats_->local_size_x();
        command.record_pipeline(pipeline_cfg_stats_,bindings,constants,dispatcher);
        bindings={output,stats}; constants.resize(2); constants[0].u32=total; constants[1].f=rescale;
        set_dispatcher(dispatcher,total);
        command.record_pipeline(pipeline_cfg_rescale_,bindings,constants,dispatcher);
    }
    return static_cast<int>(Status::Ok);
}

int EulerSampler::step_vulkan(const ncnn::VkMat& prediction,
                               const ncnn::VkMat& current,
                               float timestep,
                               float next_timestep,
                               VulkanExecutionContext& execution,
                               ncnn::VkCompute& command,
                               ncnn::VkMat& output,
                               std::string& error) const
{
    if(!pipeline_euler_ || !same_vk_shape(prediction,current) || current.w!=16)
    {
        error="Vulkan Euler inputs must be pack1 [tokens,16] tensors";
        return static_cast<int>(Status::InvalidArgument);
    }
    output.create(current.w,current.h,current.elemsize,1,execution.blob_allocator());
    if(output.empty()) return static_cast<int>(Status::OutOfMemory);
    const uint32_t total=static_cast<uint32_t>(current.w*current.h);
    const float delta=std::clamp(next_timestep/1000.0f,0.0f,1.0f)-timestep/1000.0f;
    std::vector<ncnn::VkMat> bindings={prediction,current,output};
    std::vector<ncnn::vk_constant_type> constants(2); constants[0].u32=total; constants[1].f=delta;
    ncnn::VkMat dispatcher; set_dispatcher(dispatcher,total);
    command.record_pipeline(pipeline_euler_,bindings,constants,dispatcher);
    return static_cast<int>(Status::Ok);
}
#endif
} // namespace seedvr2
