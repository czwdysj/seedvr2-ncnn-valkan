// 本文件验证 VAE 与 Engine 之间新增加的 VkMat 边界。
// runner 使用同一视频和同一 posterior 高斯噪声，分别执行旧 Mat 边界与新
// VkMat 边界，独立报告 encoder latent 和 decoder video 的最大误差与 cosine。
#include "core/runtime_context.h"
#include "model/vae.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <string>

namespace
{
ncnn::Mat flatten(const ncnn::Mat& value)
{
    ncnn::Mat output(value.c, value.d*value.h*value.w, static_cast<size_t>(4u), 1);
    int token=0;
    for(int t=0;t<value.d;++t)
        for(int y=0;y<value.h;++y)
            for(int x=0;x<value.w;++x,++token)
                for(int c=0;c<value.c;++c)
                    output.row(token)[c]=value.channel(c).depth(t).row(y)[x];
    return output;
}

void metrics(const ncnn::Mat& a, const ncnn::Mat& b, double& max_abs, double& cosine)
{
    const float* pa=a; const float* pb=b;
    const size_t count=a.total()*a.elempack;
    double dot=0.0,aa=0.0,bb=0.0; max_abs=0.0;
    for(size_t i=0;i<count;++i)
    {
        max_abs=std::max(max_abs,std::abs(static_cast<double>(pa[i])-pb[i]));
        dot+=static_cast<double>(pa[i])*pb[i]; aa+=static_cast<double>(pa[i])*pa[i];
        bb+=static_cast<double>(pb[i])*pb[i];
    }
    cosine=dot/std::sqrt(aa*bb);
}
} // namespace

int main(int argc,char** argv)
{
    if(argc!=2)
    {
        std::cerr<<"usage: "<<argv[0]<<" <vae_model_dir>\n";
        return 2;
    }
    seedvr2::RuntimeOptions options; options.device=seedvr2::DeviceType::Vulkan;
    seedvr2::RuntimeContext runtime; std::string error;
    if(runtime.initialize(options,error)!=0){std::cerr<<error<<'\n';return 3;}
    seedvr2::SeedVR2VAE vae;
    if(vae.load(argv[1],runtime)!=0){std::cerr<<vae.last_error()<<'\n';return 4;}

    ncnn::Mat video(64,64,5,3,4u,1);
    for(size_t i=0;i<video.total();++i)
        static_cast<float*>(video.data)[i]=(static_cast<int>(i%29)-14)*0.01f;
    std::mt19937_64 cpu_random(666);
    ncnn::Mat reference_latent;
    if(vae.encode(video,cpu_random,true,0.9152f,reference_latent)!=0) return 5;

    std::mt19937_64 noise_random(666);
    std::normal_distribution<float> normal(0.0f,1.0f);
    ncnn::Mat noise(reference_latent.w,reference_latent.h,reference_latent.d,16,4u,1);
    for(int c=0;c<noise.c;++c) for(int t=0;t<noise.d;++t) for(int y=0;y<noise.h;++y)
    {
        float* row=noise.channel(c).depth(t).row(y);
        for(int x=0;x<noise.w;++x) row[x]=normal(noise_random);
    }
    ncnn::Mat noise_flat=flatten(noise);
    seedvr2::VulkanExecutionContext execution(runtime);
    ncnn::VkCompute upload(execution.device());
    ncnn::VkMat video_gpu,noise_packed,noise_gpu;
    upload.record_upload(video,video_gpu,execution.option());
    upload.record_upload(noise_flat,noise_packed,execution.option());
    execution.device()->convert_packing(noise_packed,noise_gpu,1,upload,execution.option());
    if(upload.submit_and_wait()!=0) return 6;

    ncnn::VkMat latent_gpu;
    if(vae.encode_vkmat(video_gpu,noise_gpu,true,0.9152f,execution,latent_gpu)!=0)
    {std::cerr<<vae.last_error()<<'\n';return 7;}
    ncnn::Mat latent_flat;
    ncnn::VkCompute latent_download(execution.device());
    latent_download.record_download(latent_gpu,latent_flat,execution.option());
    if(latent_download.submit_and_wait()!=0) return 8;
    ncnn::Mat reference_flat=flatten(reference_latent);
    double encode_max=0.0,encode_cos=0.0;
    metrics(reference_flat,latent_flat,encode_max,encode_cos);

    ncnn::VkMat reference_packed,reference_gpu;
    ncnn::VkCompute reference_upload(execution.device());
    reference_upload.record_upload(reference_flat,reference_packed,execution.option());
    execution.device()->convert_packing(reference_packed,reference_gpu,1,
                                        reference_upload,execution.option());
    if(reference_upload.submit_and_wait()!=0) return 9;
    ncnn::Mat reference_video;
    if(vae.decode(reference_latent,0.9152f,reference_video)!=0) return 10;
    ncnn::VkMat decoded_gpu;
    if(vae.decode_vkmat(reference_gpu,reference_latent.d,reference_latent.h,
                        reference_latent.w,0.9152f,execution,decoded_gpu)!=0)
    {std::cerr<<vae.last_error()<<'\n';return 11;}
    ncnn::Mat decoded;
    ncnn::VkCompute video_download(execution.device());
    video_download.record_download(decoded_gpu,decoded,execution.option());
    if(video_download.submit_and_wait()!=0) return 12;
    double decode_max=0.0,decode_cos=0.0;
    metrics(reference_video,decoded,decode_max,decode_cos);

    std::cout<<"encode max_abs="<<encode_max<<" cosine="<<encode_cos<<'\n'
             <<"decode max_abs="<<decode_max<<" cosine="<<decode_cos<<'\n';
    return encode_cos>=0.999999 && decode_cos>=0.999999 ? 0 : 13;
}
