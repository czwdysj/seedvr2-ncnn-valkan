// 本文件实现 SeedVR2Engine Facade，把预处理、VAE、条件构造、流式 DiT、
// CFG/Euler 和后处理编排成一次 process 调用。模块只通过明确的 Mat 契约连接，
// 对外不泄露 NCNN；任何阶段失败都会保留带阶段上下文的 last_error。
#include "seedvr2/engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <utility>

#include "model/dit.h"
#include "pipeline/postprocessing.h"
#include "pipeline/preprocessing.h"
#include "core/runtime_context.h"
#include "model/sampler.h"
#include "model/vae.h"

namespace seedvr2
{
namespace
{
bool checked_element_count(int a, int b, int c, int d, std::size_t& result)
{
    if (a <= 0 || b <= 0 || c <= 0 || d <= 0)
        return false;
    const std::size_t values[] = {static_cast<std::size_t>(a), static_cast<std::size_t>(b),
                                  static_cast<std::size_t>(c), static_cast<std::size_t>(d)};
    result = 1;
    for (const std::size_t value : values)
    {
        if (result > std::numeric_limits<std::size_t>::max() / value)
            return false;
        result *= value;
    }
    return true;
}

ncnn::Mat make_text_mat(const TextEmbedding& embedding)
{
    ncnn::Mat result(embedding.channels,
                     embedding.tokens,
                     static_cast<size_t>(4u),
                     1);
    if (!result.empty())
        std::copy(embedding.data.begin(), embedding.data.end(), static_cast<float*>(result.data));
    return result;
}

ncnn::Mat make_random_latent(const ncnn::Mat& shape, std::mt19937_64& random)
{
    ncnn::Mat result(shape.w, shape.h, shape.d, shape.c, 4u, 1);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    for (int channel = 0; channel < result.c; ++channel)
        for (int frame = 0; frame < result.d; ++frame)
            for (int y = 0; y < result.h; ++y)
            {
                float* row = result.channel(channel).depth(frame).row(y);
                for (int x = 0; x < result.w; ++x)
                    row[x] = normal(random);
            }
    return result;
}

float transformed_condition_timestep(float noise_scale, const ncnn::Mat& latent)
{
    const float timestep = noise_scale * 1000.0f;
    if (timestep == 0.0f)
        return 0.0f;
    const float frames = static_cast<float>((latent.d - 1) * 4 + 1);
    const float height = static_cast<float>(latent.h * 8);
    const float width = static_cast<float>(latent.w * 8);
    const float x1 = 256.0f * 256.0f * 37.0f;
    const float x2 = 1280.0f * 720.0f * 145.0f;
    const float shift = 1.0f + (height * width * frames - x1) * (5.0f - 1.0f) / (x2 - x1);
    const float normalized = timestep / 1000.0f;
    return 1000.0f * shift * normalized / (1.0f + (shift - 1.0f) * normalized);
}

ncnn::Mat make_condition_latent(const ncnn::Mat& encoded,
                                const ncnn::Mat& augment_noise,
                                float condition_noise_scale)
{
    ncnn::Mat result(encoded.w, encoded.h, encoded.d, encoded.c, 4u, 1);
    const float ratio = transformed_condition_timestep(condition_noise_scale, encoded) / 1000.0f;
    for (int channel = 0; channel < encoded.c; ++channel)
        for (int frame = 0; frame < encoded.d; ++frame)
            for (int y = 0; y < encoded.h; ++y)
            {
                const float* source = encoded.channel(channel).depth(frame).row(y);
                const float* noise = augment_noise.channel(channel).depth(frame).row(y);
                float* destination = result.channel(channel).depth(frame).row(y);
                for (int x = 0; x < encoded.w; ++x)
                    destination[x] = (1.0f - ratio) * source[x] + ratio * noise[x];
            }
    return result;
}

ncnn::Mat make_dit_input(const ncnn::Mat& noise, const ncnn::Mat& condition)
{
    ncnn::Mat result(noise.w, noise.h, noise.d, 33, 4u, 1);
    for (int channel = 0; channel < 33; ++channel)
        for (int frame = 0; frame < noise.d; ++frame)
            for (int y = 0; y < noise.h; ++y)
            {
                float* destination = result.channel(channel).depth(frame).row(y);
                if (channel < 16)
                {
                    const float* source = noise.channel(channel).depth(frame).row(y);
                    std::copy(source, source + noise.w, destination);
                }
                else if (channel < 32)
                {
                    const float* source = condition.channel(channel - 16).depth(frame).row(y);
                    std::copy(source, source + noise.w, destination);
                }
                else
                {
                    std::fill(destination, destination + noise.w, 1.0f);
                }
            }
    return result;
}

#if NCNN_VULKAN
ncnn::Mat flatten_cthw(const ncnn::Mat& value)
{
    ncnn::Mat result(value.c, value.d * value.h * value.w, static_cast<size_t>(4u), 1);
    int token = 0;
    for (int frame = 0; frame < value.d; ++frame)
        for (int y = 0; y < value.h; ++y)
            for (int x = 0; x < value.w; ++x, ++token)
                for (int channel = 0; channel < value.c; ++channel)
                    result.row(token)[channel] = value.channel(channel).depth(frame).row(y)[x];
    return result;
}

ncnn::Mat unflatten_cthw(const ncnn::Mat& value, int frames, int height, int width)
{
    ncnn::Mat result(width, height, frames, value.w, 4u, 1);
    int token = 0;
    for (int frame = 0; frame < frames; ++frame)
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x, ++token)
                for (int channel = 0; channel < value.w; ++channel)
                    result.channel(channel).depth(frame).row(y)[x] = value.row(token)[channel];
    return result;
}

// ncnn 上传二维 Mat 时会按行数自动选择 pack4。Engine 的融合 shader 使用明确的
// token-major 标量地址，因此入口统一转换为 pack1，之后整个 sampler 循环保持该布局。
void record_upload_pack1(const ncnn::Mat& source,
                         VulkanExecutionContext& execution,
                         ncnn::VkCompute& command,
                         ncnn::VkMat& packed,
                         ncnn::VkMat& output)
{
    command.record_upload(source, packed, execution.option());
    execution.device()->convert_packing(packed, output, 1, command, execution.option());
}
#endif

// 读取模型目录可选携带的默认文本 embedding（default_pos_emb.bin / default_neg_emb.bin，
// 由 tools/export_default_embeddings.py 从官方 pos_emb.pt / neg_emb.pt 转出）。
// 文件布局：int32 tokens | int32 channels | fp32 data[tokens*channels]。
bool read_default_embedding(const std::filesystem::path& path, TextEmbedding& embedding)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        return false;
    std::int32_t header[2] = {0, 0};
    stream.read(reinterpret_cast<char*>(header), sizeof(header));
    if (!stream || header[0] <= 0 || header[1] != 5120)
        return false;
    TextEmbedding loaded;
    loaded.tokens = header[0];
    loaded.channels = header[1];
    loaded.data.resize(static_cast<std::size_t>(loaded.tokens) * loaded.channels);
    stream.read(reinterpret_cast<char*>(loaded.data.data()),
                static_cast<std::streamsize>(loaded.data.size() * sizeof(float)));
    if (!stream || !loaded.valid())
        return false;
    embedding = std::move(loaded);
    return true;
}
} // namespace

class SeedVR2Engine::Impl
{
public:
    int load(const std::string& model_dir, const RuntimeOptions& requested_options)
    {
        if (model_dir.empty())
            return fail(Status::InvalidArgument, "model_dir is empty");
        int result = context.initialize(requested_options, error);
        if (result != 0)
            return result;

        const std::filesystem::path root(model_dir);
        const std::filesystem::path vae_dir =
            std::filesystem::is_directory(root / "vae_dynamic") ? root / "vae_dynamic" : root;
        const std::filesystem::path dit_dir = std::filesystem::is_directory(root / "dit_full_fp16")
            ? root / "dit_full_fp16"
            : root;

        result = vae.load(vae_dir.string(), context);
        if (result != 0)
            return fail(static_cast<Status>(result), "VAE load failed: " + vae.last_error());
        result = dit.load(dit_dir.string(), context);
        if (result != 0)
            return fail(static_cast<Status>(result), "DiT load failed: " + dit.last_error());

        options = context.options();
        sampler = std::make_unique<EulerSampler>(options.sampling_steps);
#if NCNN_VULKAN
        if (options.device == DeviceType::Vulkan)
        {
            result = sampler->initialize_vulkan(context, error);
            if (result != 0)
                return result;
        }
#endif
        // 方案 A：模型目录可选携带官方默认文本 embedding；存在则加载，允许调用方
        // process() 传空 embedding 直接使用默认文本条件（开箱即用，无 PyTorch 依赖）。
        read_default_embedding(root / "default_pos_emb.bin", default_positive);
        read_default_embedding(root / "default_neg_emb.bin", default_negative);
        is_loaded = true;
        error.clear();
        return static_cast<int>(Status::Ok);
    }

    int process(const Video& input,
                const TextEmbedding& positive,
                const TextEmbedding& negative,
                Video& output)
    {
        if (!is_loaded)
            return fail(Status::NotLoaded, "engine is not loaded");
        // 调用方传空 embedding 时回退到模型目录内的默认文本条件。
        TextEmbedding positive_used = positive;
        if (!positive_used.valid() && default_positive.valid())
            positive_used = default_positive;
        TextEmbedding negative_used = negative;
        if (!negative_used.valid() && default_negative.valid())
            negative_used = default_negative;
        if (!positive_used.valid() || positive_used.channels != 5120)
            return fail(Status::InvalidArgument,
                        "positive embedding is empty and model dir has no default_pos_emb.bin");
        if (options.cfg_scale != 1.0f
            && (!negative_used.valid() || negative_used.channels != 5120))
            return fail(Status::InvalidArgument,
                        "negative embedding must be [tokens,5120] when CFG is enabled");

#if NCNN_VULKAN
        if (options.device == DeviceType::Vulkan)
            return process_vulkan(input, positive_used, negative_used, output);
#endif

        const bool profile = std::getenv("SEEDVR2_PROFILE") != nullptr;
        const auto t_start = std::chrono::steady_clock::now();
        auto t_enc = t_start;
        auto t_dit = t_start;
        auto t_dec = t_start;

        PreparedVideo prepared;
        int result = preprocess_video(input, prepared, error);
        if (result != 0)
            return result;
        t_enc = std::chrono::steady_clock::now();

        std::mt19937_64 random(options.seed);
        ncnn::Mat encoded;
        result = vae.encode(prepared.tensor,
                            random,
                            options.stochastic_vae,
                            options.vae_scaling_factor,
                            encoded);
        if (result != 0)
            return fail(static_cast<Status>(result), "VAE encode failed: " + vae.last_error());
        t_dit = std::chrono::steady_clock::now();

        // PyTorch 基准在 VAE posterior 之后依次生成 initial_noise 和 augment_noise。
        ncnn::Mat latent = make_random_latent(encoded, random);
        ncnn::Mat augment_noise = make_random_latent(encoded, random);
        if (latent.empty() || augment_noise.empty())
            return fail(Status::OutOfMemory, "failed to allocate diffusion noise");
        const ncnn::Mat condition =
            make_condition_latent(encoded, augment_noise, options.condition_noise_scale);
        const ncnn::Mat positive_text = make_text_mat(positive_used);
        const ncnn::Mat negative_text =
            options.cfg_scale == 1.0f ? ncnn::Mat() : make_text_mat(negative_used);
        if (condition.empty() || positive_text.empty()
            || (options.cfg_scale != 1.0f && negative_text.empty()))
            return fail(Status::OutOfMemory, "failed to allocate DiT condition tensors");

        const std::vector<float>& timesteps = sampler->timesteps();
        for (std::size_t index = 0; index < timesteps.size(); ++index)
        {
            const ncnn::Mat dit_input = make_dit_input(latent, condition);
            if (dit_input.empty())
                return fail(Status::OutOfMemory, "failed to allocate 33-channel DiT input");
            ncnn::Mat prediction;
            result = dit.forward(dit_input, positive_text, timesteps[index], prediction);
            if (result != 0)
                return fail(static_cast<Status>(result), "positive DiT failed: " + dit.last_error());

            if (options.cfg_scale != 1.0f)
            {
                ncnn::Mat negative_prediction;
                result = dit.forward(dit_input, negative_text, timesteps[index], negative_prediction);
                if (result != 0)
                    return fail(static_cast<Status>(result),
                                "negative DiT failed: " + dit.last_error());
                ncnn::Mat guided;
                result = sampler->apply_cfg(prediction,
                                            negative_prediction,
                                            options.cfg_scale,
                                            options.cfg_rescale,
                                            guided,
                                            error);
                if (result != 0)
                    return result;
                prediction = guided;
            }

            ncnn::Mat next;
            if (index + 1 < timesteps.size())
                result = sampler->step_to(
                    prediction, latent, timesteps[index], timesteps[index + 1], next, error);
            else
                result = sampler->endpoint(prediction, latent, timesteps[index], next, error);
            if (result != 0)
                return result;
            latent = next;
        }
        t_dec = std::chrono::steady_clock::now();

        ncnn::Mat decoded;
        result = vae.decode(latent, options.vae_scaling_factor, decoded);
        if (result != 0)
            return fail(static_cast<Status>(result), "VAE decode failed: " + vae.last_error());
        const auto t_post = std::chrono::steady_clock::now();

        result = postprocess_video(decoded, prepared, output, error);
        if (result == 0)
            error.clear();

        if (profile)
        {
            const auto t_end = std::chrono::steady_clock::now();
            const auto ms = [](std::chrono::steady_clock::time_point a,
                               std::chrono::steady_clock::time_point b) {
                return std::chrono::duration<double, std::milli>(b - a).count();
            };
            std::fprintf(stderr,
                         "[PROFILE] preprocess %.1f ms | vae_encode %.1f ms | dit_%zu_steps %.1f ms | vae_decode %.1f ms | postprocess %.1f ms | total %.1f ms\n",
                         ms(t_start, t_enc), ms(t_enc, t_dit), timesteps.size(),
                         ms(t_dit, t_dec), ms(t_dec, t_post), ms(t_post, t_end), ms(t_start, t_end));
        }
        return result;
    }

#if NCNN_VULKAN
    int process_vulkan(const Video& input,
                       const TextEmbedding& positive,
                       const TextEmbedding& negative,
                       Video& output)
    {
        const bool profile = std::getenv("SEEDVR2_PROFILE") != nullptr;
        const auto t_start = std::chrono::steady_clock::now();
        PreparedVideo prepared;
        int result = preprocess_video(input, prepared, error);
        if (result != 0)
            return result;
        const auto t_preprocessed = std::chrono::steady_clock::now();

        // 本阶段 VAE 仍保留 Mat 边界；VAE 输出后，所有 diffusion 大张量直到
        // decoder 输入前均驻留共享 allocator 的 VkMat，不再按采样步往返 CPU。
        std::mt19937_64 random(options.seed);
        ncnn::Mat encoded;
        result = vae.encode(prepared.tensor,
                            random,
                            options.stochastic_vae,
                            options.vae_scaling_factor,
                            encoded);
        if (result != 0)
            return fail(static_cast<Status>(result), "VAE encode failed: " + vae.last_error());
        const auto t_encoded = std::chrono::steady_clock::now();

        ncnn::Mat latent_cpu = make_random_latent(encoded, random);
        ncnn::Mat augment_cpu = make_random_latent(encoded, random);
        ncnn::Mat latent_flat = flatten_cthw(latent_cpu);
        ncnn::Mat encoded_flat = flatten_cthw(encoded);
        ncnn::Mat augment_flat = flatten_cthw(augment_cpu);
        ncnn::Mat positive_text = make_text_mat(positive);
        ncnn::Mat negative_text = options.cfg_scale == 1.0f ? ncnn::Mat() : make_text_mat(negative);
        if (latent_flat.empty() || encoded_flat.empty() || augment_flat.empty()
            || positive_text.empty() || (options.cfg_scale != 1.0f && negative_text.empty()))
            return fail(Status::OutOfMemory, "failed to allocate Vulkan sampler inputs");

        VulkanExecutionContext execution(context);
        if (!execution.valid())
            return fail(Status::OutOfMemory, "failed to acquire Engine Vulkan allocators");

        ncnn::VkMat latent_packed, latent;
        ncnn::VkMat encoded_packed, encoded_gpu;
        ncnn::VkMat augment_packed, augment_gpu;
        ncnn::VkMat positive_packed, positive_gpu;
        ncnn::VkMat negative_packed, negative_gpu;
        ncnn::VkCompute upload(execution.device());
        record_upload_pack1(latent_flat, execution, upload, latent_packed, latent);
        record_upload_pack1(encoded_flat, execution, upload, encoded_packed, encoded_gpu);
        record_upload_pack1(augment_flat, execution, upload, augment_packed, augment_gpu);
        record_upload_pack1(positive_text, execution, upload, positive_packed, positive_gpu);
        if (options.cfg_scale != 1.0f)
            record_upload_pack1(negative_text, execution, upload, negative_packed, negative_gpu);
        if (upload.submit_and_wait() != 0)
            return fail(Status::InferenceFailed, "failed to upload Vulkan sampler inputs");

        const float condition_ratio =
            transformed_condition_timestep(options.condition_noise_scale, encoded) / 1000.0f;
        ncnn::VkMat condition;
        ncnn::VkCompute condition_command(execution.device());
        result = sampler->make_condition_vulkan(encoded_gpu,
                                                 augment_gpu,
                                                 condition_ratio,
                                                 execution,
                                                 condition_command,
                                                 condition,
                                                 error);
        if (result != 0 || condition_command.submit_and_wait() != 0)
            return result != 0 ? result : fail(Status::InferenceFailed,
                                                "Vulkan condition dispatch failed");

        const int frames = encoded.d;
        const int height = encoded.h;
        const int width = encoded.w;
        const std::vector<float>& timesteps = sampler->timesteps();
        for (std::size_t index = 0; index < timesteps.size(); ++index)
        {
            ncnn::VkMat dit_input;
            ncnn::VkCompute prepare_command(execution.device());
            result = sampler->make_dit_input_vulkan(latent,
                                                    condition,
                                                    execution,
                                                    prepare_command,
                                                    dit_input,
                                                    error);
            if (result != 0 || prepare_command.submit_and_wait() != 0)
                return result != 0 ? result : fail(Status::InferenceFailed,
                                                    "Vulkan DiT input dispatch failed");

            ncnn::VkMat positive_prediction_packed;
            result = dit.forward_vkmat(dit_input,
                                       positive_gpu,
                                       frames,
                                       height,
                                       width,
                                       timesteps[index],
                                       execution,
                                       positive_prediction_packed);
            if (result != 0)
                return fail(static_cast<Status>(result), "positive DiT failed: " + dit.last_error());

            ncnn::VkMat positive_prediction;
            ncnn::VkCompute sample_command(execution.device());
            execution.device()->convert_packing(positive_prediction_packed,
                                                 positive_prediction,
                                                 1,
                                                 sample_command,
                                                 execution.option());
            ncnn::VkMat prediction = positive_prediction;
            if (options.cfg_scale != 1.0f)
            {
                ncnn::VkMat negative_prediction_packed;
                result = dit.forward_vkmat(dit_input,
                                           negative_gpu,
                                           frames,
                                           height,
                                           width,
                                           timesteps[index],
                                           execution,
                                           negative_prediction_packed);
                if (result != 0)
                    return fail(static_cast<Status>(result),
                                "negative DiT failed: " + dit.last_error());
                ncnn::VkMat negative_prediction;
                execution.device()->convert_packing(negative_prediction_packed,
                                                     negative_prediction,
                                                     1,
                                                     sample_command,
                                                     execution.option());
                ncnn::VkMat guided;
                result = sampler->apply_cfg_vulkan(positive_prediction,
                                                   negative_prediction,
                                                   options.cfg_scale,
                                                   options.cfg_rescale,
                                                   execution,
                                                   sample_command,
                                                   guided,
                                                   error);
                if (result != 0)
                    return result;
                prediction = guided;
            }

            ncnn::VkMat next;
            const float next_timestep = index + 1 < timesteps.size() ? timesteps[index + 1] : 0.0f;
            result = sampler->step_vulkan(prediction,
                                          latent,
                                          timesteps[index],
                                          next_timestep,
                                          execution,
                                          sample_command,
                                          next,
                                          error);
            if (result != 0 || sample_command.submit_and_wait() != 0)
                return result != 0 ? result : fail(Status::InferenceFailed,
                                                    "Vulkan sampler dispatch failed");
            latent = next;
        }
        const auto t_dit = std::chrono::steady_clock::now();

        ncnn::Mat latent_result_flat;
        ncnn::VkCompute download(execution.device());
        download.record_download(latent, latent_result_flat, execution.option());
        if (download.submit_and_wait() != 0)
            return fail(Status::InferenceFailed, "failed to download final Vulkan latent");
        ncnn::Mat latent_result = unflatten_cthw(latent_result_flat, frames, height, width);
        ncnn::Mat decoded;
        result = vae.decode(latent_result, options.vae_scaling_factor, decoded);
        if (result != 0)
            return fail(static_cast<Status>(result), "VAE decode failed: " + vae.last_error());
        const auto t_decoded = std::chrono::steady_clock::now();
        result = postprocess_video(decoded, prepared, output, error);

        if (profile)
        {
            const auto t_end = std::chrono::steady_clock::now();
            const auto ms = [](auto a, auto b) {
                return std::chrono::duration<double, std::milli>(b - a).count();
            };
            std::fprintf(stderr,
                         "[PROFILE Vulkan resident tensors] preprocess %.1f ms | vae_encode %.1f ms | "
                         "dit_%zu_steps %.1f ms | vae_decode %.1f ms | postprocess %.1f ms | total %.1f ms\n",
                         ms(t_start, t_preprocessed),
                         ms(t_preprocessed, t_encoded),
                         timesteps.size(),
                         ms(t_encoded, t_dit),
                         ms(t_dit, t_decoded),
                         ms(t_decoded, t_end),
                         ms(t_start, t_end));
        }
        if (result == 0)
            error.clear();
        return result;
    }
#endif

    int fail(Status status, std::string message)
    {
        error = std::move(message);
        return static_cast<int>(status);
    }

    RuntimeContext context;
    RuntimeOptions options;
    SeedVR2VAE vae;
    SeedVR2DiT dit;
    std::unique_ptr<EulerSampler> sampler;
    TextEmbedding default_positive;
    TextEmbedding default_negative;
    bool is_loaded = false;
    std::string error;
};

bool Video::valid() const noexcept
{
    std::size_t expected = 0;
    return checked_element_count(frames, height, width, channels, expected)
        && data.size() == expected && std::isfinite(fps) && fps > 0.0f;
}

bool TextEmbedding::valid() const noexcept
{
    std::size_t expected = 0;
    return checked_element_count(tokens, channels, 1, 1, expected) && data.size() == expected;
}

SeedVR2Engine::SeedVR2Engine()
    : impl_(std::make_unique<Impl>())
{
}

SeedVR2Engine::~SeedVR2Engine() = default;
SeedVR2Engine::SeedVR2Engine(SeedVR2Engine&&) noexcept = default;
SeedVR2Engine& SeedVR2Engine::operator=(SeedVR2Engine&&) noexcept = default;

int SeedVR2Engine::load(const std::string& model_dir, const RuntimeOptions& options)
{
    auto candidate = std::make_unique<Impl>();
    const int result = candidate->load(model_dir, options);
    impl_ = std::move(candidate);
    return result;
}

int SeedVR2Engine::process(const Video& input,
                           const TextEmbedding& positive,
                           const TextEmbedding& negative,
                           Video& output)
{
    return impl_->process(input, positive, negative, output);
}

bool SeedVR2Engine::loaded() const noexcept
{
    return impl_->is_loaded;
}

const std::string& SeedVR2Engine::last_error() const noexcept
{
    return impl_->error;
}

const char* status_message(Status status) noexcept
{
    switch (status)
    {
    case Status::Ok: return "ok";
    case Status::InvalidArgument: return "invalid argument";
    case Status::NotLoaded: return "engine is not loaded";
    case Status::ModelNotFound: return "model file not found";
    case Status::ModelLoadFailed: return "model load failed";
    case Status::InferenceFailed: return "inference failed";
    case Status::UnsupportedBackend: return "unsupported backend";
    case Status::OutOfMemory: return "out of memory";
    case Status::IoError: return "I/O error";
    }
    return "unknown error";
}
} // namespace seedvr2
