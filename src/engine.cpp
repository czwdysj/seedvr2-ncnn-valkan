// 本文件实现 SeedVR2Engine Facade，把预处理、VAE、条件构造、流式 DiT、
// CFG/Euler 和后处理编排成一次 process 调用。模块只通过明确的 Mat 契约连接，
// 对外不泄露 NCNN；任何阶段失败都会保留带阶段上下文的 last_error。
#include "seedvr2/engine.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <random>
#include <utility>

#include "dit.h"
#include "postprocessing.h"
#include "preprocessing.h"
#include "runtime_context.h"
#include "sampler.h"
#include "vae.h"

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
        if (!positive.valid() || positive.channels != 5120)
            return fail(Status::InvalidArgument, "positive embedding must be [tokens,5120]");
        if (options.cfg_scale != 1.0f
            && (!negative.valid() || negative.channels != 5120))
            return fail(Status::InvalidArgument,
                        "negative embedding must be [tokens,5120] when CFG is enabled");

        PreparedVideo prepared;
        int result = preprocess_video(input, prepared, error);
        if (result != 0)
            return result;

        std::mt19937_64 random(options.seed);
        ncnn::Mat encoded;
        result = vae.encode(prepared.tensor,
                            random,
                            options.stochastic_vae,
                            options.vae_scaling_factor,
                            encoded);
        if (result != 0)
            return fail(static_cast<Status>(result), "VAE encode failed: " + vae.last_error());

        // PyTorch 基准在 VAE posterior 之后依次生成 initial_noise 和 augment_noise。
        ncnn::Mat latent = make_random_latent(encoded, random);
        ncnn::Mat augment_noise = make_random_latent(encoded, random);
        if (latent.empty() || augment_noise.empty())
            return fail(Status::OutOfMemory, "failed to allocate diffusion noise");
        const ncnn::Mat condition =
            make_condition_latent(encoded, augment_noise, options.condition_noise_scale);
        const ncnn::Mat positive_text = make_text_mat(positive);
        const ncnn::Mat negative_text = options.cfg_scale == 1.0f ? ncnn::Mat() : make_text_mat(negative);
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

        ncnn::Mat decoded;
        result = vae.decode(latent, options.vae_scaling_factor, decoded);
        if (result != 0)
            return fail(static_cast<Status>(result), "VAE decode failed: " + vae.last_error());
        result = postprocess_video(decoded, prepared, output, error);
        if (result == 0)
            error.clear();
        return result;
    }

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
