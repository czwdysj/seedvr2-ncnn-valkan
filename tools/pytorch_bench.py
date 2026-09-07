#!/usr/bin/env python3
"""PyTorch 计时基准：复用 SeedVR2 推理框架，测 VAE encode / DiT / VAE decode 分阶段耗时。

与 dump_seedvr2_reference.py 的区别：不 dump 任何张量、不读 mp4，直接用随机
小视频输入跑通完整推理并打印各阶段耗时与显存，用于和 ncnn Vulkan 端到端对比。

用法：
  python pytorch_bench.py --project-dir /path/to/pytorch_model \
      --res-h 64 --res-w 64 --frames 5 --steps 1
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--res-h", type=int, default=64)
    parser.add_argument("--res-w", type=int, default=64)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args()

    # 复用 dump 脚本的 apex/flash-attn fallback 安装（必须在 import SeedVR2 前）。
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dump_seedvr2_reference import install_runtime_fallbacks

    install_runtime_fallbacks()

    import torch
    from omegaconf import OmegaConf

    project_dir = args.project_dir.resolve()
    os.chdir(project_dir)
    sys.path.insert(0, str(project_dir))

    from common.config import load_config
    from common.diffusion import classifier_free_guidance
    from common.distributed import get_device
    from common.seed import set_seed
    from models.dit_v2 import na
    from projects.video_diffusion_sr.infer import VideoDiffusionInfer

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(0)
    set_seed(666, same_across_ranks=True)

    config = load_config("./configs_3b/main.yaml")
    OmegaConf.set_readonly(config, False)
    config.diffusion.cfg.scale = 1.0
    config.diffusion.cfg.rescale = 0.0
    config.diffusion.timesteps.sampling.steps = args.steps

    def mark(label: str) -> None:
        torch.cuda.synchronize()
        alloc = torch.cuda.memory_allocated() / 1024**2
        print(f"[{label}] {time.perf_counter() - mark.t0:.3f} s  (alloc {alloc:.0f} MB)", flush=True)
        mark.t0 = time.perf_counter()

    mark.t0 = time.perf_counter()

    runner = VideoDiffusionInfer(config)
    runner.configure_dit_model(device="cuda", checkpoint="./ckpts/seedvr2_ema_3b.pth")
    runner.dit.to(dtype=torch.bfloat16)
    runner.configure_vae_model()
    if hasattr(runner.vae, "set_memory_limit"):
        runner.vae.set_memory_limit(conv_max_mem=None, norm_max_mem=None)
    runner.configure_diffusion()
    runner.dit.eval()
    runner.vae.eval()
    mark("model_load")

    # 随机视频输入 [t,c,h,w] -> normalize -> [c,t,h,w]
    video = torch.rand(args.frames, 3, args.res_h, args.res_w, device=get_device())
    cond_video = video.mul(2.0).sub(1.0).permute(1, 0, 2, 3)
    # 帧数 padding 到 (T-1)%4==0（对齐 dump 脚本逻辑）
    if cond_video.size(1) <= 4:
        padding = [cond_video[:, -1].unsqueeze(1)] * (4 - cond_video.size(1) + 1)
        cond_video = torch.cat([cond_video, torch.cat(padding, dim=1)], dim=1)
    elif (cond_video.size(1) - 1) % 4 != 0:
        padding = [cond_video[:, -1].unsqueeze(1)] * (4 - ((cond_video.size(1) - 1) % 4))
        cond_video = torch.cat([cond_video, torch.cat(padding, dim=1)], dim=1)

    # VAE encode（DiT 移到 CPU 以省显存，与官方推理一致）
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        runner.dit.to("cpu")
        runner.vae.to(get_device())
        cond_latent = runner.vae_encode([cond_video])[0]
        runner.vae.to("cpu")
        torch.cuda.empty_cache()
        runner.dit.to(get_device())
    mark("vae_encode")

    text_pos = torch.load("pos_emb.pt", map_location=get_device())
    noise = torch.randn_like(cond_latent)
    aug_noise = torch.randn_like(cond_latent)
    zero_t = torch.tensor([0.0], device=get_device())
    zero_t = runner.timestep_transform(zero_t, torch.tensor(cond_latent.shape[1:], device=get_device())[None])
    latent_blur = runner.schedule.forward(cond_latent, aug_noise, zero_t)
    condition = runner.get_condition(noise, task="sr", latent_blur=latent_blur)
    text_pos_flat, text_pos_shape = na.flatten([text_pos])
    latents, latents_shape = na.flatten([noise])
    latents_cond, _ = na.flatten([condition])
    mark("prepare")

    timesteps = runner.sampler.timesteps.timesteps
    x = latents
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        for step_index, t in enumerate(timesteps):
            dit_input = torch.cat([x, latents_cond], dim=-1).to(torch.bfloat16)
            pos = runner.dit(
                vid=dit_input,
                txt=text_pos_flat.to(torch.bfloat16),
                vid_shape=latents_shape,
                txt_shape=text_pos_shape,
                timestep=t.repeat(1),
            ).vid_sample
            pred = pos
            x = runner.sampler.get_endpoint(pred, x, t)
            break
    mark(f"dit_{len(timesteps)}steps")

    decoded_latent = na.unflatten(x, latents_shape)[0]
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        runner.dit.to("cpu")
        runner.vae.to(get_device())
        decoded = runner.vae_decode([decoded_latent])[0]
    mark("vae_decode")

    print(f"[peak_alloc] {torch.cuda.max_memory_allocated()/1024**2:.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
