#!/usr/bin/env python3
"""
Run a single SeedVR2 PyTorch reference inference and save validation tensors.

This script is intended for the first NCNN porting baseline. It runs the
SeedVR2 3B PyTorch model from `pytorch_model/`, saves the final restored video,
and dumps deterministic intermediate tensors that later NCNN and pnnx outputs
can be compared against. It injects lightweight runtime fallbacks for apex fused
normalization and flash-attn varlen attention so small reference cases can run
on environments where those optional CUDA extensions are unavailable.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import sys
import time
import types
import importlib.machinery
from pathlib import Path
from typing import Any


def install_runtime_fallbacks() -> None:
    """Install apex and flash-attn fallback modules before importing SeedVR2."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class FusedRMSNorm(nn.Module):
        """Small RMSNorm fallback matching the apex constructor used by SeedVR2."""

        def __init__(self, normalized_shape, elementwise_affine=True, eps=1e-5):
            super().__init__()
            if isinstance(normalized_shape, int):
                normalized_shape = (normalized_shape,)
            self.normalized_shape = tuple(normalized_shape)
            self.eps = eps
            self.weight = (
                nn.Parameter(torch.ones(self.normalized_shape))
                if elementwise_affine
                else None
            )

        def forward(self, x):
            out = x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
            out = out.to(dtype=x.dtype)
            return out if self.weight is None else out * self.weight

    class FusedLayerNorm(nn.LayerNorm):
        """LayerNorm fallback with apex-compatible constructor arguments."""

        def __init__(self, normalized_shape, elementwise_affine=True, eps=1e-5):
            super().__init__(
                normalized_shape=normalized_shape,
                eps=eps,
                elementwise_affine=elementwise_affine,
            )

    apex_mod = types.ModuleType("apex")
    apex_norm_mod = types.ModuleType("apex.normalization")
    apex_mod.__spec__ = importlib.machinery.ModuleSpec("apex", loader=None)
    apex_norm_mod.__spec__ = importlib.machinery.ModuleSpec("apex.normalization", loader=None)
    apex_norm_mod.FusedLayerNorm = FusedLayerNorm
    apex_norm_mod.FusedRMSNorm = FusedRMSNorm
    apex_mod.normalization = apex_norm_mod
    sys.modules.setdefault("apex", apex_mod)
    sys.modules.setdefault("apex.normalization", apex_norm_mod)

    def flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q=None,
        max_seqlen_k=None,
        deterministic=False,
        **kwargs,
    ):
        del max_seqlen_q, max_seqlen_k, deterministic, kwargs
        outputs = []
        scale = 1.0 / math.sqrt(q.shape[-1])
        # The varlen fallback preserves the segment boundaries from flash-attn.
        # Each segment is still a joint [video_window, text] self-attention.
        for i in range(cu_seqlens_q.numel() - 1):
            qs, qe = int(cu_seqlens_q[i]), int(cu_seqlens_q[i + 1])
            ks, ke = int(cu_seqlens_k[i]), int(cu_seqlens_k[i + 1])
            q_i = q[qs:qe].permute(1, 0, 2).unsqueeze(0)
            k_i = k[ks:ke].permute(1, 0, 2).unsqueeze(0)
            v_i = v[ks:ke].permute(1, 0, 2).unsqueeze(0)
            out_i = F.scaled_dot_product_attention(q_i, k_i, v_i, scale=scale)
            outputs.append(out_i.squeeze(0).permute(1, 0, 2))
        return torch.cat(outputs, dim=0)

    flash_mod = types.ModuleType("flash_attn")
    flash_mod.__spec__ = importlib.machinery.ModuleSpec("flash_attn", loader=None)
    flash_mod.flash_attn_varlen_func = flash_attn_varlen_func
    sys.modules.setdefault("flash_attn", flash_mod)


def tensor_stats(tensor) -> dict[str, Any]:
    import torch

    data = tensor.detach()
    finite = data.float()
    return {
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "device": str(data.device),
        "min": float(finite.min().item()) if finite.numel() else None,
        "max": float(finite.max().item()) if finite.numel() else None,
        "mean": float(finite.mean().item()) if finite.numel() else None,
        "std": float(finite.std(unbiased=False).item()) if finite.numel() else None,
    }


def save_tensor(path: Path, tensor, metadata: dict[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    cpu_tensor = tensor.detach().to("cpu")
    torch.save(cpu_tensor, path)
    metadata["tensors"][path.name] = tensor_stats(cpu_tensor)


def read_video_cv2(video_path: Path):
    import cv2
    import torch

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(torch.from_numpy(frame_rgb).permute(2, 0, 1).contiguous())
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from video: {video_path}")
    return torch.stack(frames, dim=0).float() / 255.0, float(fps)


def write_video_cv2(path: Path, video_uint8, fps: float) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    frames = video_uint8.detach().cpu().numpy()
    height, width = frames.shape[1], frames.shape[2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps > 0 else 24.0,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer: {path}")
    for frame_rgb in frames:
        writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    writer.release()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--res-h", type=int, default=240)
    parser.add_argument("--res-w", type=int, default=320)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--cfg-rescale", type=float, default=0.0)
    parser.add_argument("--sample-steps", type=int, default=1)
    args = parser.parse_args()

    install_runtime_fallbacks()

    import torch
    from einops import rearrange
    from omegaconf import OmegaConf
    from torchvision.transforms import Compose, Lambda, Normalize

    original_dist_barrier = torch.distributed.barrier

    def single_process_safe_barrier(*barrier_args, **barrier_kwargs):
        """Let SeedVR2 logging decorators run without a torchrun process group."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return original_dist_barrier(*barrier_args, **barrier_kwargs)
        return None

    torch.distributed.barrier = single_process_safe_barrier

    project_dir = args.project_dir.resolve()
    os.chdir(project_dir)
    sys.path.insert(0, str(project_dir))

    from common.config import load_config
    from common.diffusion import classifier_free_guidance
    from common.diffusion.samplers.base import SamplerModelArgs
    from common.distributed import get_device
    from common.seed import set_seed
    from data.image.transforms.divisible_crop import DivisibleCrop
    from data.image.transforms.na_resize import NaResize
    from data.video.transforms.rearrange import Rearrange
    from models.dit_v2 import na
    from projects.video_diffusion_sr.infer import VideoDiffusionInfer

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(0)
    set_seed(args.seed, same_across_ranks=True)
    random.seed(args.seed)

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "created_unix": time.time(),
        "command": " ".join(sys.argv),
        "project_dir": str(project_dir),
        "video": str(args.video.resolve()),
        "seed": args.seed,
        "res_h": args.res_h,
        "res_w": args.res_w,
        "cfg_scale": args.cfg_scale,
        "cfg_rescale": args.cfg_rescale,
        "sample_steps": args.sample_steps,
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "tensors": {},
    }

    config = load_config("./configs_3b/main.yaml")
    OmegaConf.set_readonly(config, False)
    config.diffusion.cfg.scale = args.cfg_scale
    config.diffusion.cfg.rescale = args.cfg_rescale
    config.diffusion.timesteps.sampling.steps = args.sample_steps

    runner = VideoDiffusionInfer(config)
    runner.configure_dit_model(device="cuda", checkpoint="./ckpts/seedvr2_ema_3b.pth")
    # The 3B checkpoint loads as fp32 weights. RTX 5090 32GB cannot keep the
    # full DiT plus fallback attention activations in fp32, so the reference
    # run stores and executes DiT weights in bf16.
    runner.dit.to(device=get_device(), dtype=torch.bfloat16)
    runner.configure_vae_model()
    # Keep the small reference case on the VAE basic causal-conv path. The
    # memory-limited slicing path queries distributed sequence-parallel ranks.
    if hasattr(runner.vae, "set_memory_limit"):
        runner.vae.set_memory_limit(conv_max_mem=None, norm_max_mem=None)
    runner.configure_diffusion()
    runner.dit.eval()
    runner.vae.eval()

    video_raw, fps = read_video_cv2(args.video)
    metadata["input_video"] = {"fps": fps, **tensor_stats(video_raw)}
    save_tensor(out_dir / "input_video_raw.pt", video_raw, metadata)

    transform = Compose(
        [
            NaResize(
                resolution=(args.res_h * args.res_w) ** 0.5,
                mode="area",
                downsample_only=False,
            ),
            Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
            DivisibleCrop((16, 16)),
            Normalize(0.5, 0.5),
            Rearrange("t c h w -> c t h w"),
        ]
    )
    input_transformed = transform(video_raw.to(get_device()))
    save_tensor(out_dir / "input_video_transformed.pt", input_transformed, metadata)

    cond_video = input_transformed
    if cond_video.size(1) <= 4:
        padding = [cond_video[:, -1].unsqueeze(1)] * (4 - cond_video.size(1) + 1)
        cond_video = torch.cat([cond_video, torch.cat(padding, dim=1)], dim=1)
    elif (cond_video.size(1) - 1) % 4 != 0:
        padding = [cond_video[:, -1].unsqueeze(1)] * (4 - ((cond_video.size(1) - 1) % 4))
        cond_video = torch.cat([cond_video, torch.cat(padding, dim=1)], dim=1)
    save_tensor(out_dir / "input_video_cut.pt", cond_video, metadata)

    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        runner.dit.to("cpu")
        runner.vae.to(get_device())
        cond_latent = runner.vae_encode([cond_video])[0]
        runner.vae.to("cpu")
        torch.cuda.empty_cache()
        runner.dit.to(get_device())
    save_tensor(out_dir / "vae_encode_latent.pt", cond_latent, metadata)

    text_pos = torch.load("pos_emb.pt", map_location=get_device())
    text_neg = torch.load("neg_emb.pt", map_location=get_device())
    save_tensor(out_dir / "text_pos_emb.pt", text_pos, metadata)
    save_tensor(out_dir / "text_neg_emb.pt", text_neg, metadata)

    noise = torch.randn_like(cond_latent)
    aug_noise = torch.randn_like(cond_latent)
    save_tensor(out_dir / "initial_noise.pt", noise, metadata)
    save_tensor(out_dir / "augment_noise.pt", aug_noise, metadata)

    zero_t = torch.tensor([0.0], device=get_device())
    zero_t = runner.timestep_transform(zero_t, torch.tensor(cond_latent.shape[1:], device=get_device())[None])
    latent_blur = runner.schedule.forward(cond_latent, aug_noise, zero_t)
    condition = runner.get_condition(noise, task="sr", latent_blur=latent_blur)
    save_tensor(out_dir / "condition.pt", condition, metadata)

    text_pos_flat, text_pos_shape = na.flatten([text_pos])
    text_neg_flat, text_neg_shape = na.flatten([text_neg])
    latents, latents_shape = na.flatten([noise])
    latents_cond, _ = na.flatten([condition])
    save_tensor(out_dir / "latents_shape.pt", latents_shape, metadata)
    save_tensor(out_dir / "text_pos_shape.pt", text_pos_shape, metadata)
    save_tensor(out_dir / "text_neg_shape.pt", text_neg_shape, metadata)

    block_dump_ids = {0, 15, 31}
    hook_handles = []

    def make_block_hook(index: int):
        def hook(module, inputs, kwargs, output):
            del module, inputs
            vid_in, txt_in = kwargs["vid"], kwargs["txt"]
            vid_out, txt_out = output[0], output[1]
            prefix = f"block_{index:02d}"
            save_tensor(out_dir / f"{prefix}_input_vid.pt", vid_in, metadata)
            save_tensor(out_dir / f"{prefix}_input_txt.pt", txt_in, metadata)
            save_tensor(out_dir / f"{prefix}_output_vid.pt", vid_out, metadata)
            save_tensor(out_dir / f"{prefix}_output_txt.pt", txt_out, metadata)
        return hook

    for idx, block in enumerate(runner.dit.blocks):
        if idx in block_dump_ids:
            hook_handles.append(block.register_forward_hook(make_block_hook(idx), with_kwargs=True))

    timesteps = runner.sampler.timesteps.timesteps
    x = latents
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        for step_index, t in enumerate(timesteps):
            dit_input = torch.cat([x, latents_cond], dim=-1)
            save_tensor(out_dir / f"dit_input_pos_step{step_index:03d}.pt", dit_input, metadata)
            pos = runner.dit(
                vid=dit_input,
                txt=text_pos_flat,
                vid_shape=latents_shape,
                txt_shape=text_pos_shape,
                timestep=t.repeat(1),
            ).vid_sample
            save_tensor(out_dir / f"dit_output_pos_step{step_index:03d}.pt", pos, metadata)

            neg = runner.dit(
                vid=dit_input,
                txt=text_neg_flat,
                vid_shape=latents_shape,
                txt_shape=text_neg_shape,
                timestep=t.repeat(1),
            ).vid_sample
            save_tensor(out_dir / f"dit_output_neg_step{step_index:03d}.pt", neg, metadata)

            pred = classifier_free_guidance(pos, neg, scale=args.cfg_scale, rescale=args.cfg_rescale)
            save_tensor(out_dir / f"cfg_output_step{step_index:03d}.pt", pred, metadata)

            x = runner.sampler.get_endpoint(pred, x, t)
            save_tensor(out_dir / f"sampler_latent_after_step{step_index:03d}.pt", x, metadata)
            break

    for handle in hook_handles:
        handle.remove()

    decoded_latent = na.unflatten(x, latents_shape)[0]
    save_tensor(out_dir / "vae_decode_input.pt", decoded_latent, metadata)
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        runner.dit.to("cpu")
        runner.vae.to(get_device())
        decoded = runner.vae_decode([decoded_latent])[0]
    save_tensor(out_dir / "vae_decode_output.pt", decoded, metadata)

    final_video_tensor = rearrange(decoded, "c t h w -> t c h w")
    save_tensor(out_dir / "final_video_tensor.pt", final_video_tensor, metadata)
    final_uint8 = final_video_tensor.clip(-1, 1).mul(0.5).add(0.5).mul(255).round().to(torch.uint8)
    final_uint8_thwc = rearrange(final_uint8, "t c h w -> t h w c")
    save_tensor(out_dir / "final_video_uint8_thwc.pt", final_uint8_thwc, metadata)
    write_video_cv2(out_dir / "final_video.mp4", final_uint8_thwc, fps)

    metadata["weights"] = {
        "dit": str((project_dir / "ckpts/seedvr2_ema_3b.pth").resolve()),
        "vae": str((project_dir / "ckpts/ema_vae.pth").resolve()),
    }
    metadata["sampling_timesteps"] = [float(v.item()) for v in timesteps.detach().cpu()]
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved reference outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
