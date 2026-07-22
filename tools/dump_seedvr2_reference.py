#!/usr/bin/env python3
"""生成 SeedVR2 3B 到 NCNN/Vulkan 转换所需的 PyTorch 数值参考。

本文件负责运行一个完整的 SeedVR2 视频增强样例，并保存 VAE、DiT、CFG、
Euler sampler 和最终视频的确定性参考张量。开启 ``--dump-all-blocks`` 后，
脚本会区分正负文本分支保存全部 32 个 Transformer block 的输入输出；指定的
deep block 还会保存 QKV、Q/K RMSNorm、RoPE 后 attention 输入和变长序列边界。
这些张量用于逐层判断 NCNN CPU 与 Vulkan 实现是否真的对齐，而不是只比较
最终视频。运行假设为 batch=1、预计算文本 embedding 和单 GPU 推理。
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


def parse_block_ids(value: str) -> set[int]:
    """解析逗号分隔的 block 编号，并拒绝超出 32 层模型范围的值。"""
    if not value.strip():
        return set()
    block_ids = {int(item.strip()) for item in value.split(",")}
    invalid = sorted(index for index in block_ids if index < 0 or index >= 32)
    if invalid:
        raise argparse.ArgumentTypeError(f"block id must be in [0, 31], got {invalid}")
    return block_ids


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
    parser.add_argument(
        "--dit-dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
        help="DiT weight/input precision; VAE reference remains bfloat16",
    )
    parser.add_argument(
        "--dump-all-blocks",
        action="store_true",
        help="save inputs and outputs for all 32 DiT blocks",
    )
    parser.add_argument(
        "--deep-blocks",
        type=parse_block_ids,
        default=parse_block_ids("0,9,10,31"),
        help="comma-separated blocks whose attention internals are saved",
    )
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
        "dit_dtype": args.dit_dtype,
        "dump_all_blocks": args.dump_all_blocks,
        "deep_blocks": sorted(args.deep_blocks),
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
    # BF16 is the official inference path. FP32 is an additional conversion
    # baseline used to separate NCNN implementation errors from BF16 rounding.
    dit_dtype = torch.bfloat16 if args.dit_dtype == "bfloat16" else torch.float32
    runner.dit.to(device=get_device(), dtype=dit_dtype)
    runner.configure_vae_model()
    # Keep the small reference case on the VAE basic causal-conv path. The
    # memory-limited slicing path queries distributed sequence-parallel ranks.
    if hasattr(runner.vae, "set_memory_limit"):
        runner.vae.set_memory_limit(conv_max_mem=None, norm_max_mem=None)
    runner.configure_diffusion()
    runner.dit.eval()
    runner.vae.eval()
    metadata["dit_runtime"] = {
        "block_count": len(runner.dit.blocks),
        "output_ada_layers": list(runner.dit.vid_out_ada.layers),
        "output_ada_dim": int(runner.dit.vid_out_ada.dim),
        "output_ada_emb_dim": int(runner.dit.vid_out_ada.emb_dim),
    }

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

    # 前 10 层使用独立 vid/txt 权重，后续层共享权重，第 31 层又是 vid-only
    # MLP。0/9/10/31 因而覆盖全部结构分支，15 保留为整网中段检查点。
    block_dump_ids = set(range(32)) if args.dump_all_blocks else {0, 15, 31}
    hook_handles = []
    capture_context = {"tag": "inactive"}

    def capture_path(name: str) -> Path:
        """把正负文本分支和 sampler step 编入文件名，避免 hook 结果互相覆盖。"""
        return out_dir / f"{capture_context['tag']}_{name}.pt"

    def save_capture(name: str, tensor) -> None:
        if capture_context["tag"] != "inactive":
            save_tensor(capture_path(name), tensor, metadata)

    def make_block_hook(index: int):
        def hook(module, inputs, kwargs, output):
            del module, inputs
            vid_in, txt_in = kwargs["vid"], kwargs["txt"]
            vid_out, txt_out = output[0], output[1]
            prefix = f"block_{index:02d}"
            save_capture(f"{prefix}_input_vid", vid_in)
            save_capture(f"{prefix}_input_txt", txt_in)
            save_capture(f"{prefix}_output_vid", vid_out)
            save_capture(f"{prefix}_output_txt", txt_out)
        return hook

    def make_mm_hook(index: int, name: str):
        """保存 MMModule 的 vid/txt 双分支输出。"""
        def hook(module, inputs, output):
            del module, inputs
            save_capture(f"block_{index:02d}_{name}_vid", output[0])
            save_capture(f"block_{index:02d}_{name}_txt", output[1])
        return hook

    def make_attention_hook(index: int):
        """保存 RoPE 后真正送入 varlen attention 的张量和分段边界。"""
        def hook(module, inputs, kwargs, output):
            del module, inputs
            prefix = f"block_{index:02d}_attention"
            for name in ("q", "k", "v", "cu_seqlens_q", "cu_seqlens_k"):
                save_capture(f"{prefix}_{name}", kwargs[name])
            save_capture(f"{prefix}_output", output)
        return hook

    def patch_input_hook(module, inputs, output):
        del module, inputs
        save_capture("patch_in_output_vid", output[0])
        save_capture("patch_in_output_shape", output[1])

    def text_input_hook(module, inputs, output):
        del module, inputs
        save_capture("text_in_output", output)

    def time_embedding_hook(module, inputs, output):
        del module, inputs
        save_capture("time_embedding", output)

    def output_ada_pre_hook(module, inputs, kwargs):
        """在原地 mul/add 前克隆输入，并保存 Ada 实际选择的调制向量。"""
        save_capture("output_ada_input_vid", inputs[0].clone())
        embedding = kwargs["emb"]
        save_capture("output_ada_input_embedding", embedding)
        layer_index = module.layers.index("out")
        modulation = rearrange(
            embedding,
            "b (d l g) -> b d l g",
            l=len(module.layers),
            g=3,
        )[..., layer_index, :]
        save_capture("output_ada_modulation", modulation)

    def output_ada_hook(module, inputs, kwargs, output):
        del module, inputs, kwargs
        save_capture("output_ada_output_vid", output)

    def patch_output_hook(module, inputs, output):
        del module, inputs
        save_capture("patch_out_output_vid", output[0])
        save_capture("patch_out_output_shape", output[1])

    hook_handles.append(runner.dit.vid_in.register_forward_hook(patch_input_hook))
    hook_handles.append(runner.dit.txt_in.register_forward_hook(text_input_hook))
    hook_handles.append(runner.dit.emb_in.register_forward_hook(time_embedding_hook))
    hook_handles.append(
        runner.dit.vid_out_ada.register_forward_pre_hook(output_ada_pre_hook, with_kwargs=True)
    )
    hook_handles.append(
        runner.dit.vid_out_ada.register_forward_hook(output_ada_hook, with_kwargs=True)
    )
    hook_handles.append(runner.dit.vid_out.register_forward_hook(patch_output_hook))

    for idx, block in enumerate(runner.dit.blocks):
        if idx in block_dump_ids:
            hook_handles.append(block.register_forward_hook(make_block_hook(idx), with_kwargs=True))
        if idx in args.deep_blocks:
            hook_handles.append(block.attn.proj_qkv.register_forward_hook(make_mm_hook(idx, "qkv")))
            hook_handles.append(block.attn.norm_q.register_forward_hook(make_mm_hook(idx, "norm_q")))
            hook_handles.append(block.attn.norm_k.register_forward_hook(make_mm_hook(idx, "norm_k")))
            hook_handles.append(block.attn.attn.register_forward_hook(make_attention_hook(idx), with_kwargs=True))
            hook_handles.append(block.attn.proj_out.register_forward_hook(make_mm_hook(idx, "attention_projected")))

    timesteps = runner.sampler.timesteps.timesteps
    x = latents
    with torch.no_grad(), torch.autocast(
        "cuda", torch.bfloat16, enabled=args.dit_dtype == "bfloat16"
    ):
        for step_index, t in enumerate(timesteps):
            dit_input = torch.cat([x, latents_cond], dim=-1).to(dit_dtype)
            text_pos_input = text_pos_flat.to(dit_dtype)
            text_neg_input = text_neg_flat.to(dit_dtype)
            save_tensor(out_dir / f"dit_input_pos_step{step_index:03d}.pt", dit_input, metadata)
            capture_context["tag"] = f"pos_step{step_index:03d}"
            pos = runner.dit(
                vid=dit_input,
                txt=text_pos_input,
                vid_shape=latents_shape,
                txt_shape=text_pos_shape,
                timestep=t.repeat(1),
            ).vid_sample
            save_tensor(out_dir / f"dit_output_pos_step{step_index:03d}.pt", pos, metadata)

            capture_context["tag"] = f"neg_step{step_index:03d}"
            neg = runner.dit(
                vid=dit_input,
                txt=text_neg_input,
                vid_shape=latents_shape,
                txt_shape=text_neg_shape,
                timestep=t.repeat(1),
            ).vid_sample
            save_tensor(out_dir / f"dit_output_neg_step{step_index:03d}.pt", neg, metadata)
            capture_context["tag"] = "inactive"

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
