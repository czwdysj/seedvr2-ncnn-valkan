#!/usr/bin/env python3
"""Export one shape-dynamic SeedVR2 VAE encoder/decoder for ncnn.

The graph accepts batch-one ``B,C,T,H,W`` tensors with variable ``T,H,W``.
Runtime-shape loops remain explicit module operators for custom ncnn layers;
convolutions, residual paths, activations, and causal padding stay native ncnn.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from einops import rearrange


class DynamicFramewiseGroupNorm(nn.Module):
    """Apply one GroupNorm independently to every frame of a 5D video."""

    def __init__(self, source: nn.GroupNorm) -> None:
        super().__init__()
        self.num_groups = source.num_groups
        self.num_channels = source.num_channels
        self.eps = source.eps
        if source.affine:
            self.weight = nn.Parameter(source.weight.detach().clone())
            self.bias = nn.Parameter(source.bias.detach().clone())
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        if sample.ndim != 5:
            return F.group_norm(sample, self.num_groups, self.weight, self.bias, self.eps)
        batch, channels, frames, height, width = sample.shape
        # Reference GroupNorm statistics are independent for every video frame.
        flat = sample.permute(0, 2, 1, 3, 4).reshape(
            batch * frames, channels, height, width
        )
        flat = F.group_norm(flat, self.num_groups, self.weight, self.bias, self.eps)
        return flat.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)


class DynamicFramewiseSpatialAttention(nn.Module):
    """Run VAE spatial self-attention independently on runtime video frames."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.attention = source

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        batch, channels, frames, height, width = sample.shape
        flat = sample.permute(0, 2, 1, 3, 4).reshape(
            batch * frames, channels, height, width
        )
        flat = self.attention(flat)
        return flat.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)


class DynamicSpaceTimeShuffle(nn.Module):
    """Project channels and perform runtime space-time pixel shuffle."""

    def __init__(self, source: nn.Conv3d, temporal_ratio: int, spatial_ratio: int) -> None:
        super().__init__()
        self.projection = source
        self.temporal_ratio = temporal_ratio
        self.spatial_ratio = spatial_ratio

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.projection(sample)
        sample = rearrange(
            sample,
            "b (x y z c) f h w -> b c (f z) (h x) (w y)",
            x=self.spatial_ratio,
            y=self.spatial_ratio,
            z=self.temporal_ratio,
        )
        if self.temporal_ratio > 1:
            # Decoder temporal upsampling duplicates the first decoded frame;
            # remove that duplicate inside the dynamic boundary so output T is
            # ``2*T-1`` without relying on a fixed-shape pnnx Crop.
            sample = torch.cat([sample[:, :, :1], sample[:, :, 2:]], dim=2)
        return sample


class VaeEncoderExport(nn.Module):
    """Expose deterministic posterior moments for a dynamic video tensor."""

    def __init__(self, vae: nn.Module, memory_state) -> None:
        super().__init__()
        self.encoder = vae.encoder
        self.memory_state = memory_state

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.encoder(sample, memory_state=self.memory_state)


class VaeDecoderExport(nn.Module):
    """Decode an unscaled dynamic latent tensor into a video tensor."""

    def __init__(self, vae: nn.Module, memory_state) -> None:
        super().__init__()
        self.decoder = vae.decoder
        self.memory_state = memory_state

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent, memory_state=self.memory_state)


def parse_shape(value: str) -> tuple[int, ...]:
    shape = tuple(int(part) for part in value.split(","))
    if len(shape) != 5 or shape[0] != 1:
        raise argparse.ArgumentTypeError("shape must be B,C,T,H,W with B=1")
    return shape


def find_pnnx(explicit: Path | None) -> Path:
    candidates = [
        explicit,
        Path(os.environ["PNNX"]) if os.environ.get("PNNX") else None,
        Path(shutil.which("pnnx")) if shutil.which("pnnx") else None,
        Path("/home/czw1/ncnn_learn/Penguin-VL-ncnn/.venv/bin/pnnx"),
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("pnnx was not found; pass --pnnx /path/to/pnnx")


def replace_group_norms(module: nn.Module) -> None:
    """Replace every GroupNorm while preserving checkpoint values and names."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.GroupNorm):
            setattr(module, name, DynamicFramewiseGroupNorm(child))
        else:
            replace_group_norms(child)


def load_dynamic_vae(project_dir: Path, checkpoint: Path) -> tuple[nn.Module, object]:
    sys.path.insert(0, str(project_dir))
    from models.video_vae_v3.modules import attn_video_vae as vae_module
    from models.video_vae_v3.modules.attn_video_vae import (
        UNetMidBlock3D,
        Upsample3D,
        VideoAutoencoderKLWrapper,
    )
    from models.video_vae_v3.modules.causal_inflation_lib import InflatedCausalConv3d
    from models.video_vae_v3.modules.types import MemoryState

    def whole_clip_causal_forward(self, sample, memory_state=MemoryState.DISABLED):
        del memory_state
        if self.temporal_padding:
            first = sample[:, :, :1]
            sample = torch.cat([first] * (self.temporal_padding * 2) + [sample], dim=2)
        return F.conv3d(
            sample,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    InflatedCausalConv3d.forward = whole_clip_causal_forward
    vae_module.causal_norm_wrapper = lambda norm_layer, sample: norm_layer(sample)

    def dynamic_mid_block_forward(
        self, hidden_states, temb=None, memory_state=MemoryState.DISABLED
    ):
        hidden_states = self.resnets[0](hidden_states, temb, memory_state=memory_state)
        for attention, resnet in zip(self.attentions, self.resnets[1:]):
            if attention is not None:
                hidden_states = attention(hidden_states)
            hidden_states = resnet(hidden_states, temb, memory_state=memory_state)
        return hidden_states

    UNetMidBlock3D.forward = dynamic_mid_block_forward

    def dynamic_upsample_forward(
        self,
        hidden_states,
        output_size=None,
        memory_state=MemoryState.DISABLED,
        **kwargs,
    ):
        del output_size, kwargs
        if self.slicing:
            raise RuntimeError("dynamic ncnn export does not support sliced upsampling")
        hidden_states = self.upscale_conv(hidden_states)
        if self.use_conv:
            conv = self.conv if self.name == "conv" else self.Conv2d_0
            hidden_states = conv(hidden_states, memory_state=memory_state)
        return hidden_states

    Upsample3D.forward = dynamic_upsample_forward

    config_path = project_dir / "models/video_vae_v3/s8_c16_t4_inflation_sd3.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config.pop("__object__")
    config["freeze_encoder"] = False
    vae = VideoAutoencoderKLWrapper(**config)
    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    loading = vae.load_state_dict(state, strict=False)
    if loading.missing_keys or loading.unexpected_keys:
        raise RuntimeError(
            f"VAE checkpoint mismatch: missing={loading.missing_keys}, "
            f"unexpected={loading.unexpected_keys}"
        )
    vae.requires_grad_(False).eval().float()

    replace_group_norms(vae)
    for mid_block in [vae.encoder.mid_block, vae.decoder.mid_block]:
        for index, attention in enumerate(mid_block.attentions):
            if attention is not None:
                mid_block.attentions[index] = DynamicFramewiseSpatialAttention(attention)
    for layer in vae.decoder.modules():
        if isinstance(layer, Upsample3D):
            layer.upscale_conv = DynamicSpaceTimeShuffle(
                layer.upscale_conv, layer.temporal_ratio, layer.spatial_ratio
            )
    return vae, MemoryState.DISABLED


def assert_two_shape_eager(module: nn.Module, shapes: tuple[tuple[int, ...], ...]) -> None:
    """Catch fixed-shape Python behavior before generating TorchScript."""
    with torch.inference_mode():
        for shape in shapes:
            output = module(torch.randn(shape, dtype=torch.float32))
            if not torch.isfinite(output).all():
                raise RuntimeError(f"non-finite PyTorch output for shape {shape}")
            print(f"eager {shape} -> {tuple(output.shape)}")


def trace_module(module: nn.Module, shape: tuple[int, ...], output_path: Path) -> None:
    torch.manual_seed(0)
    example = torch.randn(shape, dtype=torch.float32)
    with torch.inference_mode():
        expected = module(example)
        traced = torch.jit.trace(module, example, strict=True)
        actual = traced(example)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    traced.save(str(output_path))


def run_pnnx(
    pnnx: Path,
    torchscript: Path,
    shape1: tuple[int, ...],
    shape2: tuple[int, ...],
    output_dir: Path,
) -> tuple[Path, Path]:
    stem = torchscript.stem
    ncnn_param = output_dir / f"{stem}.ncnn.param"
    ncnn_bin = output_dir / f"{stem}.ncnn.bin"
    command = [
        str(pnnx),
        str(torchscript),
        f"inputshape=[{','.join(map(str, shape1))}]f32",
        f"inputshape2=[{','.join(map(str, shape2))}]f32",
        "fp16=0",
        "optlevel=2",
        "moduleop=DynamicFramewiseGroupNorm,DynamicFramewiseSpatialAttention,DynamicSpaceTimeShuffle",
        f"pnnxparam={output_dir / (stem + '.pnnx.param')}",
        f"pnnxbin={output_dir / (stem + '.pnnx.bin')}",
        f"pnnxpy={output_dir / (stem + '_pnnx.py')}",
        f"pnnxonnx={output_dir / (stem + '.pnnx.onnx')}",
        f"ncnnparam={ncnn_param}",
        f"ncnnbin={ncnn_bin}",
        f"ncnnpy={output_dir / (stem + '_ncnn.py')}",
    ]
    subprocess.run(command, cwd=output_dir, check=True)
    return ncnn_param, ncnn_bin


def inspect_dynamic_param(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    required = ("DynamicFramewiseGroupNorm", "DynamicFramewiseSpatialAttention")
    missing = [name for name in required if name not in text]
    if missing:
        raise RuntimeError(f"dynamic module operators missing from {path}: {missing}")
    unsupported = [
        line for line in text.splitlines() if line.startswith(("torch.", "aten::"))
    ]
    if unsupported:
        raise RuntimeError(f"unsupported operators remain in {path}: {unsupported[:3]}")


def main() -> int:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--project-dir", type=Path, default=root / "pytorch_model")
    parser.add_argument(
        "--checkpoint", type=Path, default=root / "weights/seedvr2_3b/ema_vae.pth"
    )
    parser.add_argument("--output-dir", type=Path, default=root / "ncnn_models/vae_dynamic")
    parser.add_argument("--pnnx", type=Path)
    parser.add_argument("--component", choices=("encoder", "decoder", "all"), default="all")
    parser.add_argument("--trace-only", action="store_true")
    parser.add_argument("--encoder-shape1", type=parse_shape, default=(1, 3, 5, 32, 32))
    parser.add_argument("--encoder-shape2", type=parse_shape, default=(1, 3, 9, 48, 64))
    parser.add_argument("--decoder-shape1", type=parse_shape, default=(1, 16, 2, 4, 4))
    parser.add_argument("--decoder-shape2", type=parse_shape, default=(1, 16, 3, 6, 8))
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pnnx = None if args.trace_only else find_pnnx(args.pnnx)
    vae, memory_state = load_dynamic_vae(args.project_dir.resolve(), args.checkpoint.resolve())

    jobs = []
    if args.component in ("encoder", "all"):
        jobs.append(
            (
                "seedvr2_vae_encoder_dynamic",
                VaeEncoderExport(vae, memory_state),
                args.encoder_shape1,
                args.encoder_shape2,
            )
        )
    if args.component in ("decoder", "all"):
        jobs.append(
            (
                "seedvr2_vae_decoder_dynamic",
                VaeDecoderExport(vae, memory_state),
                args.decoder_shape1,
                args.decoder_shape2,
            )
        )

    for name, module, shape1, shape2 in jobs:
        assert_two_shape_eager(module, (shape1, shape2))
        torchscript = output_dir / f"{name}.pt"
        trace_module(module, shape1, torchscript)
        if pnnx is not None:
            param, binary = run_pnnx(pnnx, torchscript, shape1, shape2, output_dir)
            inspect_dynamic_param(param)
            print(f"generated {param} ({param.stat().st_size} bytes)")
            print(f"generated {binary} ({binary.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
