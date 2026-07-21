#!/usr/bin/env python3
"""Compare dynamic SeedVR2 VAE ncnn C++ inference with PyTorch.

Two random batch-one videos with different ``T,H,W`` share the same dynamic
encoder/decoder model files. The test compares encoder moments and decoder
outputs and fails when FP32 max absolute error exceeds the requested tolerance.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch


def load_export_module(path: Path):
    spec = importlib.util.spec_from_file_location("seedvr2_vae_dynamic_export", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run_cpp(
    runner: Path,
    param: Path,
    binary: Path,
    sample: torch.Tensor,
    temporary_dir: Path,
    tag: str,
) -> np.ndarray:
    """Run one C,T,H,W tensor through the final C++ ncnn runtime."""
    input_path = temporary_dir / f"{tag}_input.f32"
    output_path = temporary_dir / f"{tag}_output.f32"
    array = np.ascontiguousarray(sample[0].cpu().numpy(), dtype=np.float32)
    array.tofile(input_path)
    _, channels, frames, height, width = sample.shape
    subprocess.run(
        [
            str(runner),
            str(param),
            str(binary),
            str(channels),
            str(frames),
            str(height),
            str(width),
            str(input_path),
            str(output_path),
            "4",
        ],
        check=True,
    )
    with output_path.open("rb") as stream:
        output_shape = tuple(np.fromfile(stream, dtype=np.int32, count=4).tolist())
        output = np.fromfile(stream, dtype=np.float32)
    return output.reshape(output_shape)


def report(name: str, actual: np.ndarray, expected: np.ndarray, tolerance: float) -> None:
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    max_abs = float(np.abs(difference).max())
    mean_abs = float(np.abs(difference).mean())
    rmse = float(np.sqrt(np.mean(difference * difference)))
    print(
        f"{name}: shape={actual.shape} max_abs={max_abs:.9g} "
        f"mean_abs={mean_abs:.9g} rmse={rmse:.9g}"
    )
    if not np.isfinite(actual).all() or max_abs > tolerance:
        raise AssertionError(f"{name} failed tolerance {tolerance}: max_abs={max_abs}")


def main() -> int:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--runner",
        type=Path,
        default=root / "cpp_runtime/build/seedvr2_vae_dynamic_runner",
    )
    parser.add_argument("--model-dir", type=Path, default=root / "ncnn_models/vae_dynamic")
    parser.add_argument("--max-abs", type=float, default=5e-4)
    args = parser.parse_args()

    exporter = load_export_module(root / "tools/export_seedvr2_vae_dynamic_ncnn.py")
    vae, memory_state = exporter.load_dynamic_vae(
        root / "pytorch_model", root / "weights/seedvr2_3b/ema_vae.pth"
    )
    encoder = exporter.VaeEncoderExport(vae, memory_state)
    decoder = exporter.VaeDecoderExport(vae, memory_state)
    encoder_param = args.model_dir / "seedvr2_vae_encoder_dynamic.ncnn.param"
    encoder_bin = args.model_dir / "seedvr2_vae_encoder_dynamic.ncnn.bin"
    decoder_param = args.model_dir / "seedvr2_vae_decoder_dynamic.ncnn.param"
    decoder_bin = args.model_dir / "seedvr2_vae_decoder_dynamic.ncnn.bin"

    # These odd sizes were not used by pnnx calibration and intentionally do
    # not obey round-trip alignment, proving the encoder accepts arbitrary T,H,W.
    cases = ((1, 3, 6, 37, 53), (1, 3, 10, 41, 67))
    with tempfile.TemporaryDirectory(prefix="seedvr2_vae_dynamic_") as temporary:
        temporary_dir = Path(temporary)
        for index, shape in enumerate(cases):
            torch.manual_seed(1000 + index)
            video = torch.randn(shape, dtype=torch.float32)
            with torch.inference_mode():
                expected_moments = encoder(video)
                # Deterministic posterior mean is an unscaled valid decoder input.
                latent = expected_moments[:, :16].contiguous()
                expected_video = decoder(latent)

            actual_moments = run_cpp(
                args.runner,
                encoder_param,
                encoder_bin,
                video,
                temporary_dir,
                f"case{index}_encoder",
            )
            actual_video = run_cpp(
                args.runner,
                decoder_param,
                decoder_bin,
                latent,
                temporary_dir,
                f"case{index}_decoder",
            )
            report(
                f"case{index} encoder",
                actual_moments,
                expected_moments[0].numpy(),
                args.max_abs,
            )
            report(
                f"case{index} decoder",
                actual_video,
                expected_video[0].numpy(),
                args.max_abs,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
