#!/usr/bin/env python3
"""诊断 SeedVR2 window attention 的三轴 RoPE 坐标语义。

输入是 ``dump_seedvr2_reference.py`` 保存的单个 block 深度参考张量。脚本从
Q/K Norm 输出出发，分别重建局部整数坐标、错误的文本长度时间偏移和 pixel
``[-1,1]`` 坐标三种候选，再与 PyTorch 在 BF16 attention 边界捕获的 Q/K 比较。
它只用于定位参考环境所采用的 RoPE 版本，不加载完整模型，也不修改参考文件。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


def metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    """使用 FP64 累加计算候选张量和参考张量的误差。"""
    actual = actual.double().reshape(-1)
    expected = expected.double().reshape(-1)
    difference = actual - expected
    reference_rms = expected.square().mean().sqrt().clamp_min(1e-12)
    return {
        "max_abs": difference.abs().max().item(),
        "nrmse": (difference.square().mean().sqrt() / reference_rms).item(),
        "cosine": torch.nn.functional.cosine_similarity(actual, expected, dim=0).item(),
    }


def apply_rope(value: torch.Tensor, positions: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """按当前 NCNN 权重中的 21 对/轴频率重建 adjacent-pair RoPE。"""
    output = value.clone()
    frequencies = 1.0 / (
        10000.0 ** (torch.arange(0, 42, 2, dtype=torch.float32) / 42.0)
    )
    for axis, position in enumerate(positions):
        begin = axis * 42
        first = value[..., begin : begin + 42 : 2]
        second = value[..., begin + 1 : begin + 42 : 2]
        angle = position[:, None, None] * frequencies[None, None, :]
        cosine = angle.cos()
        sine = angle.sin()
        output[..., begin : begin + 42 : 2] = first * cosine - second * sine
        output[..., begin + 1 : begin + 42 : 2] = second * cosine + first * sine
    return output


def infer_first_token_positions(normalized: torch.Tensor, rotated: torch.Tensor) -> list[list[float]]:
    """由首个 token 各维二维旋转反推出角度/频率，辅助识别坐标原点。"""
    frequencies = 1.0 / (
        10000.0 ** (torch.arange(0, 42, 2, dtype=torch.float64) / 42.0)
    )
    inferred = []
    for axis in range(3):
        begin = axis * 42
        first = normalized[0, 0, begin : begin + 42 : 2].double()
        second = normalized[0, 0, begin + 1 : begin + 42 : 2].double()
        out_first = rotated[0, 0, begin : begin + 42 : 2].double()
        out_second = rotated[0, 0, begin + 1 : begin + 42 : 2].double()
        angle = torch.atan2(first * out_second - second * out_first,
                            first * out_first + second * out_second)
        inferred.append((angle / frequencies).tolist())
    return inferred


def make_video_positions(
    frames: int,
    height: int,
    width: int,
    mode: str,
    text_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """生成与 block 0 普通窗口展平顺序一致的候选坐标。"""
    positions_t: list[float] = []
    positions_h: list[float] = []
    positions_w: list[float] = []
    window_frames = math.ceil(min(frames, 30) / 4)
    scale = math.sqrt(3600.0 / (height * width))
    window_height = math.ceil(round(height * scale) / 3)
    window_width = math.ceil(round(width * scale) / 3)
    for w0 in range(0, width, window_width):
        for h0 in range(0, height, window_height):
            for t0 in range(0, frames, window_frames):
                local_t = min(window_frames, frames - t0)
                local_h = min(window_height, height - h0)
                local_w = min(window_width, width - w0)
                for t in range(local_t):
                    for y in range(local_h):
                        for x in range(local_w):
                            if mode == "pixel":
                                positions_t.append(-1.0 if local_t == 1 else -1.0 + 2.0 * t / (local_t - 1))
                                positions_h.append(-1.0 if local_h == 1 else -1.0 + 2.0 * y / (local_h - 1))
                                positions_w.append(-1.0 if local_w == 1 else -1.0 + 2.0 * x / (local_w - 1))
                            else:
                                positions_t.append(float(t + (text_length if mode == "text_offset" else 0)))
                                positions_h.append(float(y))
                                positions_w.append(float(x))
    return tuple(torch.tensor(item) for item in (positions_t, positions_h, positions_w))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--branch", choices=("pos", "neg"), default="pos")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    prefix = f"{args.branch}_step000_block_{args.block:02d}"
    load = lambda name: torch.load(
        args.reference_dir / f"{prefix}_{name}.pt",
        map_location="cpu",
        weights_only=True,
    ).float()
    shape = torch.load(
        args.reference_dir / f"{args.branch}_step000_patch_in_output_shape.pt",
        map_location="cpu",
        weights_only=True,
    ).reshape(-1).tolist()
    text_length = load("norm_q_txt").shape[0]
    expected = {kind: load(f"attention_{kind}") for kind in ("q", "k")}
    normalized = {
        kind: {stream: load(f"norm_{kind}_{stream}") for stream in ("vid", "txt")}
        for kind in ("q", "k")
    }

    cu_seqlens = load("attention_cu_seqlens_q").to(torch.int64).reshape(-1).tolist()
    report: dict[str, object] = {
        "shape": shape,
        "text_length": text_length,
        "cu_seqlens": cu_seqlens,
        "candidates": {},
    }
    report["first_token_inferred_positions"] = {
        kind: infer_first_token_positions(normalized[kind]["vid"], expected[kind])
        for kind in ("q", "k")
    }
    window_lengths = [end - begin - text_length for begin, end in zip(cu_seqlens, cu_seqlens[1:])]
    report["text_without_rope"] = {}
    for kind in ("q", "k"):
        text_rows = torch.cat(
            [
                expected[kind][
                    cu_seqlens[window] + window_lengths[window] : cu_seqlens[window + 1]
                ]
                for window in range(len(window_lengths))
            ],
            dim=0,
        )
        report["text_without_rope"][kind] = metrics(
            text_rows,
            normalized[kind]["txt"].to(torch.bfloat16).float().repeat(len(window_lengths), 1, 1),
        )
    for mode in ("local_integer", "text_offset", "mmrope3d", "pixel"):
        positions = make_video_positions(
            *shape, "text_offset" if mode == "mmrope3d" else mode, text_length
        )
        report["candidates"][mode] = {}
        for kind in ("q", "k"):
            video = apply_rope(normalized[kind]["vid"], positions).to(torch.bfloat16).float()
            # block 0 的窗口时间长度为 1；每个窗口视频段后重复拼接同一文本段。
            text = normalized[kind]["txt"]
            if mode == "mmrope3d":
                text_position = torch.arange(text_length, dtype=torch.float32)
                text = apply_rope(text, (text_position, text_position, text_position))
            text = text.to(torch.bfloat16).float()
            rows = []
            begin = 0
            for window_video_length in window_lengths:
                rows.extend((
                    video[begin : begin + window_video_length],
                    text,
                ))
                begin += window_video_length
            candidate = torch.cat(rows, dim=0)
            report["candidates"][mode][kind] = metrics(candidate, expected[kind])

    output = json.dumps(report, ensure_ascii=False, indent=2)
    print(output)
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
