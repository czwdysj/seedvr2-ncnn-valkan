#!/usr/bin/env python3
"""验证 SeedVR2 3B DiT 的 NCNN CPU/Vulkan 转换结果。

本脚本使用 ``dump_seedvr2_reference.py --dit-dtype float32`` 生成的参考目录，
逐一执行输入头、32 个 Transformer block、输出头和完整流式 DiT。每个独立
block 都使用 PyTorch 在同一位置保存的输入，避免上游误差掩盖当前层问题；
完整 DiT 另行记录累积误差。``--full-only`` 可只同步体积较小的端到端参考张量，
适合在远程 Vulkan 主机快速复测。测试结果写为 JSON，任一启用的检查超过阈值
都会返回失败，避免只记录完整 DiT 误差却未形成回归门禁。
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch


def save_raw(path: Path, tensor: torch.Tensor) -> np.ndarray:
    """将参考 tensor 以连续 FP32 写出，并返回对应 NumPy 数组。"""
    value = tensor.detach().float().contiguous().cpu().numpy()
    value.tofile(path)
    return value


def read_matrix(path: Path) -> np.ndarray:
    """读取 runner 使用的 int32[rows,cols] + FP32 矩阵格式。"""
    with path.open("rb") as stream:
        rows, columns = struct.unpack("ii", stream.read(8))
        return np.fromfile(stream, dtype=np.float32).reshape(rows, columns)


def metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    actual = actual.reshape(-1)
    expected = expected.reshape(-1)
    if actual.shape != expected.shape:
        raise ValueError(f"张量元素数不一致: actual={actual.shape} expected={expected.shape}")
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    rmse = float(np.sqrt(np.mean(difference * difference)))
    reference_rms = max(float(np.sqrt(np.mean(expected.astype(np.float64) ** 2))), 1e-12)
    denominator = max(
        float(np.sqrt(np.sum(actual.astype(np.float64) ** 2)
                      * np.sum(expected.astype(np.float64) ** 2))),
        1e-12,
    )
    return {
        "max_abs": float(np.abs(difference).max()),
        "mean_abs": float(np.abs(difference).mean()),
        "rmse": rmse,
        "nrmse": rmse / reference_rms,
        "cosine": float(np.sum(actual.astype(np.float64) * expected) / denominator),
    }


def run(command: list[str]) -> None:
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)


def parse_block_indices(value: str) -> list[int]:
    """解析逗号分隔的 block 编号，并拒绝超出 32 层范围的输入。"""
    indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not indices or any(index < 0 or index >= 32 for index in indices):
        raise argparse.ArgumentTypeError("block 编号必须位于 0..31")
    return indices


def validate_case(args: argparse.Namespace, reference: Path, temporary: Path) -> dict:
    condition_branch = args.branch
    shape = torch.load(
        reference / f"{condition_branch}_step000_patch_in_output_shape.pt",
        map_location="cpu",
        weights_only=True,
    ).reshape(-1).tolist()
    result: dict[str, object] = {"reference": str(reference), "shape": shape, "blocks": []}

    embedding = None
    if not args.full_only:
        embedding = save_raw(
            temporary / "embedding.f32",
            torch.load(
                reference / f"{condition_branch}_step000_time_embedding.pt",
                map_location="cpu",
                weights_only=True,
            ),
        )

    for index in (() if args.full_only else args.blocks):
        inputs = {}
        for branch in ("vid", "txt"):
            inputs[branch] = save_raw(
                temporary / f"{branch}.f32",
                torch.load(
                    reference / f"{args.branch}_step000_block_{index:02d}_input_{branch}.pt",
                    map_location="cpu",
                    weights_only=True,
                ),
            )
        stem = f"seedvr2_dit_block_{index:02d}.ncnn"
        command = [
                str(args.block_runner),
                str(args.model_dir / f"{stem}.param"),
                str(args.model_dir / f"{stem}.bin"),
                str(temporary / "vid.f32"),
                str(inputs["vid"].shape[0]),
                str(temporary / "txt.f32"),
                str(inputs["txt"].shape[0]),
                str(temporary / "embedding.f32"),
                *map(str, shape),
                str(temporary / "vid_out.f32"),
                str(temporary / "txt_out.f32"),
            ]
        debug_prefix = temporary / f"block_{index:02d}"
        if args.capture_intermediates:
            command.append(str(debug_prefix))
        run(command)
        item: dict[str, object] = {"index": index}
        for branch in ("vid", "txt"):
            expected = torch.load(
                reference / f"{args.branch}_step000_block_{index:02d}_output_{branch}.pt",
                map_location="cpu",
                weights_only=True,
            ).float().numpy()
            item[branch] = metrics(read_matrix(temporary / f"{branch}_out.f32"), expected)
            if args.block_output_dir:
                args.block_output_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(
                    temporary / f"{branch}_out.f32",
                    args.block_output_dir
                    / f"{reference.name}_{args.branch}_block_{index:02d}_{branch}.f32",
                )
        item["passed"] = all(item[branch]["nrmse"] <= args.block_nrmse for branch in ("vid", "txt"))
        if args.capture_intermediates:
            reference_names = {
                "qkv_vid": "qkv_vid",
                "qkv_txt": "qkv_txt",
                "norm_q_vid": "norm_q_vid",
                "norm_q_txt": "norm_q_txt",
                "norm_k_vid": "norm_k_vid",
                "norm_k_txt": "norm_k_txt",
                "attention_q": "attention_q",
                "attention_k": "attention_k",
                "attention_v": "attention_v",
                "attention_output": "attention_output",
                "projected_vid": "attention_projected_vid",
                "projected_txt": "attention_projected_txt",
            }
            item["intermediates"] = {}
            for capture_name, reference_name in reference_names.items():
                expected = torch.load(
                    reference
                    / f"{args.branch}_step000_block_{index:02d}_{reference_name}.pt",
                    map_location="cpu",
                    weights_only=True,
                ).float().numpy()
                item["intermediates"][capture_name] = metrics(
                    read_matrix(Path(f"{debug_prefix}_{capture_name}.f32")), expected
                )
        result["blocks"].append(item)
        print(
            f"{reference.name} block {index:02d}: "
            f"vid={item['vid']['nrmse']:.6g} txt={item['txt']['nrmse']:.6g}"
        )

    if args.blocks_only:
        result["passed"] = all(item["passed"] for item in result["blocks"])
        if embedding is not None:
            result["embedding_shape"] = list(embedding.shape)
        return result

    # 完整 DiT 使用原始 33 通道 token 与预计算文本 embedding，验证 32 层可串联。
    video = save_raw(
        temporary / "full_vid.f32",
        torch.load(reference / "dit_input_pos_step000.pt", map_location="cpu", weights_only=True),
    )
    text = save_raw(
        temporary / "full_txt.f32",
        torch.load(
            reference / f"text_{condition_branch}_emb.pt",
            map_location="cpu",
            weights_only=True,
        ),
    )
    original_shape = [shape[0], shape[1] * 2, shape[2] * 2]
    run(
        [
            str(args.full_runner),
            str(args.model_dir),
            str(args.model_dir),
            str(temporary / "full_vid.f32"),
            str(video.shape[0]),
            str(temporary / "full_txt.f32"),
            str(text.shape[0]),
            str(args.timestep),
            *map(str, original_shape),
            str(temporary / "full_output.f32"),
        ]
    )
    expected = torch.load(
        reference / f"dit_output_{condition_branch}_step000.pt",
        map_location="cpu",
        weights_only=True,
    ).float().numpy()
    result["full"] = metrics(read_matrix(temporary / "full_output.f32"), expected)
    if args.full_output_dir:
        args.full_output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            temporary / "full_output.f32",
            args.full_output_dir / f"{reference.name}_{condition_branch}.f32",
        )
    result["full_passed"] = (
        result["full"]["nrmse"] <= args.full_nrmse
        and result["full"]["cosine"] >= args.full_cosine
    )
    result["passed"] = (
        all(item["passed"] for item in result["blocks"])
        and result["full_passed"]
    )
    if embedding is not None:
        result["embedding_shape"] = list(embedding.shape)
    print(
        f"{reference.name} full: nrmse={result['full']['nrmse']:.6g} "
        f"cosine={result['full']['cosine']:.9g}"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, action="append", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--block-runner", type=Path, required=True)
    parser.add_argument("--full-runner", type=Path, required=True)
    parser.add_argument("--block-nrmse", type=float, default=0.01)
    parser.add_argument("--full-nrmse", type=float, default=0.002)
    parser.add_argument("--full-cosine", type=float, default=0.999999)
    parser.add_argument("--full-only", action="store_true")
    parser.add_argument("--blocks-only", action="store_true")
    parser.add_argument("--blocks", type=parse_block_indices, default=list(range(32)))
    parser.add_argument("--branch", choices=("pos", "neg"), default="pos")
    parser.add_argument("--capture-intermediates", action="store_true")
    parser.add_argument("--timestep", type=float, default=1000.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-output-dir", type=Path)
    parser.add_argument("--block-output-dir", type=Path)
    args = parser.parse_args()
    if args.full_only and args.blocks_only:
        parser.error("--full-only 与 --blocks-only 不能同时使用")

    report = {
        "block_nrmse_threshold": args.block_nrmse,
        "full_nrmse_threshold": args.full_nrmse,
        "full_cosine_threshold": args.full_cosine,
        "full_only": args.full_only,
        "blocks_only": args.blocks_only,
        "blocks": args.blocks,
        "branch": args.branch,
        "capture_intermediates": args.capture_intermediates,
        "cases": [],
    }
    with tempfile.TemporaryDirectory(prefix="seedvr2_dit_ncnn_") as directory:
        temporary = Path(directory)
        for reference in args.reference_dir:
            report["cases"].append(validate_case(args, reference.resolve(), temporary))
    report["passed"] = all(case["passed"] for case in report["cases"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.output}: {'PASS' if report['passed'] else 'FAIL'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
