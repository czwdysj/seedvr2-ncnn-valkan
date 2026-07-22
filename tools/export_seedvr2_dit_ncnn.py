#!/usr/bin/env python3
"""把 SeedVR2 3B DiT checkpoint 流式导出为分块 NCNN param/bin。

本文件不实例化 3B PyTorch 模型，也不把约 13.6 GB checkpoint 整体映射到
内存。它借助 FakeTensorMode 读取 tensor 元数据和 checkpoint 文件偏移，再按
输入模块、32 个 Transformer block、输出模块逐个复制权重。最终模型采用
batch=1、动态视频 T/H/W、预计算文本 embedding；block 文件可由 C++ runtime
逐个加载和释放，适配显存较小的 Vulkan 设备。普通矩阵计算仍由 block 自定义
层内部创建的 NCNN 原生 InnerProduct 执行，param 中的自定义层只负责动态调度。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable

import numpy as np
import torch
from torch._subclasses.fake_tensor import FakeTensorMode


FP16_TAG = 0x01306B47
FP32_TAG = 0x00000000
DIM = 2560
HEADS = 20
HEAD_DIM = 128
MLP_HIDDEN = 6912
NORM_EPS = 1e-5
BLOCK_COUNT = 32


@dataclass(frozen=True)
class TensorInfo:
    """描述 checkpoint 内一个连续 tensor 的磁盘位置。"""

    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    offset: int
    numel: int


class CheckpointReader:
    """通过 PyTorch 记录的 checkpoint offset 随机读取单个权重。"""

    def __init__(self, checkpoint: Path) -> None:
        self.checkpoint = checkpoint.resolve()
        with FakeTensorMode():
            state = torch.load(
                self.checkpoint,
                map_location="cpu",
                weights_only=True,
            )
        self.tensors: dict[str, TensorInfo] = {}
        for name, tensor in state.items():
            if not tensor.is_contiguous():
                raise ValueError(f"non-contiguous checkpoint tensor is unsupported: {name}")
            storage = tensor.untyped_storage()
            checkpoint_offset = getattr(storage, "_checkpoint_offset", None)
            if checkpoint_offset is None:
                raise RuntimeError(f"checkpoint offset is unavailable for {name}")
            element_size = tensor.element_size()
            self.tensors[name] = TensorInfo(
                name=name,
                shape=tuple(tensor.shape),
                dtype=tensor.dtype,
                offset=int(checkpoint_offset) + tensor.storage_offset() * element_size,
                numel=tensor.numel(),
            )

    def info(self, name: str) -> TensorInfo:
        try:
            return self.tensors[name]
        except KeyError as error:
            raise KeyError(f"checkpoint tensor is missing: {name}") from error

    def copy_fp32(self, name: str, output: BinaryIO, chunk_bytes: int = 16 << 20) -> int:
        """原样复制 FP32 tensor，避免为大矩阵分配同等大小的 Python buffer。"""
        info = self.info(name)
        if info.dtype != torch.float32:
            raise TypeError(f"expected float32 tensor, got {info.dtype}: {name}")
        remaining = info.numel * 4
        with self.checkpoint.open("rb", buffering=0) as source:
            source.seek(info.offset)
            while remaining:
                data = source.read(min(remaining, chunk_bytes))
                if not data:
                    raise EOFError(f"checkpoint ended while reading {name}")
                output.write(data)
                remaining -= len(data)
        return info.numel * 4

    def copy_fp16(self, name: str, output: BinaryIO, chunk_elements: int = 4 << 20) -> int:
        """分块把 FP32 权重转为 NCNN fp16-storage，并保持四字节对齐。"""
        info = self.info(name)
        if info.dtype != torch.float32:
            raise TypeError(f"expected float32 tensor, got {info.dtype}: {name}")
        remaining = info.numel
        with self.checkpoint.open("rb", buffering=0) as source:
            source.seek(info.offset)
            while remaining:
                count = min(remaining, chunk_elements)
                data = source.read(count * 4)
                if len(data) != count * 4:
                    raise EOFError(f"checkpoint ended while reading {name}")
                output.write(np.frombuffer(data, dtype="<f4").astype("<f2").tobytes())
                remaining -= count
        byte_count = info.numel * 2
        padding = (-byte_count) % 4
        if padding:
            output.write(b"\0" * padding)
        return byte_count + padding


class NcnnBinWriter:
    """按 ModelBinFromDataReader 的读取约定写入 NCNN 权重 blob。"""

    def __init__(self, path: Path, reader: CheckpointReader, storage: str) -> None:
        self.path = path
        self.reader = reader
        self.storage = storage
        self.stream = path.open("wb")
        self.entries: list[dict[str, object]] = []

    def close(self) -> None:
        self.stream.close()

    def tagged_weight(self, name: str) -> None:
        """写入 InnerProduct 使用的 type=0 权重及精度 tag。"""
        if self.storage == "fp16":
            self.stream.write(struct.pack("<I", FP16_TAG))
            byte_count = self.reader.copy_fp16(name, self.stream)
        else:
            self.stream.write(struct.pack("<I", FP32_TAG))
            byte_count = self.reader.copy_fp32(name, self.stream)
        self.entries.append(
            {"name": name, "kind": "tagged_weight", "storage": self.storage, "bytes": byte_count + 4}
        )

    def raw_fp32(self, name: str) -> None:
        """写入 bias、norm 和 Ada 常量；NCNN type=1 blob 不带精度 tag。"""
        byte_count = self.reader.copy_fp32(name, self.stream)
        self.entries.append(
            {"name": name, "kind": "raw_fp32", "storage": "fp32", "bytes": byte_count}
        )


def parse_blocks(value: str) -> list[int]:
    if value == "all":
        return list(range(BLOCK_COUNT))
    if value == "none":
        return []
    result = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if any(index < 0 or index >= BLOCK_COUNT for index in result):
        raise argparse.ArgumentTypeError(
            "blocks must be 'all', 'none', or comma-separated values in [0,31]"
        )
    return result


def write_param(path: Path, layer_type: str, inputs: Iterable[str], outputs: Iterable[str], params: str) -> None:
    """写出带显式 Input 层的最小 NCNN 图。

    NCNN 只有在 param 中出现 Input 层时才会把对应名称注册为可由
    Extractor::input() 绑定的 blob。自定义层声明 bottom 名称本身不会创建
    外部输入，因此每个独立加载的子模型都必须补齐这些 Input 层。
    """
    input_names = list(inputs)
    output_names = list(outputs)
    blob_count = len(set(input_names + output_names))
    custom_line = " ".join(
        [layer_type, path.stem, str(len(input_names)), str(len(output_names))]
        + input_names
        + output_names
        + ([params] if params else [])
    )
    input_lines = [f"Input input_{name} 0 1 {name}" for name in input_names]
    lines = input_lines + [custom_line]
    path.write_text(
        f"7767517\n{len(lines)} {blob_count}\n" + "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def ada_names(prefix: str) -> list[str]:
    return [
        f"{prefix}.attn_shift",
        f"{prefix}.attn_scale",
        f"{prefix}.attn_gate",
        f"{prefix}.mlp_shift",
        f"{prefix}.mlp_scale",
        f"{prefix}.mlp_gate",
    ]


def export_block(reader: CheckpointReader, output_dir: Path, index: int, storage: str) -> dict[str, object]:
    """导出一个可独立加载的 Transformer block。"""
    shared = index >= 10
    is_last = index == BLOCK_COUNT - 1
    shifted = index % 2 == 1
    stem = f"seedvr2_dit_block_{index:02d}"
    param_path = output_dir / f"{stem}.ncnn.param"
    bin_path = output_dir / f"{stem}.ncnn.bin"
    params = (
        f"0={index} 1={int(shared)} 2={int(is_last)} 3={int(shifted)} "
        f"4={DIM} 5={HEADS} 6={HEAD_DIM} 7={MLP_HIDDEN} 8={NORM_EPS}"
    )
    write_param(
        param_path,
        "SeedVR2DiTBlock",
        ("vid", "txt", "emb", "vid_shape"),
        ("vid_out", "txt_out"),
        params,
    )

    writer = NcnnBinWriter(bin_path, reader, storage)
    try:
        branches = ["all"] if shared else ["vid", "txt"]
        for branch in branches:
            base = f"blocks.{index}"
            for name in ada_names(f"{base}.ada.{branch}"):
                writer.raw_fp32(name)
            writer.tagged_weight(f"{base}.attn.proj_qkv.{branch}.weight")
            writer.tagged_weight(f"{base}.attn.proj_out.{branch}.weight")
            writer.raw_fp32(f"{base}.attn.proj_out.{branch}.bias")
            writer.raw_fp32(f"{base}.attn.norm_q.{branch}.weight")
            writer.raw_fp32(f"{base}.attn.norm_k.{branch}.weight")
            writer.tagged_weight(f"{base}.mlp.{branch}.proj_in_gate.weight")
            writer.tagged_weight(f"{base}.mlp.{branch}.proj_in.weight")
            writer.tagged_weight(f"{base}.mlp.{branch}.proj_out.weight")
        writer.raw_fp32(f"blocks.{index}.attn.rope.rope.freqs")
    finally:
        writer.close()
    return {
        "index": index,
        "shared_weights": shared,
        "last_layer": is_last,
        "shifted_window": shifted,
        "param": param_path.name,
        "bin": bin_path.name,
        "bin_bytes": bin_path.stat().st_size,
        "weights": writer.entries,
    }


def export_input(reader: CheckpointReader, output_dir: Path, storage: str) -> dict[str, object]:
    param_path = output_dir / "seedvr2_dit_input.ncnn.param"
    bin_path = output_dir / "seedvr2_dit_input.ncnn.bin"
    write_param(
        param_path,
        "SeedVR2DiTInput",
        ("vid", "txt", "timestep", "vid_shape"),
        ("vid_out", "txt_out", "emb", "patched_shape"),
        f"0={DIM} 1=33 2=5120 3=256 4=15360",
    )
    names = [
        ("vid_in.proj.weight", True),
        ("vid_in.proj.bias", False),
        ("txt_in.weight", True),
        ("txt_in.bias", False),
        ("emb_in.proj_in.weight", True),
        ("emb_in.proj_in.bias", False),
        ("emb_in.proj_hid.weight", True),
        ("emb_in.proj_hid.bias", False),
        ("emb_in.proj_out.weight", True),
        ("emb_in.proj_out.bias", False),
    ]
    writer = NcnnBinWriter(bin_path, reader, storage)
    try:
        for name, tagged in names:
            writer.tagged_weight(name) if tagged else writer.raw_fp32(name)
    finally:
        writer.close()
    return {"param": param_path.name, "bin": bin_path.name, "bin_bytes": bin_path.stat().st_size, "weights": writer.entries}


def export_output(reader: CheckpointReader, output_dir: Path, storage: str) -> dict[str, object]:
    param_path = output_dir / "seedvr2_dit_output.ncnn.param"
    bin_path = output_dir / "seedvr2_dit_output.ncnn.bin"
    write_param(
        param_path,
        "SeedVR2DiTOutput",
        ("vid", "emb", "vid_shape"),
        ("vid_out", "output_shape"),
        f"0={DIM} 1=16 2={NORM_EPS}",
    )
    writer = NcnnBinWriter(bin_path, reader, storage)
    try:
        for name in ("vid_out_norm.weight", "vid_out_ada.out_shift", "vid_out_ada.out_scale"):
            writer.raw_fp32(name)
        writer.tagged_weight("vid_out.proj.weight")
        writer.raw_fp32("vid_out.proj.bias")
    finally:
        writer.close()
    return {"param": param_path.name, "bin": bin_path.name, "bin_bytes": bin_path.stat().st_size, "weights": writer.entries}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=root / "weights/seedvr2_3b/seedvr2_ema_3b.pth",
    )
    parser.add_argument("--output-dir", type=Path, default=root / "ncnn_models/dit")
    parser.add_argument("--storage", choices=("fp32", "fp16"), default="fp16")
    parser.add_argument("--blocks", type=parse_blocks, default=parse_blocks("all"))
    parser.add_argument("--skip-input", action="store_true")
    parser.add_argument("--skip-output", action="store_true")
    parser.add_argument("--skip-sha256", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reader = CheckpointReader(args.checkpoint)
    manifest: dict[str, object] = {
        "format": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "storage": args.storage,
        "dynamic_batch": False,
        "dynamic_thw": True,
        "input": None,
        "blocks": [],
        "output": None,
    }
    if not args.skip_input:
        manifest["input"] = export_input(reader, args.output_dir, args.storage)
    for index in args.blocks:
        print(f"exporting block {index:02d}", flush=True)
        manifest["blocks"].append(export_block(reader, args.output_dir, index, args.storage))
    if not args.skip_output:
        manifest["output"] = export_output(reader, args.output_dir, args.storage)

    if not args.skip_sha256:
        files = sorted(args.output_dir.glob("*.ncnn.*"))
        manifest["sha256"] = {path.name: sha256(path) for path in files}
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
