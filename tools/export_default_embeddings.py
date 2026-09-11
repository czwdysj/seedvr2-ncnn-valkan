#!/usr/bin/env python3
"""把官方 pos_emb.pt / neg_emb.pt 转成 Engine 可直接加载的默认文本 embedding。

SeedVR2 官方仓库提供的 pos_emb.pt / neg_emb.pt 是预计算好的默认正/负文本
embedding（bfloat16，[tokens, 5120]）。本脚本把它们转成无依赖的原始二进制，
使 ncnn 运行时在不引入任何 PyTorch 依赖的情况下提供开箱即用的默认文本条件。

输出文件格式（小端）：
    int32 tokens
    int32 channels
    fp32  data[tokens * channels]

用法：
    python tools/export_default_embeddings.py \
        --pos pytorch_model/pos_emb.pt \
        --neg pytorch_model/neg_emb.pt \
        --out embeddings/
"""
import argparse
from pathlib import Path

import torch


def export(src: Path, dst: Path) -> None:
    tensor = torch.load(src, map_location="cpu", weights_only=False)
    if isinstance(tensor, dict):
        raise SystemExit(f"{src}: expected a plain Tensor, got dict with keys {list(tensor)}")
    tensor = tensor.detach().to(torch.float32).contiguous().cpu()
    if tensor.dim() != 2:
        raise SystemExit(f"{src}: expected 2-D [tokens, channels], got {tuple(tensor.shape)}")
    tokens, channels = tensor.shape
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as stream:
        stream.write(int(tokens).to_bytes(4, "little"))
        stream.write(int(channels).to_bytes(4, "little"))
        stream.write(tensor.numpy().tobytes())
    print(f"{src} -> {dst}: [{tokens}, {channels}] fp32, {dst.stat().st_size} bytes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pos", default="pytorch_model/pos_emb.pt")
    parser.add_argument("--neg", default="pytorch_model/neg_emb.pt")
    parser.add_argument("--out", default="embeddings/")
    args = parser.parse_args()
    out = Path(args.out)
    export(Path(args.pos), out / "default_pos_emb.bin")
    export(Path(args.neg), out / "default_neg_emb.bin")


if __name__ == "__main__":
    main()
