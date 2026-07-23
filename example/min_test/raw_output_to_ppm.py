#!/usr/bin/env python3
"""把 seedvr2_cli 的单帧 FP32 THWC 输出转换成可查看的 PPM。

输入文件前 16 字节是四个 little-endian int32，依次表示 T、H、W、C，后续为
范围 [0,1] 的连续 FP32 THWC 数据。本脚本只支持 T=1、C=3 的最小测试输出，
会检查文件长度与 finite 数值，限幅并量化为 RGB8；它只使用 Python 标准库。
"""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path


def convert(input_path: Path, output_path: Path) -> tuple[int, int, float, float]:
    """读取并验证 raw，写出 P6 PPM，返回宽、高和输入数值范围。"""
    payload = input_path.read_bytes()
    if len(payload) < 16:
        raise ValueError("输出文件不足 16 字节，缺少 T,H,W,C 文件头")
    frames, height, width, channels = struct.unpack_from("<4i", payload)
    if frames != 1 or channels != 3 or height <= 0 or width <= 0:
        raise ValueError(f"只支持 T=1,C=3，实际为 {(frames, height, width, channels)}")

    count = frames * height * width * channels
    expected_bytes = 16 + count * 4
    if len(payload) != expected_bytes:
        raise ValueError(f"文件长度应为 {expected_bytes}，实际为 {len(payload)}")
    values = struct.unpack_from(f"<{count}f", payload, 16)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("输出包含 NaN 或 Inf")

    rgb = bytes(round(max(0.0, min(1.0, value)) * 255.0) for value in values)
    output_path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + rgb)
    return width, height, min(values), max(values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    width, height, minimum, maximum = convert(args.input, args.output)
    print(
        f"已写入 {args.output}: {width}x{height}, "
        f"finite=true, min={minimum:.6g}, max={maximum:.6g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
