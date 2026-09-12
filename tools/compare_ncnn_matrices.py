#!/usr/bin/env python3
"""比较两个 NCNN 测试 runner 输出的 FP32 矩阵。

输入文件格式为 int32 ``[rows, columns]`` 文件头，随后是连续 FP32 payload。
脚本输出 max_abs、RMSE、NRMSE 与 cosine，并可用命令行阈值作为自动化门禁。
它用于 CPU/Vulkan、resident/streaming 等运行路径的直接数值一致性检查。
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


def read_matrix(path: Path) -> np.ndarray:
    """读取并严格校验矩阵文件头和 payload 长度。"""
    with path.open("rb") as stream:
        header = stream.read(8)
        if len(header) != 8:
            raise ValueError(f"{path}: 文件头不足 8 字节")
        rows, columns = struct.unpack("ii", header)
        value = np.fromfile(stream, dtype=np.float32)
    if rows <= 0 or columns <= 0 or value.size != rows * columns:
        raise ValueError(
            f"{path}: 声明形状为 ({rows},{columns})，payload 元素数为 {value.size}"
        )
    return value.reshape(rows, columns)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("actual", type=Path)
    parser.add_argument("expected", type=Path)
    parser.add_argument("--max-nrmse", type=float, default=float("inf"))
    parser.add_argument("--min-cosine", type=float, default=-1.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    actual = read_matrix(args.actual).astype(np.float64)
    expected = read_matrix(args.expected).astype(np.float64)
    if actual.shape != expected.shape:
        raise ValueError(f"矩阵形状不一致: actual={actual.shape}, expected={expected.shape}")
    difference = actual - expected
    rmse = float(np.sqrt(np.mean(difference * difference)))
    reference_rms = max(float(np.sqrt(np.mean(expected * expected))), 1e-12)
    denominator = max(float(np.linalg.norm(actual) * np.linalg.norm(expected)), 1e-12)
    report = {
        "actual": str(args.actual),
        "expected": str(args.expected),
        "shape": list(actual.shape),
        "max_abs": float(np.max(np.abs(difference))),
        "rmse": rmse,
        "nrmse": rmse / reference_rms,
        "cosine": float(np.sum(actual * expected) / denominator),
    }
    report["passed"] = (
        report["nrmse"] <= args.max_nrmse and report["cosine"] >= args.min_cosine
    )
    output = json.dumps(report, ensure_ascii=False, indent=2)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
