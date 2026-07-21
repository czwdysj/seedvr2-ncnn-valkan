import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

from toy_pnnx_net import build_model


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "out"
INPUT_SHAPE = (1, 3, 32, 32)


def find_pnnx() -> str:
    env_pnnx = os.environ.get("PNNX")
    if env_pnnx:
        return env_pnnx

    path_pnnx = shutil.which("pnnx")
    if path_pnnx:
        return path_pnnx

    local_candidate = Path("/home/czw1/ncnn_learn/Penguin-VL-ncnn/.venv/bin/pnnx")
    if local_candidate.exists():
        return str(local_candidate)

    raise FileNotFoundError("Cannot find pnnx. Set PNNX=/path/to/pnnx and rerun.")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    torch.manual_seed(20260716)

    model = build_model()
    example = torch.randn(*INPUT_SHAPE)

    with torch.no_grad():
        eager_output = model(example)

    torchscript_path = OUT_DIR / "toy_pnnx_net.pt"
    traced = torch.jit.trace(model, example)
    traced.save(str(torchscript_path))

    np.save(OUT_DIR / "input.npy", example.numpy())
    np.save(OUT_DIR / "pytorch_output.npy", eager_output.numpy())

    pnnx = find_pnnx()
    cmd = [
        pnnx,
        str(torchscript_path),
        "inputshape=[1,3,32,32]",
        "fp16=0",
        "optlevel=2",
        f"pnnxparam={OUT_DIR / 'toy_pnnx_net.pnnx.param'}",
        f"pnnxbin={OUT_DIR / 'toy_pnnx_net.pnnx.bin'}",
        f"pnnxpy={OUT_DIR / 'toy_pnnx_net_pnnx.py'}",
        f"pnnxonnx={OUT_DIR / 'toy_pnnx_net.pnnx.onnx'}",
        f"ncnnparam={OUT_DIR / 'toy_pnnx_net.ncnn.param'}",
        f"ncnnbin={OUT_DIR / 'toy_pnnx_net.ncnn.bin'}",
        f"ncnnpy={OUT_DIR / 'toy_pnnx_net_ncnn.py'}",
    ]

    print("Running:")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)

    print("\nGenerated files:")
    for path in sorted(OUT_DIR.iterdir()):
        print(f"{path.relative_to(ROOT)}  {path.stat().st_size} bytes")

    print("\nUseful next commands:")
    print("python inspect_param.py out/toy_pnnx_net.pnnx.param")
    print("python inspect_param.py out/toy_pnnx_net.ncnn.param")


if __name__ == "__main__":
    main()
