from pathlib import Path

import ncnn
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "out"


def run_ncnn(input_nchw: np.ndarray) -> np.ndarray:
    with ncnn.Net() as net:
        net.load_param(str(OUT_DIR / "toy_pnnx_net.ncnn.param"))
        net.load_model(str(OUT_DIR / "toy_pnnx_net.ncnn.bin"))

        with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(input_nchw.squeeze(0)).clone())
            _, out0 = ex.extract("out0")
            return np.array(out0, dtype=np.float32).reshape(1, -1)


def main() -> None:
    input_nchw = np.load(OUT_DIR / "input.npy").astype(np.float32)
    pytorch_output = np.load(OUT_DIR / "pytorch_output.npy").astype(np.float32)
    ncnn_output = run_ncnn(input_nchw)

    diff = np.abs(pytorch_output - ncnn_output)
    print("pytorch:", pytorch_output)
    print("ncnn:   ", ncnn_output)
    print("max abs diff:", float(diff.max()))
    print("mean abs diff:", float(diff.mean()))


if __name__ == "__main__":
    main()
