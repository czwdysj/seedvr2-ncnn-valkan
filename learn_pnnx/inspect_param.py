import sys
from pathlib import Path


def parse_param(path: Path) -> None:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if len(lines) < 2:
        raise ValueError(f"{path} is too short to be a param file")

    print(f"file: {path}")
    print(f"magic: {lines[0]}")
    print(f"counts: {lines[1]}")
    print()
    print(f"{'idx':>3}  {'op_type':<24} {'name':<28} {'inputs':<24} {'outputs':<24} params")
    print("-" * 120)

    for idx, line in enumerate(lines[2:]):
        parts = line.split()
        if len(parts) < 4:
            print(f"{idx:>3}  malformed: {line}")
            continue

        op_type = parts[0]
        name = parts[1]
        input_count = int(parts[2])
        output_count = int(parts[3])
        input_start = 4
        output_start = input_start + input_count
        param_start = output_start + output_count

        inputs = ",".join(parts[input_start:output_start]) or "-"
        outputs = ",".join(parts[output_start:param_start]) or "-"
        params = " ".join(parts[param_start:])

        print(f"{idx:>3}  {op_type:<24} {name:<28} {inputs:<24} {outputs:<24} {params}")


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python inspect_param.py out/toy_pnnx_net.ncnn.param")
        raise SystemExit(2)

    parse_param(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
