#!/usr/bin/env bash
# 本文件是一键执行 SeedVR2 NCNN 最小端到端测试的入口。
# 输入是单帧 16x16 RGB FP32 THWC 图标和一个 [1,5120] 零文本 embedding；
# 输出先由正式 seedvr2_cli 写成带 T,H,W,C 文件头的 FP32 raw，再转换为可查看图片。
# 运行假设是项目根目录存在 ncnn_models，且 WSL 中可以调用 cmake、python3 和可选 ffmpeg。
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/../.." && pwd)"
build_dir="${project_dir}/build"
cli="${build_dir}/seedvr2_cli"
model_dir="${project_dir}/ncnn_models"
threads="${SEEDVR2_THREADS:-8}"

if [[ ! -x "${cli}" ]]; then
    cmake -S "${project_dir}" -B "${build_dir}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DSEEDVR2_ENABLE_VULKAN=OFF
    cmake --build "${build_dir}" -j"${threads}"
fi

required_files=(
    "${model_dir}/vae_dynamic/seedvr2_vae_encoder_dynamic.ncnn.param"
    "${model_dir}/vae_dynamic/seedvr2_vae_decoder_dynamic.ncnn.param"
    "${model_dir}/dit_full_fp16/seedvr2_dit_input.ncnn.param"
    "${model_dir}/dit_full_fp16/seedvr2_dit_block_00.ncnn.param"
    "${model_dir}/dit_full_fp16/seedvr2_dit_block_31.ncnn.param"
    "${model_dir}/dit_full_fp16/seedvr2_dit_output.ncnn.param"
)
for file in "${required_files[@]}"; do
    if [[ ! -f "${file}" ]]; then
        printf '缺少模型文件: %s\n' "${file}" >&2
        exit 2
    fi
done

raw_output="${script_dir}/output.f32"
ppm_output="${script_dir}/output.ppm"
png_output="${script_dir}/output.png"

"${cli}" \
    "${model_dir}" \
    "${script_dir}/input_1x16x16_thwc.f32" 1 16 16 \
    "${script_dir}/text_zero_1x5120.f32" 1 \
    "${script_dir}/text_zero_1x5120.f32" 1 \
    "${raw_output}" "${threads}"

python3 "${script_dir}/raw_output_to_ppm.py" "${raw_output}" "${ppm_output}"
if command -v ffmpeg >/dev/null 2>&1; then
    ffmpeg -y -loglevel error -i "${ppm_output}" "${png_output}"
    printf '可视化结果: %s\n' "${png_output}"
else
    printf '未找到 ffmpeg，可视化结果保留为: %s\n' "${ppm_output}"
fi
printf '原始 FP32 输出: %s\n' "${raw_output}"
