#!/usr/bin/env bash
# 本文件对端侧推理所需模型做离线完整性验证。
# 输入是 SEEDVR2_MODEL_DIR 指向的模型目录；输出为明确的成功或失败状态。
# 它校验远端发布的 DiT/VAE SHA256 清单、32 个 block、输入输出头、动态 VAE
# 以及仓库内置的默认文本 embedding，不加载 7GB 权重，因此适合下载后和 CI 使用。
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${SEEDVR2_MODEL_DIR:-$ROOT_DIR/models}"

die()
{
    echo "[verify-models] error: $*" >&2
    exit 1
}

verify_component()
{
    local component="$1"
    local directory="$MODEL_DIR/$component"
    [ -s "$directory/SHA256SUMS" ] || die "$component/SHA256SUMS is missing"
    echo "[verify-models] checking $component ..."
    (cd "$directory" && sha256sum --check --strict SHA256SUMS) \
        || die "$component contains missing or corrupted files"
}

for filename in default_pos_emb.bin default_neg_emb.bin; do
    [ -s "$MODEL_DIR/$filename" ] || die "$filename is missing"
    cmp -s "$ROOT_DIR/embeddings/$filename" "$MODEL_DIR/$filename" \
        || die "$filename differs from the bundled embedding"
done

verify_component dit_full_fp16
verify_component vae_dynamic

for index in $(seq -w 0 31); do
    [ -s "$MODEL_DIR/dit_full_fp16/seedvr2_dit_block_${index}.ncnn.param" ] \
        || die "DiT block $index param is missing"
    [ -s "$MODEL_DIR/dit_full_fp16/seedvr2_dit_block_${index}.ncnn.bin" ] \
        || die "DiT block $index bin is missing"
done
for filename in seedvr2_dit_input.ncnn.param seedvr2_dit_input.ncnn.bin \
                seedvr2_dit_output.ncnn.param seedvr2_dit_output.ncnn.bin; do
    [ -s "$MODEL_DIR/dit_full_fp16/$filename" ] || die "$filename is missing"
done
for filename in seedvr2_vae_encoder_dynamic.ncnn.param seedvr2_vae_encoder_dynamic.ncnn.bin \
                seedvr2_vae_decoder_dynamic.ncnn.param seedvr2_vae_decoder_dynamic.ncnn.bin; do
    [ -s "$MODEL_DIR/vae_dynamic/$filename" ] || die "$filename is missing"
done

echo "[verify-models] PASS: complete SeedVR2 model set in $MODEL_DIR"
