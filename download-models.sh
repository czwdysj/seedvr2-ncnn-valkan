#!/usr/bin/env bash
# 下载 ncnn 推理权重并组装 models/ 目录（约 7.5GB）。
#
# 权重托管在 HuggingFace；默认走 hf-mirror.com（国内加速），海外用户可：
#   SEEDVR2_HF_ENDPOINT=https://huggingface.co ./download-models.sh
# 或自定义仓库：
#   SEEDVR2_HF_REPO=<user>/<repo> ./download-models.sh
#
# 默认文本 embedding（default_*_emb.bin，共约 2.5MB）已内置在仓库 embeddings/
# 目录中，本脚本直接复制进 models/，无需下载。
set -euo pipefail
cd "$(dirname "$0")"

HF_REPO="${SEEDVR2_HF_REPO:-vvzc/seedvr2-ncnn-models}"
ENDPOINT="${SEEDVR2_HF_ENDPOINT:-https://hf-mirror.com}"
BASE="${ENDPOINT}/${HF_REPO}/resolve/main"

mkdir -p models/dit_full_fp16 models/vae_dynamic

# 内置默认文本 embedding：复制进 models/（Engine 未传文本时自动使用）。
cp -f embeddings/default_pos_emb.bin models/
cp -f embeddings/default_neg_emb.bin models/
echo "[get]  default_pos_emb.bin / default_neg_emb.bin (bundled)"

download() {
    local rel="$1"
    local dst="models/$rel"
    if [ -s "$dst" ]; then
        echo "[skip] $rel (already present)"
        return
    fi
    echo "[get]  $rel"
    mkdir -p "$(dirname "$dst")"
    curl -L --fail --retry 3 --progress-bar -o "$dst" "$BASE/$rel"
}

# DiT 32 个 Transformer block（fp16 权重，约 6.4GB）。
for index in $(seq -w 0 31); do
    download "dit_full_fp16/seedvr2_dit_block_${index}.ncnn.param"
    download "dit_full_fp16/seedvr2_dit_block_${index}.ncnn.bin"
done
# DiT 输入/输出投影头。
download "dit_full_fp16/seedvr2_dit_input.ncnn.param"
download "dit_full_fp16/seedvr2_dit_input.ncnn.bin"
download "dit_full_fp16/seedvr2_dit_output.ncnn.param"
download "dit_full_fp16/seedvr2_dit_output.ncnn.bin"
# VAE encoder + decoder。
download "vae_dynamic/seedvr2_vae_encoder_dynamic.ncnn.param"
download "vae_dynamic/seedvr2_vae_encoder_dynamic.ncnn.bin"
download "vae_dynamic/seedvr2_vae_decoder_dynamic.ncnn.param"
download "vae_dynamic/seedvr2_vae_decoder_dynamic.ncnn.bin"

echo ""
echo "[download] done -> models/ ($(du -sh models | cut -f1))"
echo "[download] next: ./build/seedvr2-ncnn-vulkan -i input.mp4 -o output.mp4"
