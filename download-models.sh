#!/usr/bin/env bash
# 本文件下载并组装可直接推理的 SeedVR2 ncnn 模型目录。
# 输入是 Hugging Face 权重仓库及其 SHA256SUMS，输出是 models/ 下完整的 DiT、
# VAE 和默认文本 embedding。大文件先写入 .part，支持断点续传；只有哈希正确
# 才原子替换正式文件。镜像不可用时自动回退到 Hugging Face 官方端点。
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

HF_REPO="${SEEDVR2_HF_REPO:-vvzc/seedvr2-ncnn-models}"
PRIMARY_ENDPOINT="${SEEDVR2_HF_ENDPOINT:-https://hf-mirror.com}"
OFFICIAL_ENDPOINT="https://huggingface.co"
MODEL_DIR="${SEEDVR2_MODEL_DIR:-$ROOT_DIR/models}"
VERIFY_ONLY=0

die()
{
    echo "[models] error: $*" >&2
    exit 1
}

usage()
{
    cat <<'EOF'
usage: ./download-models.sh [--verify-only]

Environment:
  SEEDVR2_MODEL_DIR    output model directory (default: ./models)
  SEEDVR2_HF_REPO      Hugging Face repository
  SEEDVR2_HF_ENDPOINT  preferred endpoint (default: https://hf-mirror.com)
EOF
}

case "${1:-}" in
    "") ;;
    --verify-only) VERIFY_ONLY=1 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown option: $1" ;;
esac

command -v curl >/dev/null 2>&1 \
    || die "curl is required. Install it with: sudo apt install curl"
command -v sha256sum >/dev/null 2>&1 \
    || die "sha256sum is required. Install GNU coreutils."

mkdir -p "$MODEL_DIR/dit_full_fp16" "$MODEL_DIR/vae_dynamic"

fetch_url()
{
    local endpoint="$1"
    local relative_path="$2"
    local destination="$3"
    curl --location --fail --retry 4 --retry-all-errors --connect-timeout 20 \
        --continue-at - --output "$destination" \
        "$endpoint/$HF_REPO/resolve/main/$relative_path"
}

fetch_with_fallback()
{
    local relative_path="$1"
    local destination="$2"
    if fetch_url "$PRIMARY_ENDPOINT" "$relative_path" "$destination"; then
        return 0
    fi
    if [ "$PRIMARY_ENDPOINT" != "$OFFICIAL_ENDPOINT" ]; then
        echo "[models] preferred endpoint failed; retrying $relative_path from Hugging Face ..." >&2
        fetch_url "$OFFICIAL_ENDPOINT" "$relative_path" "$destination"
        return 0
    fi
    return 1
}

download_checksum_file()
{
    local component="$1"
    local target="$MODEL_DIR/$component/SHA256SUMS"
    local partial="$target.part"
    # 校验清单很小，每次重新获取，避免远端发布更新后继续信任旧清单。
    rm -f -- "$partial"
    fetch_with_fallback "$component/SHA256SUMS" "$partial" \
        || die "failed to download $component/SHA256SUMS"
    [ -s "$partial" ] || die "$component/SHA256SUMS is empty"
    mv -f -- "$partial" "$target"
}

download_checked_file()
{
    local component="$1"
    local expected="$2"
    local filename="$3"
    local relative_path="$component/$filename"
    local target="$MODEL_DIR/$relative_path"
    local partial="$target.part"

    case "$filename" in
        */*|""|.*) die "unsafe filename in $component/SHA256SUMS: $filename" ;;
    esac
    mkdir -p "$(dirname "$target")"
    if [ -f "$target" ] && [ "$(sha256sum "$target" | awk '{print $1}')" = "$expected" ]; then
        echo "[models] verified $relative_path"
        return 0
    fi

    echo "[models] downloading $relative_path"
    if ! fetch_with_fallback "$relative_path" "$partial"; then
        die "failed to download $relative_path; keep $partial for the next resume"
    fi
    if [ "$(sha256sum "$partial" | awk '{print $1}')" != "$expected" ]; then
        echo "[models] resumed payload checksum mismatch; retrying from byte zero ..." >&2
        rm -f -- "$partial"
        fetch_with_fallback "$relative_path" "$partial" \
            || die "failed to redownload $relative_path"
    fi
    [ "$(sha256sum "$partial" | awk '{print $1}')" = "$expected" ] \
        || die "SHA256 mismatch after clean download: $relative_path"
    mv -f -- "$partial" "$target"
}

install_embeddings()
{
    local filename source target partial
    for filename in default_pos_emb.bin default_neg_emb.bin; do
        source="$ROOT_DIR/embeddings/$filename"
        target="$MODEL_DIR/$filename"
        partial="$target.part"
        [ -s "$source" ] || die "bundled embedding is missing: embeddings/$filename"
        if [ -f "$target" ] && cmp -s "$source" "$target"; then
            echo "[models] verified $filename"
            continue
        fi
        cp -- "$source" "$partial"
        [ "$(sha256sum "$source" | awk '{print $1}')" = "$(sha256sum "$partial" | awk '{print $1}')" ] \
            || die "failed to copy bundled embedding: $filename"
        mv -f -- "$partial" "$target"
        echo "[models] installed $filename"
    done
}

if [ "$VERIFY_ONLY" = "0" ]; then
    install_embeddings
    for component in dit_full_fp16 vae_dynamic; do
        download_checksum_file "$component"
        while read -r expected filename extra; do
            [ -n "${expected:-}" ] || continue
            [ -z "${extra:-}" ] || die "invalid checksum line for $component/$filename"
            [[ "$expected" =~ ^[0-9a-fA-F]{64}$ ]] \
                || die "invalid SHA256 in $component/SHA256SUMS"
            download_checked_file "$component" "${expected,,}" "$filename"
        done < "$MODEL_DIR/$component/SHA256SUMS"
    done
fi

SEEDVR2_MODEL_DIR="$MODEL_DIR" "$ROOT_DIR/scripts/verify-models.sh"
echo
echo "[models] ready: $MODEL_DIR"
echo "[models] run: ./build/seedvr2-ncnn-vulkan -i input.mp4 -o output.mp4 --models '$MODEL_DIR' --resident"
