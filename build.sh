#!/usr/bin/env bash
# 本文件是 Linux/WSL2 的完整构建入口。
# 输入为固定版本的 ncnn 子模块与 patches/ 中的兼容补丁，输出为可直接运行的
# seedvr2-ncnn-vulkan 和轻量测试程序。脚本会检查编译器、CMake、Vulkan、
# ffmpeg 和独立 GPU，逐个幂等应用补丁，并且只在 CTest 通过后报告构建成功。
# CI 或仅验证 CPU 编译时可设置 SEEDVR2_ENABLE_VULKAN=OFF、SEEDVR2_SKIP_GPU_CHECK=1。
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

BUILD_DIR="${BUILD_DIR:-build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"
ENABLE_VULKAN="${SEEDVR2_ENABLE_VULKAN:-ON}"
BUILD_TESTS="${SEEDVR2_BUILD_TESTS:-ON}"
SKIP_GPU_CHECK="${SEEDVR2_SKIP_GPU_CHECK:-0}"

die()
{
    echo "[build] error: $*" >&2
    exit 1
}

require_command()
{
    command -v "$1" >/dev/null 2>&1 || die "missing command '$1'. $2"
}

version_at_least()
{
    printf '%s\n%s\n' "$2" "$1" | sort -V -C
}

check_dependencies()
{
    require_command git "Install it with: sudo apt install git"
    require_command cmake "Install it with: sudo apt install cmake"
    require_command c++ "Install it with: sudo apt install build-essential"
    require_command ffmpeg "Install it with: sudo apt install ffmpeg"
    require_command ffprobe "ffprobe is provided by the ffmpeg package."

    local cmake_version
    cmake_version="$(cmake --version | awk 'NR == 1 {print $3}')"
    version_at_least "$cmake_version" "3.16" \
        || die "CMake >= 3.16 is required, found $cmake_version"

    if [ "$ENABLE_VULKAN" = "ON" ]; then
        require_command vulkaninfo "Install Vulkan headers/tools with: sudo apt install libvulkan-dev vulkan-tools"
        if [ ! -f /usr/include/vulkan/vulkan.h ] && ! pkg-config --exists vulkan 2>/dev/null; then
            die "Vulkan headers were not found. Install libvulkan-dev."
        fi
        if [ "$SKIP_GPU_CHECK" != "1" ]; then
            local summary
            summary="$(vulkaninfo --summary 2>&1)" \
                || die "vulkaninfo failed. Check the GPU driver and WSL2 GPU passthrough."
            echo "$summary" | grep -Eiq 'NVIDIA|AMD|Radeon|Intel' \
                || die "no hardware Vulkan GPU was found (software llvmpipe/lavapipe is unsupported)."
            echo "[build] Vulkan hardware device detected."
        fi
    fi
}

initialize_submodules()
{
    echo "[build] initializing pinned git submodules ..."
    # 与 zimage-ncnn-vulkan 一样只获取固定提交需要的工作树；ncnn 完整历史对
    # 国内网络代价很高。少数 Git 服务不允许浅取固定提交时再回退普通克隆。
    if ! git submodule update --init --recursive --depth 1; then
        echo "[build] shallow submodule checkout failed; retrying without --depth ..." >&2
        git submodule update --init --recursive \
            || die "failed to fetch ncnn submodules. Check access to https://github.com/Tencent/ncnn.git"
    fi
    [ -f ncnn/CMakeLists.txt ] || die "ncnn submodule is missing."
    if [ "$ENABLE_VULKAN" = "ON" ]; then
        [ -f ncnn/glslang/CMakeLists.txt ] \
            || die "ncnn/glslang is missing; run: git submodule update --init --recursive"
    fi
}

apply_ncnn_patches()
{
    local patch_file
    for patch_file in "$ROOT_DIR"/patches/*.patch; do
        [ -f "$patch_file" ] || die "no ncnn patches found in patches/."
        if git -C ncnn apply --check "$patch_file" >/dev/null 2>&1; then
            git -C ncnn apply "$patch_file"
            echo "[build] applied $(basename "$patch_file")"
        elif git -C ncnn apply --reverse --check "$patch_file" >/dev/null 2>&1; then
            echo "[build] already applied $(basename "$patch_file")"
        else
            die "patch conflicts with pinned ncnn: $(basename "$patch_file"). Do not update the ncnn submodule independently."
        fi
    done
}

check_dependencies
initialize_submodules
apply_ncnn_patches

echo "[build] configuring $BUILD_DIR ($BUILD_TYPE, Vulkan=$ENABLE_VULKAN, tests=$BUILD_TESTS) ..."
cmake -S . -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DSEEDVR2_ENABLE_VULKAN="$ENABLE_VULKAN" \
    -DSEEDVR2_BUILD_TESTS="$BUILD_TESTS"

echo "[build] compiling with $JOBS job(s) ..."
cmake --build "$BUILD_DIR" --parallel "$JOBS"

if [ "$BUILD_TESTS" = "ON" ]; then
    echo "[build] running lightweight CTest gate ..."
    ctest --test-dir "$BUILD_DIR" --output-on-failure -L smoke
fi

echo
echo "[build] done: $BUILD_DIR/seedvr2-ncnn-vulkan"
echo "[build] next: ./download-models.sh"
echo "[build] run:  $BUILD_DIR/seedvr2-ncnn-vulkan -i input.mp4 -o output.mp4 --resident"
