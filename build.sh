#!/usr/bin/env bash
# 一键构建 seedvr2-ncnn-vulkan（Linux / WSL2 / macOS）。
# 流程：初始化 ncnn submodule → 应用 ncnn patches（Convolution3D Vulkan + cpu 修复）
#       → CMake configure → 编译。产物：build/seedvr2-ncnn-vulkan
#
# 依赖：cmake >= 3.16、C++17 编译器、git、Vulkan 开发头文件
#   Ubuntu/Debian: sudo apt install build-essential cmake git libvulkan-dev vulkan-tools
set -euo pipefail
cd "$(dirname "$0")"

BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"

# 1. 初始化全部 git submodule（ncnn 及其 glslang 等，幂等，已就绪时秒回）。
echo "[build] initializing git submodules ..."
git submodule update --init --recursive \
    || echo "[build] warning: some optional submodules failed to clone (pybind11 only matters for NCNN python bindings)."
if [ ! -f ncnn/glslang/CMakeLists.txt ]; then
    echo "[build] error: ncnn/glslang is missing (required for Vulkan shaders)."
    echo "[build] hint: if github.com is unreachable, set a proxy first, e.g."
    echo '[build]   git config --global url."https://gh-proxy.com/https://github.com/".insteadOf "https://github.com/"'
    exit 1
fi

# 2. 应用项目自带的 ncnn patches（幂等：已应用则跳过）。
if [ ! -f ncnn/src/layer/vulkan/convolution3d_vulkan.cpp ]; then
    echo "[build] applying ncnn patches ..."
    for patch_file in patches/*.patch; do
        # git -C ncnn 会切换工作目录，patch 必须用绝对路径。
        abs_patch="$(cd "$(dirname "$patch_file")" && pwd)/$(basename "$patch_file")"
        if ! git -C ncnn apply --check "$abs_patch" 2>/dev/null; then
            echo "[build] patch does not apply (maybe already applied or version drift): $patch_file"
            exit 1
        fi
        git -C ncnn apply "$abs_patch"
        echo "[build]   applied $(basename "$patch_file")"
    done
else
    echo "[build] ncnn patches already applied, skipping."
fi

# 3. CMake configure + build。
echo "[build] configuring ($BUILD_TYPE, -j$JOBS) ..."
cmake -S . -B build -DCMAKE_BUILD_TYPE="$BUILD_TYPE" -DSEEDVR2_ENABLE_VULKAN=ON
echo "[build] compiling ..."
cmake --build build -j "$JOBS"

echo ""
echo "[build] done -> build/seedvr2-ncnn-vulkan"
echo "[build] next: ./download-models.sh && ./build/seedvr2-ncnn-vulkan -i in.mp4 -o out.mp4"
