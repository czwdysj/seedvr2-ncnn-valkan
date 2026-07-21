# 本文件负责构建 SeedVR2 VAE 动态形状 NCNN 测试运行器。
# 这里把自定义 CPU layer 和一个命令行 runner 编到同一个可执行文件里，
# 主要用于验证 pnnx 导出的 VAE 子图能否在不同 T,H,W 输入下和 PyTorch 对齐。
cmake_minimum_required(VERSION 3.16)
project(seedvr2_vae_dynamic LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# 当前 runner 优先做数值正确性验证，所以只需要 NCNN 库本体，
# 不构建工具、示例、benchmark 和 Python 绑定，减少本地编译时间。
set(NCNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
# 这条链路目前验证 CPU 自定义 layer；后续迁移到 Vulkan 时，
# 需要把对应 layer 扩展为 forward_vkcompute 并重新打开 NCNN_VULKAN。
set(NCNN_VULKAN OFF CACHE BOOL "" FORCE)

if(NOT DEFINED NCNN_SOURCE_DIR)
    set(NCNN_SOURCE_DIR "/home/czw1/ncnn_learn/ncnn")
endif()
add_subdirectory("${NCNN_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/ncnn")

# runner 显式注册下面三个动态 layer，避免把 SeedVR2 VAE 中依赖运行时
# 帧数或空间尺寸的逻辑硬编码进静态 param 图里。
add_executable(
    seedvr2_vae_dynamic_runner
    vae_dynamic_runner.cpp
    ../custom_layers/dynamic_framewise_group_norm.cpp
    ../custom_layers/dynamic_framewise_spatial_attention.cpp
    ../custom_layers/dynamic_space_time_shuffle.cpp)
target_include_directories(seedvr2_vae_dynamic_runner PRIVATE ../custom_layers)
target_link_libraries(seedvr2_vae_dynamic_runner PRIVATE ncnn)
