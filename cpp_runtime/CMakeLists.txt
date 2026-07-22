# 本文件负责构建 SeedVR2 NCNN 的分层数值测试运行器。
# 当前包含动态 VAE runner 和 DiT block runner；它们分别加载真实 param/bin，
# 用于在组装完整 Engine 前验证每个自定义层及其周围原生 NCNN 子图。
cmake_minimum_required(VERSION 3.16)
project(seedvr2_ncnn_runtime LANGUAGES CXX)

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

# DiT runner 一次只加载一个 block，使 3B 模型可以在有限内存设备上逐层验证。
add_executable(
    seedvr2_dit_block_runner
    dit_block_runner.cpp
    ../custom_layers/seedvr2_dit_block.cpp)
target_include_directories(seedvr2_dit_block_runner PRIVATE ../custom_layers)
target_link_libraries(seedvr2_dit_block_runner PRIVATE ncnn)

# DiT 输入与输出头单独测试，便于分别定位 patch、embedding 和 output Ada。
add_executable(
    seedvr2_dit_input_runner
    dit_input_runner.cpp
    ../custom_layers/seedvr2_dit_input.cpp)
target_include_directories(seedvr2_dit_input_runner PRIVATE ../custom_layers)
target_link_libraries(seedvr2_dit_input_runner PRIVATE ncnn)

add_executable(
    seedvr2_dit_output_runner
    dit_output_runner.cpp
    ../custom_layers/seedvr2_dit_output.cpp)
target_include_directories(seedvr2_dit_output_runner PRIVATE ../custom_layers)
target_link_libraries(seedvr2_dit_output_runner PRIVATE ncnn)

# 完整 DiT runner 按 block 流式加载，验证转换后的 3B 主干可以真正串联推理。
add_executable(
    seedvr2_dit_full_runner
    dit_full_runner.cpp
    ../custom_layers/seedvr2_dit_input.cpp
    ../custom_layers/seedvr2_dit_block.cpp
    ../custom_layers/seedvr2_dit_output.cpp)
target_include_directories(seedvr2_dit_full_runner PRIVATE ../custom_layers)
target_link_libraries(seedvr2_dit_full_runner PRIVATE ncnn)
