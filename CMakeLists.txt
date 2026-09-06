# 本文件是 SeedVR2 NCNN Runtime 的顶层构建入口。
# 它生成可复用静态库、正式 CLI 和分层数值测试；NCNN 可由源码子目录构建，
# Vulkan 作为显式选项保留，但六个自定义层补齐 forward_vkcompute 前运行时会拒绝该后端。
cmake_minimum_required(VERSION 3.16)
project(seedvr2_ncnn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(SEEDVR2_BUILD_APPS "Build SeedVR2 command-line applications" ON)
option(SEEDVR2_BUILD_TESTS "Build SeedVR2 validation runners" ON)
option(SEEDVR2_ENABLE_VULKAN "Build NCNN with Vulkan support" OFF)
set(NCNN_SOURCE_DIR "/home/czw1/ncnn_learn/ncnn" CACHE PATH "Path to the NCNN source tree")

set(NCNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(NCNN_VULKAN ${SEEDVR2_ENABLE_VULKAN} CACHE BOOL "" FORCE)

if(NOT EXISTS "${NCNN_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "NCNN_SOURCE_DIR does not point to an NCNN source tree: ${NCNN_SOURCE_DIR}")
endif()
add_subdirectory("${NCNN_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/ncnn")

add_library(
    seedvr2_ncnn STATIC
    src/engine.cpp
    src/preprocessing.cpp
    src/postprocessing.cpp
    src/vae.cpp
    src/dit.cpp
    src/sampler.cpp
    src/vulkan_context.cpp
    custom_layers/dynamic_framewise_group_norm.cpp
    custom_layers/dynamic_framewise_spatial_attention.cpp
    custom_layers/dynamic_space_time_shuffle.cpp
    custom_layers/seedvr2_dit_input.cpp
    custom_layers/seedvr2_dit_block.cpp
    custom_layers/seedvr2_dit_output.cpp)
target_include_directories(
    seedvr2_ncnn
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/custom_layers)
target_link_libraries(seedvr2_ncnn PUBLIC ncnn)
# 当前值是一个有意的能力闸门：只有六个 layer 都具备 Vulkan 实现后才能改为 1。
target_compile_definitions(seedvr2_ncnn PRIVATE SEEDVR2_CUSTOM_LAYERS_VULKAN=0)

if(SEEDVR2_BUILD_APPS)
    add_executable(seedvr2_cli apps/seedvr2_cli.cpp)
    target_link_libraries(seedvr2_cli PRIVATE seedvr2_ncnn)
endif()

if(SEEDVR2_BUILD_TESTS)
    enable_testing()

    add_executable(seedvr2_runtime_unit_test tests/runtime_unit_test.cpp)
    target_include_directories(seedvr2_runtime_unit_test PRIVATE src)
    target_link_libraries(seedvr2_runtime_unit_test PRIVATE seedvr2_ncnn)
    add_test(NAME seedvr2_runtime_unit_test COMMAND seedvr2_runtime_unit_test)

    add_executable(seedvr2_vae_dynamic_runner tests/vae_dynamic_runner.cpp)
    target_include_directories(seedvr2_vae_dynamic_runner PRIVATE custom_layers)
    target_link_libraries(seedvr2_vae_dynamic_runner PRIVATE seedvr2_ncnn)

    add_executable(seedvr2_dit_input_runner tests/dit_input_runner.cpp)
    target_include_directories(seedvr2_dit_input_runner PRIVATE custom_layers)
    target_link_libraries(seedvr2_dit_input_runner PRIVATE seedvr2_ncnn)

    add_executable(seedvr2_dit_block_runner tests/dit_block_runner.cpp)
    target_include_directories(seedvr2_dit_block_runner PRIVATE custom_layers)
    target_link_libraries(seedvr2_dit_block_runner PRIVATE seedvr2_ncnn)

    add_executable(seedvr2_dit_output_runner tests/dit_output_runner.cpp)
    target_include_directories(seedvr2_dit_output_runner PRIVATE custom_layers)
    target_link_libraries(seedvr2_dit_output_runner PRIVATE seedvr2_ncnn)

    add_executable(seedvr2_dit_full_runner tests/dit_full_runner.cpp)
    target_include_directories(seedvr2_dit_full_runner PRIVATE src)
    target_link_libraries(seedvr2_dit_full_runner PRIVATE seedvr2_ncnn)

    # Vulkan 自定义层数值对齐测试：仅在开启 Vulkan 时编译，避免 CPU 构建失败。
    if(SEEDVR2_ENABLE_VULKAN)
        add_executable(seedvr2_shuffle_vulkan_runner tests/shuffle_vulkan_runner.cpp)
        target_include_directories(seedvr2_shuffle_vulkan_runner PRIVATE custom_layers)
        target_link_libraries(seedvr2_shuffle_vulkan_runner PRIVATE seedvr2_ncnn)
        add_test(NAME seedvr2_shuffle_vulkan_runner COMMAND seedvr2_shuffle_vulkan_runner)

        add_executable(seedvr2_groupnorm_vulkan_runner tests/groupnorm_vulkan_runner.cpp)
        target_include_directories(seedvr2_groupnorm_vulkan_runner PRIVATE custom_layers)
        target_link_libraries(seedvr2_groupnorm_vulkan_runner PRIVATE seedvr2_ncnn)
        add_test(NAME seedvr2_groupnorm_vulkan_runner COMMAND seedvr2_groupnorm_vulkan_runner)

        add_executable(seedvr2_attention_vulkan_runner tests/attention_vulkan_runner.cpp)
        target_include_directories(seedvr2_attention_vulkan_runner PRIVATE custom_layers)
        target_link_libraries(seedvr2_attention_vulkan_runner PRIVATE seedvr2_ncnn)
        add_test(NAME seedvr2_attention_vulkan_runner COMMAND seedvr2_attention_vulkan_runner)
    endif()
endif()
