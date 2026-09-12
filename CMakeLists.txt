# 本文件是 SeedVR2 NCNN Runtime 的顶层构建入口。
# 它生成可复用静态库、正式 CLI 和分层数值测试；NCNN 可由源码子目录构建，
# Vulkan 作为显式选项保留；安装阶段同时导出可供外部项目 find_package 的 CMake 包。
cmake_minimum_required(VERSION 3.16)
project(seedvr2_ncnn LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(SEEDVR2_BUILD_APPS "Build SeedVR2 command-line applications" ON)
option(SEEDVR2_BUILD_TESTS "Build SeedVR2 validation runners" ON)
option(SEEDVR2_ENABLE_VULKAN "Build NCNN with Vulkan support" OFF)
set(NCNN_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ncnn" CACHE PATH "Path to the NCNN source tree (git submodule)")

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
    src/core/engine.cpp
    src/pipeline/preprocessing.cpp
    src/pipeline/postprocessing.cpp
    src/model/vae.cpp
    src/model/dit.cpp
    src/model/sampler.cpp
    src/core/vulkan_context.cpp
    src/layers/dynamic_framewise_group_norm.cpp
    src/layers/dynamic_framewise_spatial_attention.cpp
    src/layers/dynamic_space_time_shuffle.cpp
    src/layers/seedvr2_dit_input.cpp
    src/layers/seedvr2_dit_block.cpp
    src/layers/seedvr2_dit_output.cpp)
target_include_directories(
    seedvr2_ncnn
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(seedvr2_ncnn PUBLIC ncnn)
# 六个自定义层均已实现 forward_vkcompute，能力闸门随 Vulkan 构建开关打开；
# 未开启 Vulkan 构建时保持 0，运行时仍拒绝 Vulkan 后端。
if(SEEDVR2_ENABLE_VULKAN)
    target_compile_definitions(seedvr2_ncnn PRIVATE SEEDVR2_CUSTOM_LAYERS_VULKAN=1)
else()
    target_compile_definitions(seedvr2_ncnn PRIVATE SEEDVR2_CUSTOM_LAYERS_VULKAN=0)
endif()

if(SEEDVR2_BUILD_APPS)
    add_executable(seedvr2-ncnn-vulkan src/cli/main.cpp)
    target_link_libraries(seedvr2-ncnn-vulkan PRIVATE seedvr2_ncnn)

    add_executable(seedvr2_e2e_profile tests/e2e_profile.cpp)
    target_link_libraries(seedvr2_e2e_profile PRIVATE seedvr2_ncnn)
endif()

if(SEEDVR2_BUILD_TESTS)
    enable_testing()

    if(SEEDVR2_BUILD_APPS)
        add_test(NAME seedvr2_cli_help COMMAND seedvr2-ncnn-vulkan --help)
        set_tests_properties(seedvr2_cli_help PROPERTIES LABELS "smoke;cli")
    endif()

    add_executable(seedvr2_runtime_unit_test tests/runtime_unit_test.cpp)
    target_include_directories(seedvr2_runtime_unit_test PRIVATE src)
    target_link_libraries(seedvr2_runtime_unit_test PRIVATE seedvr2_ncnn)
    add_test(NAME seedvr2_runtime_unit_test COMMAND seedvr2_runtime_unit_test)
    set_tests_properties(seedvr2_runtime_unit_test PROPERTIES LABELS "smoke")

    add_executable(seedvr2_vae_dynamic_runner tests/vae_dynamic_runner.cpp)
    target_include_directories(seedvr2_vae_dynamic_runner PRIVATE src)
    target_link_libraries(seedvr2_vae_dynamic_runner PRIVATE seedvr2_ncnn)

    add_executable(seedvr2_dit_input_runner tests/dit_input_runner.cpp)
    target_include_directories(seedvr2_dit_input_runner PRIVATE src)
    target_link_libraries(seedvr2_dit_input_runner PRIVATE seedvr2_ncnn)

    add_executable(seedvr2_dit_block_runner tests/dit_block_runner.cpp)
    target_include_directories(seedvr2_dit_block_runner PRIVATE src)
    target_link_libraries(seedvr2_dit_block_runner PRIVATE seedvr2_ncnn)

    add_executable(seedvr2_dit_output_runner tests/dit_output_runner.cpp)
    target_include_directories(seedvr2_dit_output_runner PRIVATE src)
    target_link_libraries(seedvr2_dit_output_runner PRIVATE seedvr2_ncnn)

    add_executable(seedvr2_dit_full_runner tests/dit_full_runner.cpp)
    target_include_directories(seedvr2_dit_full_runner PRIVATE src)
    target_link_libraries(seedvr2_dit_full_runner PRIVATE seedvr2_ncnn)

    # Vulkan 自定义层数值对齐测试：仅在开启 Vulkan 时编译，避免 CPU 构建失败。
    if(SEEDVR2_ENABLE_VULKAN)
        add_executable(seedvr2_shuffle_vulkan_runner tests/shuffle_vulkan_runner.cpp)
        target_include_directories(seedvr2_shuffle_vulkan_runner PRIVATE src)
        target_link_libraries(seedvr2_shuffle_vulkan_runner PRIVATE seedvr2_ncnn)
        add_test(NAME seedvr2_shuffle_vulkan_runner COMMAND seedvr2_shuffle_vulkan_runner)
        set_tests_properties(seedvr2_shuffle_vulkan_runner PROPERTIES LABELS "smoke;vulkan")

        add_executable(seedvr2_groupnorm_vulkan_runner tests/groupnorm_vulkan_runner.cpp)
        target_include_directories(seedvr2_groupnorm_vulkan_runner PRIVATE src)
        target_link_libraries(seedvr2_groupnorm_vulkan_runner PRIVATE seedvr2_ncnn)
        add_test(NAME seedvr2_groupnorm_vulkan_runner COMMAND seedvr2_groupnorm_vulkan_runner)
        set_tests_properties(seedvr2_groupnorm_vulkan_runner PROPERTIES LABELS "smoke;vulkan")

        add_executable(seedvr2_attention_vulkan_runner tests/attention_vulkan_runner.cpp)
        target_include_directories(seedvr2_attention_vulkan_runner PRIVATE src)
        target_link_libraries(seedvr2_attention_vulkan_runner PRIVATE seedvr2_ncnn)
        add_test(NAME seedvr2_attention_vulkan_runner COMMAND seedvr2_attention_vulkan_runner)
        set_tests_properties(seedvr2_attention_vulkan_runner PROPERTIES LABELS "smoke;vulkan")

        add_executable(seedvr2_dit_input_vulkan_runner tests/dit_input_vulkan_runner.cpp)
        target_include_directories(seedvr2_dit_input_vulkan_runner PRIVATE src)
        target_link_libraries(seedvr2_dit_input_vulkan_runner PRIVATE seedvr2_ncnn)
        add_test(NAME seedvr2_dit_input_vulkan_runner COMMAND seedvr2_dit_input_vulkan_runner)
        set_tests_properties(seedvr2_dit_input_vulkan_runner PROPERTIES LABELS "smoke;vulkan")

        add_executable(seedvr2_dit_output_vulkan_runner tests/dit_output_vulkan_runner.cpp)
        target_include_directories(seedvr2_dit_output_vulkan_runner PRIVATE src)
        target_link_libraries(seedvr2_dit_output_vulkan_runner PRIVATE seedvr2_ncnn)
        add_test(NAME seedvr2_dit_output_vulkan_runner COMMAND seedvr2_dit_output_vulkan_runner)
        set_tests_properties(seedvr2_dit_output_vulkan_runner PROPERTIES LABELS "smoke;vulkan")

        add_executable(seedvr2_dit_block_vulkan_runner tests/dit_block_vulkan_runner.cpp)
        target_include_directories(seedvr2_dit_block_vulkan_runner PRIVATE src)
        target_link_libraries(seedvr2_dit_block_vulkan_runner PRIVATE seedvr2_ncnn)
        add_test(NAME seedvr2_dit_block_vulkan_runner COMMAND seedvr2_dit_block_vulkan_runner)
        set_tests_properties(seedvr2_dit_block_vulkan_runner PROPERTIES LABELS "smoke;vulkan")

        add_executable(seedvr2_conv3d_vulkan_runner tests/conv3d_vulkan_runner.cpp)
        target_link_libraries(seedvr2_conv3d_vulkan_runner PRIVATE seedvr2_ncnn)
        add_test(NAME seedvr2_conv3d_vulkan_runner COMMAND seedvr2_conv3d_vulkan_runner)
        set_tests_properties(seedvr2_conv3d_vulkan_runner PROPERTIES LABELS "smoke;vulkan")

        add_executable(seedvr2_sampler_vulkan_runner tests/sampler_vulkan_runner.cpp)
        target_include_directories(seedvr2_sampler_vulkan_runner PRIVATE src)
        target_link_libraries(seedvr2_sampler_vulkan_runner PRIVATE seedvr2_ncnn)
        add_test(NAME seedvr2_sampler_vulkan_runner COMMAND seedvr2_sampler_vulkan_runner)
        set_tests_properties(seedvr2_sampler_vulkan_runner PROPERTIES LABELS "smoke;vulkan")

        add_executable(seedvr2_vae_vkmat_boundary_runner tests/vae_vkmat_boundary_runner.cpp)
        target_include_directories(seedvr2_vae_vkmat_boundary_runner PRIVATE src)
        target_link_libraries(seedvr2_vae_vkmat_boundary_runner PRIVATE seedvr2_ncnn)

        add_executable(seedvr2_perf_bench tests/perf_bench.cpp)
        target_include_directories(seedvr2_perf_bench PRIVATE src)
        target_link_libraries(seedvr2_perf_bench PRIVATE seedvr2_ncnn)
    endif()
endif()

install(TARGETS seedvr2_ncnn
        EXPORT seedvr2_ncnnTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/seedvr2
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT seedvr2_ncnnTargets
        FILE seedvr2_ncnnTargets.cmake
        NAMESPACE SeedVR2::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/seedvr2_ncnn)
configure_package_config_file(
    cmake/seedvr2_ncnnConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/seedvr2_ncnnConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/seedvr2_ncnn)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/seedvr2_ncnnConfigVersion.cmake
    VERSION 0.1.1
    COMPATIBILITY SameMajorVersion)
install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/seedvr2_ncnnConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/seedvr2_ncnnConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/seedvr2_ncnn)
if(SEEDVR2_BUILD_APPS)
    install(TARGETS seedvr2-ncnn-vulkan
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
