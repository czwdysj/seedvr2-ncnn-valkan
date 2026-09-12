# 本文件模拟仓库外的最小 CMake 调用方，用于验证安装包能够独立查找和链接。
cmake_minimum_required(VERSION 3.16)
project(seedvr2_install_consumer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(seedvr2_ncnn CONFIG REQUIRED)

add_executable(seedvr2_install_consumer main.cpp)
target_link_libraries(seedvr2_install_consumer PRIVATE SeedVR2::seedvr2_ncnn)
