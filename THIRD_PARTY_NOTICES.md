# 第三方许可说明

本项目的 Apache-2.0 许可只覆盖本仓库自行编写的代码，不改变第三方代码、工具和模型
各自的许可条款。发布二进制或模型时应同时保留对应上游的版权和许可文本。

| 组件 | 用途 | 上游 | 许可 |
|---|---|---|---|
| SeedVR / SeedVR2 | 模型结构、参考实现和原始权重 | https://github.com/ByteDance-Seed/SeedVR | Apache-2.0 |
| ncnn | C++ 推理框架和 Vulkan 后端 | https://github.com/Tencent/ncnn | BSD-3-Clause |
| glslang | 构建期 GLSL 到 SPIR-V 编译 | https://github.com/KhronosGroup/glslang | BSD-3-Clause 等，详见其仓库 |
| FFmpeg | CLI 视频解码和编码（系统进程） | https://ffmpeg.org | LGPL/GPL，取决于系统构建选项 |
| SeedVR2 ncnn models | 转换后的 fp16 推理权重 | https://huggingface.co/vvzc/seedvr2-ncnn-models | Apache-2.0，继承上游模型要求 |
| Big Buck Bunny demo clip | README 真实视频输入与推理对比 | https://peach.blender.org/about/ | CC BY 3.0，(c) copyright 2008, Blender Foundation / www.bigbuckbunny.org |

`ncnn/` 和它的递归子模块保留各自仓库中的许可文件。本项目不把 FFmpeg 源码或二进制
嵌入静态库，命令行程序只调用用户系统中安装的 `ffmpeg` 与 `ffprobe`。
