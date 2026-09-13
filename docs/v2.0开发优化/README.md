# SeedVR2 ncnn v2.0 开发优化

本目录只保存 v2.0 阶段的分析、设计、测试方法和实测结果。实现代码仍按职责放在
`src/`、`tests/`、`patches/` 和 `custom_layers/`，不与文档混放。

## 当前阶段

- [VAE decoder 瓶颈与 Convolution3D 优化](VAE解码卷积优化.md)
- [RTX 5090 原始测试数据](性能数据.csv)

当前完成的是第一阶段低风险优化：保持 FP32、pack1 张量布局和原权重格式，单个
Vulkan invocation 并行计算四个输出通道。320×240 与 480×256 的完整 decoder
均接近 2 倍加速，且新旧实现输出 checksum 完全一致。

## 下一阶段建议

1. 用 Vulkan timestamp 对 35 个 Convolution3D 逐层排序，找出高分辨率热点层。
2. 为输入和权重增加真正的 pack4 路径，减少标量权重读取与地址计算。
3. 对 3×3×3、stride=1 的主路径做专用 shader，再评估 shared memory tiling。
4. 最后再考虑 FP16/BF16 和矩阵乘单元；这一步会改变数值特征，应独立验收。
