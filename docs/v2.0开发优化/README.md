# SeedVR2 ncnn v2.0 开发优化

本目录只保存 v2.0 阶段的分析、设计、测试方法和实测结果。实现代码仍按职责放在
`src/`、`tests/`、`patches/` 和 `custom_layers/`，不与文档混放。

## 当前阶段

- [VAE decoder 瓶颈与 Convolution3D 优化](VAE解码卷积优化.md)
- [RTX 5090 原始测试数据](性能数据.csv)

当前已经完成两阶段低风险优化：第一阶段让一个 Vulkan invocation 并行计算四个
输出通道；第二阶段把对应权重预打包为连续 `vec4`。主张量仍保持 pack1，两个真实
尺寸相对原始实现累计提升约 2.9 倍，且各阶段输出 checksum 完全一致。

- [第一阶段：四输出通道并行](VAE解码卷积优化.md)
- [第二阶段：权重 pack4 向量读取](Convolution3D权重pack4优化.md)

## 下一阶段建议

1. 用 Vulkan timestamp 对 35 个 Convolution3D 逐层排序，找出高分辨率热点层。
2. 让三类 VAE 动态层支持 pack4，再启用主激活张量 pack4，避免层间反复转换。
3. 对 3×3×3、stride=1 的主路径做专用 shader，再评估 shared memory tiling。
4. 最后再考虑 FP16/BF16 和矩阵乘单元；这一步会改变数值特征，应独立验收。
