# SeedVR2 3B PyTorch 到 NCNN CPU 完整转换报告

## 结论

SeedVR2 3B 的 VAE Encoder、VAE Decoder、DiT 输入头、32 个 Transformer
Block 和 DiT 输出头均已转换为 NCNN `.param/.bin`，并在 CPU 上完成动态尺寸、
逐层和完整串联验证。本阶段按最新要求不实现 Vulkan。

最终服务器模型目录：

```text
/root/autodl-tmp/seedvr2_ncnn_models/dit_full_fp16/
/root/autodl-tmp/seedvr2_ncnn_models/vae_dynamic/
```

DiT FP16 存储包约 6.4GB，包含 34 组独立子模型：输入头、32 个 Block、输出头。
VAE 动态模型约 1.0GB，包含 Encoder 和 Decoder。DiT Block 采用流式加载，运行时
只保留一个 Block 权重，避免 3B 权重同时常驻内存。

## 转换方法

### VAE

VAE 先由 TorchScript/pnnx 转换普通计算，再将静态图无法表达的运行时逻辑替换
为三个 CPU 自定义层：

| 模块 | 实现 |
| --- | --- |
| 3D 因果卷积 | `Concat/Crop + Convolution3D`，卷积本身不带 mask |
| 普通激活、切片、重排 | NCNN 原生层 |
| 逐帧 GroupNorm | `DynamicFramewiseGroupNorm` |
| bottleneck 空间注意力 | `DynamicFramewiseSpatialAttention` |
| 时空上采样重排 | `DynamicSpaceTimeShuffle` |

同一套 VAE 文件支持 batch=1 下不同 `T/H/W`。应用层应把输入 pad 到模型有效
尺寸：空间至少为 8 的倍数；需要 encode/decode 严格恢复帧数时使用 `T=4n+1`。

### DiT

完整 `NaDiT.forward` 不能直接交给 pnnx，原因包括 Python `Cache`、动态窗口
slice、list flatten/unflatten、变长 joint attention 和 vid/txt 权重共享切换。
因此按以下边界转换：

| 子模型 | 实现 |
| --- | --- |
| `SeedVR2DiTInput` | 动态 2x2 patchify；Linear 使用原生 `InnerProduct` |
| `SeedVR2DiTBlock` x32 | Linear/MLP 使用 `InnerProduct`；Ada、窗口、MM-RoPE、attention 手写 |
| `SeedVR2DiTOutput` | RMSNorm、output Ada、Linear、动态 2x2 unpatch |

Block 0-9 保存独立 vid/txt 权重，Block 10-31 使用共享权重。奇数 Block 使用
shifted window。Block 31 按 PyTorch 的 `vid_only` 行为跳过文本 Ada/MLP，并
保留源码中的双倍文本残差语义。

PyTorch 在 attention 前显式执行 `.bfloat16()`。CPU 自定义层在 Q/K/V 和
attention 输出边界执行 round-to-nearest-even BF16 舍入，其余 GEMM 保持 FP32，
以贴近原模型而不改变 NCNN `Mat` ABI。

## 动态接口

DiT 输入：

```text
vid       [T*H*W, 33]
txt       [Ltxt, 5120]
timestep  [1]
vid_shape [T,H,W]
```

`H/W` 必须为偶数，输入头输出 `[T*(H/2)*(W/2),2560]`。窗口、文本长度、
patch/unpatch 和输出尺寸均在运行时计算，不绑定 320x240 或 480x270。

VAE 输入采用 NCNN `Mat(w=W,h=H,d=T,c=C)`，不保留 batch=1 维。

## 测试结果

### VAE 动态随机测试

两组尺寸均未参与 pnnx 导出校准，阈值为 `max_abs <= 5e-4`：

| 模块 | 输入 | 输出 | max abs | mean abs |
| --- | --- | --- | ---: | ---: |
| Encoder | `[1,3,6,37,53]` | `[32,2,4,6]` | `4.19617e-5` | `5.34756e-6` |
| Decoder | `[1,16,2,4,6]` | `[3,5,32,48]` | `7.43121e-5` | `7.95422e-6` |
| Encoder | `[1,3,10,41,67]` | `[32,3,5,8]` | `5.53131e-5` | `5.51284e-6` |
| Decoder | `[1,16,3,5,8]` | `[3,9,40,64]` | `8.49962e-5` | `7.31824e-6` |

### DiT 64 次逐层测试

两个真实视频参考分别产生 patch 网格 `[2,15,20]` 和 `[2,16,30]`。每个示例
对 32 个 Block 独立执行，共 64 次，全部满足 `NRMSE <= 0.01`。实际最坏
NRMSE 低于 `0.00144`。测试 JSON 位于服务器 FP32 验证目录的
`validation_all_blocks.json`。

### 完整 DiT 串联

| 示例 | 输出 | CPU 时间 | NRMSE | cosine |
| --- | --- | ---: | ---: | ---: |
| 320x240 | `[2400,16]` | 59 秒 | `0.016998` | `0.999855` |
| 480x270（有效高 256） | `[3840,16]` | 79 秒 | `0.012347` | `0.999924` |

完整串联误差高于独立 Block，是 32 层 attention BF16 舍入、CUDA/CPU GEMM 和
SDPA 实现差异的累积；方向和最终解码结果保持高度一致。

### 真实视频 CPU 路径

两个 5 帧示例均已依次运行 NCNN VAE Encoder、完整 NCNN DiT、Euler 一步
endpoint 和 NCNN VAE Decoder：

| 示例 | Encoder | DiT | Decoder | 最终 PSNR | mean abs |
| --- | ---: | ---: | ---: | ---: | ---: |
| 320x240 | 209 秒 | 59 秒 | 566 秒（8 线程） | `48.93 dB` | `0.003353` |
| 480x270（有效高 256） | 344 秒 | 79 秒 | 332 秒（25 线程） | `55.79 dB` | `0.002387` |

服务器输出：

```text
/root/autodl-tmp/seedvr2_reference_fp32/example_001_320x240/ncnn_cpu_final_video.mp4
/root/autodl-tmp/seedvr2_reference_fp32/example_002_480x270/ncnn_cpu_final_video.mp4
```

## 自定义层报告

每个自定义层的独立报告位于 `docs/custom_layers/`：

```text
dynamic_framewise_group_norm_zh.md
dynamic_framewise_spatial_attention_zh.md
dynamic_space_time_shuffle_zh.md
seedvr2_dit_input_zh.md
seedvr2_dit_block_zh.md
seedvr2_dit_output_zh.md
```

## 构建与复现

```bash
cmake -S cpp_runtime -B cpp_runtime/build \
  -DNCNN_SOURCE_DIR=/path/to/ncnn -DCMAKE_BUILD_TYPE=Release
cmake --build cpp_runtime/build -j 8

python tools/test_seedvr2_vae_dynamic_ncnn.py \
  --runner cpp_runtime/build/seedvr2_vae_dynamic_runner \
  --model-dir ncnn_models/vae_dynamic

python tools/test_seedvr2_dit_ncnn.py \
  --reference-dir /path/to/example_001_320x240 \
  --reference-dir /path/to/example_002_480x270 \
  --model-dir /path/to/dit_full_fp16 \
  --block-runner cpp_runtime/build/seedvr2_dit_block_runner \
  --full-runner cpp_runtime/build/seedvr2_dit_full_runner \
  --output validation_dit.json
```

## 当前限制

- batch 固定为 1，文本 embedding 预计算。
- VAE 为 whole-clip 动态推理，不包含跨 clip streaming cache。
- DiT CPU attention 是正确性基线，不是高性能实现。
- 正常 50-step、CFG=7.5 需要每步正负两次 DiT，纯 CPU 时间很长；本次真实视频
  使用与参考一致的 1-step、CFG=1 验证完整数据路径。
- Vulkan 按当前要求暂不实现。
