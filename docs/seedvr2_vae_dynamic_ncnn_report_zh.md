# SeedVR2 VAE 动态 NCNN 转换报告

## 结论

原先示例 1 的 VAE param 固定了 `T/H/W`。现在已经生成一套新的动态模型，同一份 encoder/decoder param 和 bin 可以接收不同的 batch-one `T/H/W`，不需要按视频尺寸重新转换或修改 param。

最终模型目录：

```text
/home/czw1/ncnn_learn/seedvr2_ncnn/ncnn_models/vae_dynamic/
```

文件：

```text
seedvr2_vae_encoder_dynamic.ncnn.param
seedvr2_vae_encoder_dynamic.ncnn.bin
seedvr2_vae_decoder_dynamic.ncnn.param
seedvr2_vae_decoder_dynamic.ncnn.bin
```

SHA-256：

```text
encoder param 34cf03ee094c8b73578b1bf630445cc01b1476025319a2bed1ed383f0e34f908
encoder bin   1ab6b1721de41e050a527b9c07a522f3f0a9578d2ec23246626f21878be211e2
decoder param 1fbe76c4ea9c52450f2cb09af4238ef37d2c085b9e6d0e526cbdfc034a7cb005
decoder bin   94893f4171a5bfc338e4793cc33d565e1b5be0be5fff08feb768629c86c2ee6c
```

## 动态接口

NCNN 不保留 batch=1 这一维，C++ 输入使用 `ncnn::Mat(w=W,h=H,d=T,c=C)`。

Encoder：

```text
PyTorch: [1,3,T,H,W]
NCNN:    [3,T,H,W]
输出:    [32,ceil(T/4),floor(H/8),floor(W/8)]
```

输出前 16 通道为 posterior mean，后 16 通道为 logvar。随机采样和 latent scaling factor `0.9152` 仍在图外。

Decoder：

```text
PyTorch: [1,16,Tlatent,Hlatent,Wlatent]
NCNN:    [16,Tlatent,Hlatent,Wlatent]
输出:    [3,4*Tlatent-3,8*Hlatent,8*Wlatent]
```

若要求 encode 后 decode 严格恢复相同视频尺寸，输入应满足：

```text
T = 4*n + 1
H % 8 == 0
W % 8 == 0
```

Encoder 可以直接接收不满足整除条件的尺寸，但边缘不足一个 8 像素块的部分不会进入 latent，decoder 输出也不会自动恢复原始尺寸。实际应用应先将右侧/底部 pad 到 8 的倍数，并将末尾帧复制到最近的 `4*n+1`，decode 后再 crop 回原始 `T/H/W`。模型文件本身不再绑定某个具体尺寸。

## 转换结构

继续使用 NCNN 原生层的部分：

```text
Convolution3D
Padding / Crop / Concat
BinaryOp
Swish
Reshape / Permute / Slice
```

运行时尺寸无法由静态图表达的部分保留为三个动态层：

1. `DynamicFramewiseGroupNorm`
   - 输入输出为 `C,T,H,W`。
   - 每一帧独立计算 32 组 GroupNorm。
   - 不将运行时 `T` 展开成固定数量的 GroupNorm 节点。
2. `DynamicFramewiseSpatialAttention`
   - 在 VAE bottleneck 对每帧 `H*W` token 独立做空间自注意力。
   - 加载原 GroupNorm、Q/K/V 和 output projection 权重。
3. `DynamicSpaceTimeShuffle`
   - 执行原始 `1x1x1 Conv3D + channel rearrange`。
   - 从运行时输入计算输出 `T/H/W`。
   - 时间上采样时在层内删除重复 head，得到 `2*T-1`，避免动态 Crop 被 pnnx 固定。

因果 3D Conv 没有使用权重 mask。它继续由“复制首帧到时间轴左侧 + 时间 padding 为 0 的原生 Convolution3D”实现，运行时不会读取未来帧。

## 数值测试

PNNX 使用以下两组尺寸推导动态维度：

```text
encoder: [1,3,5,32,32] 和 [1,3,9,48,64]
decoder: [1,16,2,4,4] 和 [1,16,3,6,8]
```

最终验收使用了未参与 PNNX 导出的两个随机输入视频。两者都故意使用奇数宽高和非 `4*n+1` 帧数，以验证同一套模型可以直接接收任意运行时尺寸：

| 随机视频 | 输入形状 | 模块 | 输出形状 | max abs | mean abs | RMSE |
| --- | --- | --- | --- | ---: | ---: | ---: |
| case 0 | `[1,3,6,37,53]` | encoder | `[32,2,4,6]` | `1.66893e-5` | `2.94112e-6` | `3.87865e-6` |
| case 0 | mean `[1,16,2,4,6]` | decoder | `[3,5,32,48]` | `8.18521e-5` | `6.45388e-6` | `8.87502e-6` |
| case 1 | `[1,3,10,41,67]` | encoder | `[32,3,5,8]` | `2.28882e-5` | `2.87299e-6` | `3.89061e-6` |
| case 1 | mean `[1,16,3,5,8]` | decoder | `[3,9,40,64]` | `1.13636e-4` | `7.25689e-6` | `9.82329e-6` |

测试通过标准为 FP32 `max_abs <= 5e-4`。四项均通过。

## 构建和复现

导出：

```bash
python tools/export_seedvr2_vae_dynamic_ncnn.py --component all
```

构建 C++ runner：

```bash
cmake -S cpp_runtime -B cpp_runtime/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp_runtime/build --target seedvr2_vae_dynamic_runner -j 8
```

运行完整随机数值测试：

```bash
python tools/test_seedvr2_vae_dynamic_ncnn.py
```

## 当前限制

- batch 固定为 1。
- 当前是 whole-clip 推理，不包含 streaming `memory_state=ACTIVE` 和跨 clip cache。
- C++ 动态层当前实现并验证的是 FP32 CPU 路径。
- 普通 NCNN 层具备 Vulkan 后端，但三个动态层尚未实现 `VkMat` shader，因此当前 runner 明确关闭 Vulkan。不能把本次结果表述为 Vulkan 动态 VAE 已完成。
- 下一阶段需要为三个动态层补 Vulkan pipeline，优先顺序为 GroupNorm、space-time shuffle、spatial attention，并使用相同随机测试做 CPU/Vulkan/PyTorch 三方对齐。
