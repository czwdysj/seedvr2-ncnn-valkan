# SeedVR2 PyTorch 到 NCNN 转换阶段报告

本文记录当前 SeedVR2 3B super-resolution 路线从 PyTorch 转向 ncnn 的实际检查结果、pnnx 转换探针结果、不可直接转换模块和下一步实现计划。当前目标固定为：SeedVR2 3B、batch size 1、预计算文本嵌入、使用 `test_vectors` 中两个参考样例做数值对齐。

## 结论

转换阶段已经开始。VAE 的 batch=1、动态 `T,H,W`、整段推理版本已经生成可加载的 ncnn `.param/.bin`；完整 SeedVR2 的 DiT 仍不能按“整网一次 pnnx 转换”的方式处理。DiT 的动态窗口、cache、list/unflatten 和 varlen attention 需要拆成原生子图与 C++/自定义层。

推荐转换边界如下：

1. `预处理 + shape 调度`：C++ 手写，不走 pnnx。
2. `VAE encode/decode`：动态整段模式已转换；普通计算使用 ncnn 原生层，运行时尺寸逻辑使用三个项目自定义层，streaming memory 暂未进入首版。
3. `DiT patch in/out、Linear、MLP、RMSNorm、GELU/SwiGLU`：可拆成 pnnx 子图或手工映射 ncnn 原生层。
4. `DiT NaSwinAttention`：不建议 pnnx，必须自定义核心层或拆成多个固定 shape runtime kernel。
5. `CFG、Euler sampler、flatten/unflatten`：C++ 手写，直接对齐 `.pt`。

## 已实际转换成功的内容

### VAE 动态形状模型

最终模型位于：

```text
/home/czw1/ncnn_learn/seedvr2_ncnn/ncnn_models/vae_dynamic/
```

包含：

```text
seedvr2_vae_encoder_dynamic.ncnn.param
seedvr2_vae_encoder_dynamic.ncnn.bin
seedvr2_vae_decoder_dynamic.ncnn.param
seedvr2_vae_decoder_dynamic.ncnn.bin
```

动态输入/输出约定：

| 模型 | 输入 | 输出 |
| --- | --- | --- |
| encoder | `[1,3,T,H,W]`，`B,C,T,H,W` | `[1,32,ceil(T/4),floor(H/8),floor(W/8)]` posterior moments |
| decoder | `[1,16,Tl,Hl,Wl]`，未乘 `0.9152` 的 latent | `[1,3,4*Tl-3,8*Hl,8*Wl]` 视频张量 |

encoder 输出 32 通道，其中前 16 通道是 mean，后 16 通道是 logvar。随机采样与 `0.9152` latent scale 保留在 ncnn 图外，便于确定性验证。

实际落到 ncnn 的原生关键层包括：

```text
Convolution3D
Swish
Crop / Slice / Concat / Reshape / Permute
BinaryOp
```

运行时尺寸无法由静态 param 表达的部分实现为三个项目自定义层：

1. `DynamicFramewiseGroupNorm`：按运行时 `T` 对每一帧独立做 GroupNorm。
2. `DynamicFramewiseSpatialAttention`：按运行时 `T/H/W` 对每帧空间 token 做自注意力。
3. `DynamicSpaceTimeShuffle`：执行 `1x1x1 Conv3D + channel rearrange`，并动态计算输出尺寸。

因果头扩展使用复制首帧和原生 `Convolution3D`，3D 卷积权重本身不带 mask。两个最终模型的 ncnn `load_param/load_model` 均成功。三个自定义层目前只完成 FP32 CPU 后端，Vulkan 后端尚未实现。

测试结果：

| 模型 | 测试形状 | max abs diff | mean abs diff |
| --- | --- | ---: | ---: |
| encoder | `[1,3,6,37,53]` | `1.66893e-5` | `2.94112e-6` |
| decoder | `[1,16,2,4,6]` | `8.18521e-5` | `6.45388e-6` |
| encoder | `[1,3,10,41,67]` | `2.28882e-5` | `2.87299e-6` |
| decoder | `[1,16,3,5,8]` | `1.13636e-4` | `7.25689e-6` |

完整动态接口、尺寸约束和复现方法见 `docs/seedvr2_vae_dynamic_ncnn_report_zh.md`。

### 普通 MLP 探针

已用本机可用 pnnx：

```text
/home/czw1/ncnn_learn/Penguin-VL-ncnn/.venv/bin/pnnx
```

完成了一个代表性普通子图转换探针：

```text
/home/czw1/ncnn_learn/seedvr2_ncnn/pnnx_export/probes/mlp_linear_gelu/
```

输出文件：

```text
probe_mlp_linear_gelu.pt
probe_mlp_linear_gelu.pnnx.param
probe_mlp_linear_gelu.pnnx.bin
probe_mlp_linear_gelu.pnnx.onnx
probe_mlp_linear_gelu.ncnn.param
probe_mlp_linear_gelu.ncnn.bin
probe_mlp_linear_gelu_pnnx.py
probe_mlp_linear_gelu_ncnn.py
probe_mlp_linear_gelu_reference.pt
```

ncnn param 关键层：

```text
Input
Gemm
GELU
Gemm
```

这说明 SeedVR2 中类似 `Linear -> GELU -> Linear` 的普通 MLP 子图可以通过 pnnx 转成 ncnn。该探针验证的是算子路径，不是完整 SeedVR2 权重转换。

## 当前还没有完整转换成功的内容

目前没有完成完整 SeedVR2 3B 模型的 `.param/.bin` 转换。原因不是权重或参考数据缺失，而是模型结构决定不能直接全图 pnnx。

直接转换完整 `NaDiT.forward` 的主要阻塞点：

- `Cache` 对象参与运行时 shape/cache 控制，不是纯 Tensor 图。
- `na.flatten` / `na.unflatten` 返回 list 和 shape 张量，pnnx 很难稳定表达。
- `NaSwinAttention` 在 Python 层动态生成窗口 slice，并依赖输入 `T,H,W`。
- `FlashAttentionVarlen` 使用 `cu_seqlens` 变长 attention，ncnn 没有等价原生层。
- `NaMMRotaryEmbedding3d` 会按视频窗口和文本长度生成 mmRoPE 频率，且视频 token 与文本 token 使用不同位置规则。
- `MMModule` 在 vid/txt 两个分支之间切换共享权重或独立权重，直接 trace 后图结构不适合维护。
- `AdaSingle` 根据 `hid_len` repeat timestep embedding，再做 shift/scale/gate，属于动态调度。

VAE 当前尚未覆盖的运行模式：

- streaming `memory_state=ACTIVE` 与跨 clip cache。
- VAE slicing/memory limit 和 sequence-parallel 路径。
- batch 大于 1。
- 三个动态自定义层的 Vulkan 后端。

## 需要自定义或手写的层/模块

### 1. `SeedVR2WindowPartition`

对应 PyTorch：

```text
pytorch_model/models/dit_v2/window.py
pytorch_model/models/dit_v2/nablocks/attention/mmattn.py
```

功能：

- 根据 `720pwin_by_size_bysize` 或 `720pswin_by_size_bysize` 计算窗口。
- 对 flattened video token 做 window partition。
- 保存 reverse index，用于 attention 后恢复原始 token 顺序。

输入/输出建议：

```text
输入：vid_qkv [L, 3*heads*head_dim], vid_shape [B,3]
输出：vid_qkv_win [Lwin, 3*heads*head_dim], window_shape, window_count, reverse_index
```

### 2. `SeedVR2RepeatConcatText`

对应 PyTorch：

```text
na.repeat_concat_idx(...)
```

功能：

- 每个视频窗口拼接完整文本 token。
- 输出 joint attention 的拼接 token。
- attention 后拆回 video/text，并对重复文本输出做 coalesce/聚合。

这是视频窗口 attention 和文本融合的核心。

### 3. `SeedVR2MMRoPE3D`

对应 PyTorch：

```text
pytorch_model/models/dit_v2/rope.py
```

功能：

- 为视频窗口 token 生成 3D RoPE。
- 为文本 token 生成语言 RoPE，并 repeat 到窗口级。
- 对 q/k 执行 rotary embedding。

注意：示例 1 和示例 2 的 `L` 不同，必须按 `vid_shape/txt_shape/window_shape` 动态生成或预计算频率表。

### 4. `SeedVR2VarlenWindowAttention`

对应 PyTorch：

```text
FlashAttentionVarlen
```

功能：

- 对每个 `[video_window_tokens + text_tokens]` 段做 self-attention。
- 使用 `cu_seqlens` 控制变长 segment。
- 输出后拆回 video/text。

ncnn 原生没有该层。CPU 版本可以先用普通 matmul + softmax 实现；Vulkan 版本需要单独优化。

### 5. `SeedVR2AdaSingle`

对应 PyTorch：

```text
pytorch_model/models/dit_v2/modulation.py
```

功能：

- 从 timestep embedding 中取出每层每分支的 shift/scale/gate。
- 按 `hid_len` repeat 到 video/text token 长度。
- 对 attention/MLP 前后做调制。

可选择手写为 C++ runtime 函数，不一定要做成 ncnn layer。

### 6. `SeedVR2CausalConv3D`（仅 streaming 版本可能需要）

对应 PyTorch：

```text
pytorch_model/models/video_vae_v3/modules/inflated_layers.py
```

功能：

- 时间维 causal padding。
- 可选 memory state。
- 兼容 inflation 后的 3D 权重。

固定整段版本已使用 `Crop + Concat + Convolution3D` 精确表达，不需要该自定义层。只有后续跨 clip memory/cache 版本可能需要状态型自定义实现。

## 建议拆分后的转换单元

### DiT 部分

| 单元 | 转换方式 | 当前状态 |
| --- | --- | --- |
| `txt_in Linear + norm` | pnnx 或手工 ncnn | 待转换 |
| `TimeEmbedding` | pnnx 或手工 ncnn | 待转换 |
| `NaPatchIn/NaPatchOut` | 固定 shape 手写 reshape + InnerProduct | 待转换 |
| `RMSNorm/LayerNorm` | ncnn 原生或自定义 RMSNorm | 待转换 |
| `Linear qkv/out` | pnnx / ncnn InnerProduct | 代表性 Linear 已通过探针 |
| `MLP/SwiGLUMLP` | pnnx / ncnn 原生算子组合 | `Linear+GELU+Linear` 探针已成功 |
| `NaSwinAttention` | 自定义层/手写 runtime | 必须自定义 |
| `AdaSingle` | 手写 runtime 或自定义层 | 必须自定义 |
| `CFG + sampler` | C++ 手写 | 待实现 |

### VAE 部分

| 单元 | 转换方式 | 当前状态 |
| --- | --- | --- |
| 普通 Conv/Norm/Act | pnnx + ncnn 原生层 | 已转换 |
| ResBlock | pnnx + ncnn 原生层 | 已转换 |
| Causal 3D Conv temporal padding | `Crop + Concat + Convolution3D` | 固定整段模式已转换 |
| framewise GroupNorm/Attention | `Slice + GroupNorm/MHA + Concat` | 已转换 |
| space-time upsample | `Concat + Deconvolution3D` | 已转换 |
| memory state | C++ runtime 管理 | streaming 版本待实现 |
| slicing/memory limit | 第一版禁用，固定整段推理 | PyTorch reference 已禁用 |

## 对齐标准文件位置

示例 1：

```text
/home/czw1/ncnn_learn/seedvr2_ncnn/test_vectors/example_001_320x240
```

示例 2：

```text
/home/czw1/ncnn_learn/seedvr2_ncnn/test_vectors/example_002_480x270
```

每个目录都包含同名 `.pt` 文件。转换后应该按下面顺序对齐：

1. 预处理：

```text
input_video_raw.pt
input_video_transformed.pt
input_video_cut.pt
```

2. VAE encode：

```text
vae_encode_latent.pt
```

3. 条件和噪声：

```text
initial_noise.pt
augment_noise.pt
condition.pt
```

4. DiT 输入：

```text
dit_input_pos_step000.pt
latents_shape.pt
text_pos_emb.pt
text_neg_emb.pt
text_pos_shape.pt
text_neg_shape.pt
```

5. DiT block：

```text
block_00_input_vid.pt
block_00_input_txt.pt
block_00_output_vid.pt
block_00_output_txt.pt
block_15_input_vid.pt
block_15_input_txt.pt
block_15_output_vid.pt
block_15_output_txt.pt
block_31_input_vid.pt
block_31_input_txt.pt
block_31_output_vid.pt
block_31_output_txt.pt
```

6. DiT 输出、CFG、采样：

```text
dit_output_pos_step000.pt
dit_output_neg_step000.pt
cfg_output_step000.pt
sampler_latent_after_step000.pt
```

7. VAE decode 和最终视频：

```text
vae_decode_input.pt
vae_decode_output.pt
final_video_tensor.pt
final_video_uint8_thwc.pt
final_video.mp4
metadata.json
```

## 推荐执行顺序

第一阶段先不要转完整模型，而是把 DiT 的一个 block 拆开：

1. 先实现 `RMSNorm + AdaSingle + MLP/SwiGLU`，对齐 `block_00_input_*` 到 MLP 前后的局部输出。
2. 再实现 `WindowPartition + RepeatConcatText + MMRoPE3D + VarlenAttention`，对齐 `block_00_output_vid.pt` 和 `block_00_output_txt.pt`。
3. block 0 对齐后，复用同一实现跑 block 15 和 block 31。
4. 再串 32 层 DiT，对齐 `dit_output_pos_step000.pt` 和 `dit_output_neg_step000.pt`。
5. 最后接 CFG/sampler/VAE decode，对齐 final tensors。

VAE 建议单独作为第二条线：

1. 固定示例 1 的 `vae_encode_latent.pt` 做 encode 对齐。
2. 固定 `vae_decode_input.pt` 做 decode 对齐。
3. 先禁用 slicing/memory limit，保持和当前 reference 一致。

## 当前风险

- 3B 权重大，完整模型拆分导出需要谨慎控制内存和磁盘。
- pnnx 对 trace 中的动态 Python 控制流不会给出稳定、可维护的 ncnn 图。
- 即便某些动态逻辑被 trace 成固定 shape 图，也不建议直接使用，因为示例 1 和示例 2 shape 不同，会导致图不可泛化。
- 首版 ncnn runtime 应该支持固定两组 shape，通过 `.pt` 对齐后再抽象泛化。

## 本轮测试

已执行：

```text
/home/czw1/ncnn_learn/Penguin-VL-ncnn/.venv/bin/pnnx probe_mlp_linear_gelu.pt inputshape=[16,128] fp16=0 optlevel=2
```

结果：

- 成功生成 `probe_mlp_linear_gelu.ncnn.param`
- 成功生成 `probe_mlp_linear_gelu.ncnn.bin`
- ncnn 图包含 `Gemm -> GELU -> Gemm`

VAE 测试：

```text
python tools/export_seedvr2_vae_dynamic_ncnn.py --component all
python tools/test_seedvr2_vae_dynamic_ncnn.py
```

结果：

- 同一套动态 encoder/decoder 模型完成两组未参与导出的随机尺寸测试。
- 四项 PyTorch/NCNN FP32 对齐结果均满足 `max_abs <= 5e-4`。
- encoder/decoder 的 `load_param` 和 `load_model` 均成功。
- 当前验证是 CPU 路径，不代表 Vulkan 已完成。

受环境限制未完成：

- 未执行完整 SeedVR2 3B 全图 pnnx，因为根据代码审计，该路线不可维护且预计会卡在动态窗口 attention、Cache、varlen attention 和 VAE causal memory。
- 原服务器 `connect.westd.seetacloud.com:41325` 当前已关闭连接，无法在本轮继续远端验收。

## 下一步计划

下一步为 VAE 的三个动态自定义层实现 Vulkan pipeline，并完成 PyTorch、NCNN CPU、NCNN Vulkan 三方对齐。同时创建第一个真实 DiT 子图：`DiT block 内的 MLP/SwiGLU 分支`，验证权重抽取、ncnn InnerProduct、SwiGLU、RMSNorm/Ada 调制和 `.pt` 对齐流程，再推进窗口 attention 自定义层。
