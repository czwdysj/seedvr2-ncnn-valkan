# SeedVR2 自定义层全景：六个自定义层在做什么、为什么需要、怎么算

本文是六个自定义层的统一参考文档，服务于两个目的：

1. 讲清楚每个层的**职责、解决的问题、计算公式和动态逻辑**，并对应到
   本仓库的 C++ 实现与 `pytorch_model/` 下的 PyTorch 参考代码；
2. 作为 Vulkan 化（P2-P4 阶段）的需求基线：GPU shader 必须复现完全相同的
   动态语义和数值约定。

各层的更细致的专项报告见同目录下 `<layer>_zh.md`；本文是它们的总纲。

---

## 0. 总览与切分原则

| # | 自定义层 | 所属 | 实例数 | 一句话职责 |
| --- | --- | --- | ---: | --- |
| 1 | `DynamicFramewiseGroupNorm` | VAE | ~50 | 逐帧、分组的归一化（统计范围 = 单帧 × 单组） |
| 2 | `DynamicFramewiseSpatialAttention` | VAE | 1 | bottleneck 逐帧单头空间自注意力（内嵌逐帧 GroupNorm） |
| 3 | `DynamicSpaceTimeShuffle` | VAE | 3 | decoder 上采样：1×1×1 Conv3D + 通道→时空重排（含 2T−1 首帧删除） |
| 4 | `SeedVR2DiTInput` | DiT | 1 | 2×2 patchify + 文本投影 + timestep → Ada embedding |
| 5 | `SeedVR2DiTBlock` | DiT | 32 | 完整多模态 Transformer block（动态窗口 attention 所在） |
| 6 | `SeedVR2DiTOutput` | DiT | 1 | affine RMSNorm + 输出 Ada + 投影 + unpatchify |

### 0.1 判断「要不要自定义」的原则

pnnx 按示例尺寸 trace，ncnn 原生算子按固定语义计算。逐层判断标准：

1. **纯计算、形状静态** → 原生算子（Conv3D、InnerProduct、SiLU、Linear…）；
2. **归约/重排的索引边界依赖运行时 T/H/W** → 自定义层（第 1、3、4 层的 patchify）；
3. **融合语义 + 动态序列**（attention 与 norm/RoPE/残差共享动态上下文）→ 自定义层（第 2、5 层）;
4. **大矩阵乘** → 即使在自定义层内部，也复用 ncnn 原生 `InnerProduct`
   （`create_layer("InnerProduct")` 手动创建），自定义代码只做编排。

第 4 条是 Vulkan 化的结构基础：原生 InnerProduct 有现成 GPU 实现，
自定义 shader 只需覆盖动态编排部分。

### 0.2 C++ ↔ PyTorch 代码映射

| 自定义层 | C++ | PyTorch 参考 |
| --- | --- | --- |
| FramewiseGroupNorm | `custom_layers/dynamic_framewise_group_norm.cpp` | `pytorch_model/models/video_vae_v3/modules/attn_video_vae.py`（ResnetBlock3D 的 GroupNorm） |
| FramewiseSpatialAttention | `custom_layers/dynamic_framewise_spatial_attention.cpp` | 同上（VAE attention block） |
| SpaceTimeShuffle | `custom_layers/dynamic_space_time_shuffle.cpp` | 同上（`Upsample3D`，upscale_ratio 处） |
| DiTInput | `custom_layers/seedvr2_dit_input.cpp` | `pytorch_model/models/dit/patch.py`（`NaPatchIn`）+ `nadit.py`（timestep MLP） |
| DiTBlock | `custom_layers/seedvr2_dit_block.cpp` | `pytorch_model/models/dit/nablocks/mmsr_block.py` + `blocks/mmdit_window_block.py` + `na.py` + `rope.py` |
| DiTOutput | `custom_layers/seedvr2_dit_output.cpp` | `patch.py`（`NaPatchOut`）+ `mmsr_block.py` final layer |

### 0.3 全部自定义层共享的工程约定

- 张量统一用 `ncnn::Mat(w=W, h=H, d=T, c=C)` 表示逻辑 `[C,T,H,W]` 视频，
  token 序列用 `Mat(w=dim, h=L)` 表示 `[L,dim]`；
- `load_param` 从导出脚本写入的 shape 参数（param 10/11 等）恢复通道数，
  避免在 param 里维护冗余易错的 channels 字段；
- `load_model` 按 pnnx 写 bin 的属性顺序显式逐项加载，顺序错位会静默失真；
- 数值对齐优先：关键归约用 double 累加，attention 前模拟 bfloat16 舍入；
- 所有权重以 FP32 存储（目录名 `dit_full_fp16` 指 PyTorch 侧来源精度）。

---

## 1. DynamicFramewiseGroupNorm —— 逐帧分组归一化

### 1.1 职责

VAE 残差块内所有归一化（约 50 处）。输入输出同形状：`Mat(W,H,T,C)` ↔ 逻辑 `[C,T,H,W]`。

### 1.2 解决的问题

PyTorch 原始实现：

```python
x [B,C,T,H,W] -> permute [B,T,C,H,W] -> reshape [B*T,C,H,W]
              -> nn.GroupNorm(32) -> reshape/permute 回 [B,C,T,H,W]
```

即**每帧是一个独立样本**。两个原生路径都走不通：

- ncnn 原生 GroupNorm 对 4D Mat 的统计范围是 `w*h*d`（含全部帧），帧间互相污染；
- pnnx 把 `B*T` 拆帧固化为 trace 时的常量，换 T 后图失效
  （实测验证见 `test_demo/framewise_groupnorm_probe.cpp`：直接替换 max|diff|=0.344，
  原生组合图换 T 后静默输出错误形状）。

自定义的是**逐帧统计边界**，不是 GroupNorm 公式本身。

### 1.3 计算公式

对每帧 `t`、每组 `g`（`groups=32`，`eps=1e-6`），统计范围 = 组内通道 × 该帧全部空间位置：

```text
count     = (C/32) · H · W
μ(t,g)    = (1/count) · Σ x[c,t,h,w]           （c ∈ 组 g，遍历 h,w）
σ²(t,g)   = E[x²] − μ²                          （有偏方差，double 累加）
y[c,t,h,w] = γc · (x − μ(t,g)) / √(σ²(t,g)+ε) + βc
```

### 1.4 动态逻辑

- `T/H/W` 完全运行时决定，不进 param；
- `channels` 从 param 10 中 bias 的 shape 恢复，校验 `channels % 32 == 0`；
- bin 顺序：先 bias 后 weight（pnnx 属性顺序约定）。

---

## 2. DynamicFramewiseSpatialAttention —— VAE 瓶颈逐帧空间注意力

### 2.1 职责

VAE bottleneck 的自注意力块，对每一帧独立做**单头**空间注意力，内嵌逐帧 GroupNorm。

### 2.2 解决的问题

- token 数 `N = H·W` 由运行时尺寸决定，attention 矩阵 `[N,N]` 大小动态，
  pnnx 无法稳定导出 reshape/softmax 链；
- ncnn 原生 MultiHeadAttention 无法匹配：权重布局、单头缩放（1/√C 而非 1/√head_dim）、
  内嵌 GroupNorm、残差边界都不同；
- 必须**严格不跨帧**：把 `[T·H·W, C]` 当一条序列会让帧间互相可见，
  正确语义是按帧的块对角 attention。自定义层用显式 frame 循环保证这一点。

### 2.3 计算公式

对帧 `t`，`N = H·W`，`n = GroupNorm32(x[t])`（eps=1e-6）：

```text
Q = n·Wq + bq,  K = n·Wk + bk,  V = n·Wv + bv        # [N,C]，W 为 C×C
A = softmax( Q·Kᵀ / √C )                              # 单头：缩放 1/√C
y[t] = x[t] + (A·V)·Wout + bout                       # 残差
```

点积与 softmax 分母用 double 累加对齐 PyTorch FP32 参考。

### 2.4 动态逻辑

- 每帧单独展开为 `[N,C]` token，逐帧循环内完成 norm→QKV→attention→投影→残差；
- bin 权重顺序：`norm_bias, norm_weight, k_bias, k_weight, out_bias, out_weight,
  q_bias, q_weight, v_bias, v_weight`（pnnx 按模块属性名字典序外的固定约定写入）。

---

## 3. DynamicSpaceTimeShuffle —— decoder 时空上采样

### 3.1 职责

VAE Decoder 上采样 = learned 1×1×1 Conv3D + MAGViT 风格 channel-to-space-time 重排。
空间倍率固定 `s=2`。三个实例：

| 实例 | 输入通道 | projection 通道 | 时间倍率 r_t | 输出通道 |
| --- | ---: | ---: | ---: | ---: |
| upsample 0 | 512 | 4096 | 2 | 512 |
| upsample 1 | 512 | 4096 | 2 | 512 |
| upsample 2 | 256 | 1024 | 1 | 256 |

### 3.2 解决的问题

- ncnn PixelShuffle 只表达空间二维重排，表达不了 `C·s²·r_t → C` 的三轴重排；
- 输出时间长度是**动态表达式 `r_t·T − 1`**，pnnx 的 Crop/Reshape 绑定 trace 时的 T；
- 删除的不是尾帧而是 **raw frame 1**（首帧复制产生的重复位），任何 resize/尾部
  裁剪都替代不了，必须在通道→时间映射时精确跳过。

### 3.3 计算公式

第一步，1×1×1 Conv3D（只混通道，不读邻域）：

```text
z[pc, t, y, x] = Σ_ci W[pc, ci] · x[ci, t, y, x] + b[pc]
pc ∈ [0, C·s²·r_t)
```

第二步，把投影通道索引解码为输出时空偏移：

```text
pc = ((oy·2 + ox)·r_t + ot)·C + c        # oy,ox ∈ {0,1}；ot ∈ [0, r_t)
raw_t = t·r_t + ot
输出位置 = [c, t', 2y+oy, 2x+ox]
t' = raw_t            (r_t == 1 或 raw_t < 1)
t' = 跳过 raw_t == 1   （首帧复制位，直接丢弃）
t' = raw_t − 1        (raw_t > 1，后续整体左移一位)
```

输出形状：`[C, r_t·T − 1 (r_t>1), 2H, 2W]`。这就是 VAE「5 帧 latent 解码出
`1+4·(T−1)` 帧」的来源。

### 3.4 动态逻辑

- `r_t` 从 param 反推：`projected_channels / (in_channels · s²)`，不写死；
- `r_t = 1` 时退化为纯空间上采样（第三个实例）。

---

## 4. SeedVR2DiTInput —— patchify 与条件 embedding

### 4.1 职责

DiT 第一段。四个输入 `vid[L_v,33]`、`txt[L_t,5120]`、`timestep[1]`、`vid_shape[T,H,W]`，
四个输出 `vid_out[L_v',2560]`、`txt_out[L_t,2560]`、`emb[15360]`、`patched_shape[T,H/2,W/2]`。

### 4.2 解决的问题

- patchify 的 einops 重排 `(T t)(H h)(W w) c -> T H W (t h w c)` 依赖运行时 `vid_shape`，
  pnnx 无法对 `L_v = T·H·W` 的动态 token 数稳定导出；
- timestep embedding 链含固定结构的 MLP + 逐元素 SiLU，与动态 patchify 一起打包为
  一层，保持 ncnn 图极简（DiT 图每段只有一个节点）。

### 4.3 计算公式

**patchify（2×2，空间）**：token `p = (t, y, x)`（patch 网格坐标）按 `(t h w c)` 顺序
拼接 4 个空间位置各 33 通道：

```text
patches[p] = concat( vid[(t·H + 2y+dy)·W + 2x+dx]  for dy∈{0,1}, dx∈{0,1} )   # [132]
vid_out[p] = W_v · patches[p] + b_v                                            # Linear(132→2560)
txt_out    = W_t · txt + b_t                                                   # Linear(5120→2560)
```

对应 `patch.py` 的 `rearrange(vid, vid_shape, "(T t) (H h) (W w) c -> T H W (t h w c)")`，
`t=1`、`h=w=2`。

**timestep → Ada embedding**（对应 `nadit.py` 的 timestep embedding）：

```text
f_i = exp( −ln(10000) · i / 128 ),  i ∈ [0,128)
s   = [ sin(τ·f_0..f_127) ; cos(τ·f_0..f_127) ]            # [256]
emb = L3( SiLU( L2( SiLU( L1(s) ) ) ) )                     # 256→2560→2560→15360
```

`emb[15360] = 2560 × 2 × 3` 是全部 32 个 block 共享的 Ada 参数池：
维度 `c`、支路 `li`（0=attn, 1=mlp）、分量（0=shift, 1=scale, 2=gate）。

### 4.4 动态逻辑

- `H/W` 必须为偶数（2×2 patchify 约束），`L_v' = T·(H/2)·(W/2)`；
- `patched_shape` 以 int32 blob 输出，供后续 block 的窗口划分使用；
- 五个 InnerProduct 均在 `create_pipeline` 中初始化（Vulkan 化时在此挂 GPU pipeline）。

---

## 5. SeedVR2DiTBlock —— 多模态 Transformer block（窗口 attention 所在）

### 5.1 职责与固定配置

完整的多模态 Transformer block，32 个实例共享同一 C++ 类。固定配置：
`dim=2560`、`heads=20`、`head_dim=128`、`mlp_hidden=6912`、`norm_eps=1e-5`。
输入 `vid[L_v,2560]`、`txt[L_t,2560]`、`emb[15360]`、`vid_shape[T,H,W]`；
输出形状不变。**大矩阵乘全部复用原生 InnerProduct**（QKV 2560→7680、proj_out
2560→2560、SwiGLU 2560→6912→6912→2560 无 bias）。

### 5.2 解决的问题

整个 block 无法用原生算子组合表达，原因集中在 attention 的动态部分：

- 窗口划分的数量/形状/偏移由运行时 T/H/W 决定，窗口索引无法跨 blob 传递；
- QK-norm + MM-RoPE + bfloat16 舍入 + 文本跨窗口汇聚是融合语义；
- block 10+ 权重共享、block 31 `vid_only` 特例等 cache 兼容行为。

### 5.3 forward 流程（vid / txt 双支路，`shared_weights` 时共用权重）

```text
# attention 支路
a_x = Ada_in( RMSNorm(x), emb, attn_shift, attn_scale, slot=0 )    # x ∈ {vid, txt}
QKV_x = Linear(a_x)                                                # 2560→7680
vid_attn = WindowAttention(QKV_vid, QKV_txt, vid_shape)            # §5.4
x' = Ada_out( proj_out(vid_attn), vid, emb, attn_gate, slot=0 ) + vid
（vid_only 块：txt 只做 RMSNorm 与外层残差，无 attention Ada）

# MLP 支路（SwiGLU）
m = Ada_in( RMSNorm(x'), emb, mlp_shift, mlp_scale, slot=1 )
x'' = Ada_out( SwiGLU(m), x', emb, mlp_gate, slot=1 ) + x'
```

**Ada 公式**（slot = `layer_index·3`，attn `li=0`，mlp `li=1`）：

```text
Ada_in:  y[c] = x[c] · (emb[c·6 + li·3 + 1] + scale[c]) + emb[c·6 + li·3] + shift[c]
Ada_out: y[c] = x[c] · (emb[c·6 + li·3 + 2] + gate[c])  + residual[c]
```

### 5.4 动态窗口划分（`make_windows`）

```text
scale       = √(3600 / (H·W))                       # 自适应缩放到基准面积
resized_h   = round_ties_to_even(H · scale)          # Python round 语义 → nearbyint
resized_w   = round_ties_to_even(W · scale)
window_h    = ceil(resized_h / 3)                    # 空间窗约 3×3
window_w    = ceil(resized_w / 3)
window_t    = ceil(min(T, 30) / 4)                   # 时间窗约 4 帧，封顶 30 帧

普通窗口：  窗口 i 覆盖 [i·w, min((i+1)·w, D))，数量 = ceil(D / w)
shifted：   网格偏移半窗。shift=0.5（当窗口数 < 维度长度时）
            窗口 i 覆盖 [max(0,(i−0.5)·w), min((i+0.5)·w, D))，
            数量 = ceil((D − 0.5)/w) + 1
            视频窗口仍是不相交划分，仅网格错位、边界窗口被截断
```

对应 PyTorch `mmsr_block.py` 的 `window_op((t,h,w), window)` + `na.window_idx`。

### 5.5 窗口 attention

每个窗口内的序列 = 窗口内视频 token + **全部文本 token**（PyTorch 侧文本被
`repeat_concat_idx` 复制到每个窗口）。对每个 head（20 个）：

```text
# QK-norm + MM-RoPE（q 和 k 都做；v 只做 bfloat16 舍入）
q_h = bf16( RoPE( RMSNorm(q_h) · norm_q , pos ) )
k_h = bf16( RoPE( RMSNorm(k_h) · norm_k , pos ) )

# 位置编码（MM-RoPE 的“多模态”含义：文本与视频共用一套频率、不同坐标）
视频 token: pos = ( L_txt + (t − t0),  y − y0,  x − x0 )     # 窗口局部坐标 + 文本偏移
文本 token: pos = ( l, l, l )                                 # l = 文本 token 序号

# RoPE：3 轴 × 21 对频率（rope_freqs[21] 从 bin 加载，head_dim=128 中使用 126 维）
for axis ∈ {t,h,w}, pair ∈ [0,21):
    d = axis·42 + pair·2
    angle = pos[axis] · freqs[pair]
    (q[k][d], q[k][d+1]) ← (q·cos − k·sin, k·cos + q·sin)    # 二维平面旋转

# attention（double 累加点积；softmax 减最大值）
scores = Q·Kᵀ / √128
P     = softmax(scores)
out_q = bf16( Σ_key P[key] · V[key] )                         # 逐查询累加后再舍入
```

**输出写回**：

- 视频窗口互不相交 → 每个 token 只写一次，直接写回 `vid_output` 全局行；
- 文本 token 参与**每个**窗口 → 每 head 维护 `text_accumulator` 跨窗口累加，
  结束后 × `1/window_count` 平均。这与 PyTorch「文本 repeat 到各窗口 →
  window_reverse 汇聚取平均」语义等价，且避免了写冲突。

### 5.6 vid_only 特例（最后一个 block）

PyTorch 的 `MMModule(vid_only=True)` 原样返回 txt，但 block 外层仍对 txt 执行
attention/MLP 支路，数值上等价于 `txt_out = 2 × txt_attn`。C++ 显式保留这一
看似多余的行为以保证数值兼容——这是 cache 键复用带来的隐式语义，移植时极易漏掉。

---

## 6. SeedVR2DiTOutput —— 输出投影与 unpatchify

### 6.1 职责

三输入 `vid[L,2560]`、`emb[15360]`、`shape[T,H,W]`，两输出
`out[16, T, 2H, 2W]`（即 `Mat(2W, 2H, T, 16)`）与 `output_shape`。

### 6.2 计算公式

```text
n   = RMSNorm(x) · norm_w                                    # affine，eps=1e-5
y[c] = n[c] · (emb[c·6 + 1] + scale[c]) + emb[c·6] + shift[c] # Ada（见 6.3）
p   = Linear(2560 → 16·4) + b                                 # [L, 64]
out[c, t', 2y+dy, 2x+dx] = p[(t·H + y)·W + x, (dy·2+dx)·16 + c]   # (t h w c) 逆展开
```

### 6.3 关键细节：emb 的 cache 键碰撞

完整 PyTorch 图中该层的 Ada 复用了 **block-0 的 `emb_repeat_0_vid` cache 键**，
因此 embedding 必须按 `[2560, 2, 3]` 的**第 0 层（attn 槽）**解释
（`emb[c·6+0]`/`emb[c·6+1]`），而不是按输出层自己的 `[5120,1,3]` 形状解释。
这是移植时靠中间张量比对才发现的隐式行为。

### 6.4 动态逻辑

unpatchify 是 patchify 的精确逆操作（`(t h w c)` 顺序还原为空间邻域），
输出空间尺寸 ×2；`output_shape` 以 int32 blob 传回调用方。

---

## 7. Vulkan 化需求清单（按层）

| 层 | shader 工作量 | 要点 |
| --- | --- | --- |
| FramewiseGroupNorm | 小 | 逐 (帧,组) 归约 + affine；经典 reduction 模板 |
| SpaceTimeShuffle | 小 | 纯数据搬运 + 通道混合（1×1×1 卷积可交给原生 pipeline 或合并进 gather） |
| FramewiseSpatialAttention | 中 | 归约 + [N,N] softmax；N=H·W 动态 |
| DiTInput / DiTOutput | 中 | patch(unpatch) gather/scatter + 逐元素 + 原生 InnerProduct 委托 |
| DiTBlock | 大 | 嵌套 InnerProduct 的 forward_vkcompute 委托 → Ada/逐元素 shader → MM-RoPE → 动态窗口 attention（含文本跨窗口汇聚） |

统一约束：shader 与 CPU 实现共享同一套测试向量与窗口/位置公式；
数值验收从「逐字节一致」切换为相对误差容差（见 README 后续计划）。
