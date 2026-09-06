# SeedVR2DiTInput 的 Vulkan 实现

> 这是六个 SeedVR2 自定义层中**第四个**完成 Vulkan 化的算子，也是进入 DiT 侧的第一个。
> 它最大的特点是**大量复用 ncnn 原生 InnerProduct**（「自定义层编排 + 原生算子干重活」），
> 由此引出本层最有面试价值的两个坑：复用子层时的 **vkdev 传递**与 **elempack 手动管理**。

## 1. 这个算子实现了什么

`SeedVR2DiTInput` 是 3B DiT 的输入层，4 输入 4 输出：

| 输入 | 形状 | 输出 | 形状 |
|---|---|---|---|
| vid | `[T·H·W, 33]` | vid_out | `[T·(H/2)·(W/2), dim=2560]` |
| txt | `[L_txt, 5120]` | txt_out | `[L_txt, dim]` |
| timestep | 标量 | emb | `[15360]` |
| vid_shape | `[T,H,W]`（int） | patched_shape | `[T, H/2, W/2]`（int） |

核心逻辑（DiT 的 patchify + 时间编码）：

```
vid [T·H·W, 33]  --2x2 patchify-->  [T·(H/2)·(W/2), 132]  --video_proj(132→2560)-->  vid_out
txt [L_txt, 5120] --text_proj(5120→2560)-->  txt_out
timestep --sinusoidal(256 维)-->  time_mlp(256→2560→2560→15360, SiLU)-->  emb
```

其中 5 个矩阵乘投影（video/text/time_in/time_hidden/time_out）**全部复用 ncnn 原生
`InnerProduct`**（`create_layer("InnerProduct")` 返回 `Layer_final`，自动获得 CPU+Vulkan
双实现）。自定义代码只写三个原生算子表达不了的动态 shader：**patchify**（2×2 空间重排）、
**sinusoidal**（时间正弦/余弦编码）、**silu**（SiLU 激活）。

## 2. 用了什么方法

### 三个自定义 shader + 五个复用 InnerProduct

```cpp
// patchify shader 核心：2x2 patch 的 (dy,dx,c) 拼接，行步长用 w 而非 cstep
uint pidx = gi / p.patch_channels;
uint pc = gi % p.patch_channels;
uint block = pc / p.video_channels;   // [0,4)
uint dy = block / 2u, dx = block % 2u, c = pc % p.video_channels;
uint src = (t * p.height + y * 2u + dy) * p.width + x * 2u + dx;
buffer_st1(patches_blob, pidx * p.patch_channels + pc,
           buffer_ld1(video_blob, src * p.video_channels + c));
```

### 复用 InnerProduct 的 forward

```cpp
// Layer_final 的 forward(VkMat) 自动委托给 InnerProduct_vulkan
video_projection->forward(patches, video_output, cmd, opt);
text_projection->forward(text, text_output, cmd, opt);
```

## 3. 遇到了什么问题（及其解决）

### 3.1 【最关键】复用 InnerProduct 必须手动传递 vkdev

**现象**：`upload_model` 阶段报 `video upload_model failed -1`。

**根因**：`create_layer("InnerProduct")` 返回的 `Layer_final` 同时持有 `layer_cpu` 和
`layer_vulkan`。而 `Layer_final::load_param` 里有这样一段：

```cpp
if (layer_vulkan) {
    if (vkdev) { ... layer_vulkan->load_param(pd); ... return; }
    delete layer_vulkan;      // vkdev 为空 → 删除 layer_vulkan，fallback 到 CPU
    layer_vulkan = 0;
}
```

Net 只会给**顶层自定义层**设置 `vkdev`（`net.cpp:1423 layer->vkdev = d->vkdev`），
不会递归设置子层的。所以 `load_inner_product` 里创建子层后，若没手动传 vkdev，
子层的 `layer_vulkan` 在 `load_param` 时就被删掉了，之后只能走 CPU。

**解决**：`load_inner_product` 里，`load_param` **之前**设置 `layer->vkdev`：

```cpp
ncnn::Layer* load_inner_product(const ncnn::ModelBin& mb, int input_size,
                                int output_size, bool bias, const ncnn::VulkanDevice* vkdev)
{
    ncnn::Layer* layer = ncnn::create_layer("InnerProduct");
    if (!layer) return nullptr;
#if NCNN_VULKAN
    if (vkdev) layer->vkdev = vkdev;   // 必须在 load_param 之前！
#endif
    ... layer->load_param(pd); layer->load_model(mb);
}
```

### 3.2 【最关键】复用 InnerProduct 必须手动管理 elempack

**现象**：`out2`（时间 MLP 输出）误差巨大（CPU 0.009 vs GPU -8），而 `out1`（text 投影，
2D gemm 路径）完全一致。

**根因**：ncnn 正常流程里，Net 在层之间做 `convert_layout`，把 `support_vulkan_packing`
的层的输入 pack 成 `elempack=4`。但 DiTInput 的中间量（sinusoidal）是**内部 workspace**，
不走 Net 的 `convert_layout`。而 `InnerProduct_vulkan` 对 `num_input % 4 == 0` 会固定用
`in_elempack=4`：

```cpp
int in_elempack = num_input % 4 == 0 ? 4 : 1;   // num_input=16 → 4
```

我的 sinusoidal 是 `elempack=1`，直接送进去，`Flatten_vulkan` 对 dims=1 又原样返回，
导致 pipeline 按 pack4 读、数据按 pack1 存，读错。

**解决**：模拟 Net 的 `convert_layout`，在送入 InnerProduct 前 `convert_packing` 到 pack4，
并把 silu 改成 pack4 shader（因为 time_hidden 的输出是 pack4）：

```cpp
// sinusoidal pack1 → pack4，匹配 InnerProduct 的 in_elempack=4
ncnn::VkMat sinusoidal_pack4;
sinusoidal_pack4.create(sinusoidal_dim / 4, 4u * 4, 4, opt.workspace_vkallocator);
vkdev->convert_packing(sinusoidal, sinusoidal_pack4, 4, cmd, opt);

time_projection_in->forward(sinusoidal_pack4, time_hidden, cmd, opt);
```

silu 用 pack4 宏（`sfpvec4` + `buffer_ld4/st4`）：

```glsl
layout(binding = 0, std430) buffer bottom_top_data { sfpvec4 bottom_top_blob[]; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= p.total) return;
    afpvec4 v = buffer_ld4(bottom_top_blob, i);
    v = v / (afpvec4(1.0f) + exp(-v));
    buffer_st4(bottom_top_blob, i, v);
}
```

这印证了 ncnn 内置 `multiheadattention_vulkan` 的做法：它复用 InnerProduct 后，
也是手动 `vkdev->convert_packing` 处理 elempack。

### 3.3 2D VkMat 的 cstep 陷阱

**现象**：`out0`（video 投影）误差巨大（~1e16）。

**根因**：2D `Mat`/`VkMat` 的 buffer 是 **row-major**（行步长 = w），而 `cstep` 对 dims=2
是 `w*h`（总大小）。我最初在 patchify shader 里用 `cstep` 做跨行寻址（把它当成了 4D 的
通道步长），导致读错。4D Mat 的 `cstep` 才是「通道步长」（`align(w*h*d, 4)`）。

**解决**：patchify shader 里统一用 `video_channels`/`patch_channels`（= w）做行步长，
不用 `cstep`。

### 3.4 标量输入（shape/timestep）的同步读取

**现象**：`shape=(0,0,0) timestep=0`，patchify 拿到错误的形状。

**根因**：`record_upload` 的 `convert_packing` 是**异步记录**的，forward 里直接
`mapped_ptr` 读 bottom 读到的是尚未写入的旧数据。

**解决**：用 `record_download` + `submit_and_wait` + `reset` 同步下载标量到 CPU 再读
（数据仅 4 个标量，正确性优先阶段可接受）：

```cpp
ncnn::Mat shape_cpu, timestep_cpu;
cmd.record_download(shape, shape_cpu, opt);
cmd.record_download(timestep, timestep_cpu, opt);
cmd.submit_and_wait();
cmd.reset();
const int* shape_data = static_cast<const int*>(shape_cpu.data);
```

### 3.5 GLSL 关键字陷阱（patch / half）

`patch` 是 GLSL 的 tessellation 关键字（`patch in`/`patch out`），`half` 是 fp16 类型
保留字。shader 里 `uint patch = ...`、`uint half = ...` 都会编译失败。改名为 `pidx`、
`half_dim`。

## 4. 性能考量

**当前版本**是「正确性优先」：

- patchify/sinusoidal/silu 都是逐元素或纯搬运 shader，memory-bound；
- 5 个矩阵乘走 InnerProduct 的 Vulkan 实现（gemm 路径对 2D 输入，sum8/pack4 对 1D）；
- `submit_and_wait` 只为同步读 4 个标量，会打断一次流水线（后续可优化为把 shape/timestep
  走 push constant 或 uniform 传入，避免同步等待）。

**可优化点**：时间 MLP 的三个 InnerProduct 与两个 SiLU 可以融合成更少 dispatch；但这是
DiT 输入层，每帧只执行一次，不是性能热点，优化收益有限。

## 5. 面试问答

**Q1：这层为什么复用 InnerProduct 而不是全部手写？**
5 个投影是标准矩阵乘，ncnn 原生 InnerProduct 已有高效 CPU（SIMD）和 Vulkan 实现，
复用避免重复造轮子，且权重加载/量化/混合精度都交给成熟代码。自定义代码只写
patchify/sinusoidal/silu 这些原生算子表达不了的动态逻辑——这正是 DiT 侧「容器+引擎」
结构的第一站。

**Q2：复用 InnerProduct 时最容易踩什么坑？**
两个：① **vkdev 不传递**——Net 只给顶层层设 vkdev，子层的 `layer_vulkan` 会在
`load_param` 时被删（fallback CPU），必须在 `load_param` 前手动传；② **elempack 不匹配**
——Net 正常会在层间 convert_layout 做 pack4，但内部复用时跳过了这一步，必须手动
`convert_packing`，否则 InnerProduct 按 pack4 读、数据按 pack1 存。

**Q3：patchify 的 2x2 重排顺序怎么保证和 PyTorch einops 一致？**
patch 内通道按 `(dy, dx, c)` 顺序拼接（block = dy*2+dx），对应 einops 的 `(h,w,c)` 展开；
shader 里由 patch 索引反推 `(t,y,x)` 空间坐标，再由 `(dy,dx)` 反推源位置 `(t*H+y*2+dy)*W+x*2+dx`。

**Q4：为什么中间激活用 fp32 workspace？**
和前三个算子一致：正确性优先阶段用 fp32 保证与 CPU 基线逐元素对齐；数据 buffer 用
sfp（受 fp16 控制），统计/中间量强制 float。DiT 输入层的中间量（132 维 patches）很小，
fp32 开销可忽略。

## 6. 验证结果

`tests/dit_input_vulkan_runner.cpp` 用小尺寸（dim=32 等）对 4 组参数分别跑 CPU 与
Vulkan，比较 4 个输出。实测（WSL llvmpipe）：

```
[case 0] vc=33 T=2 H=4 W=4              : max|diff| = 0.000000 PASS
[case 1] vc=33 T=1 H=2 W=6 (非4对齐)     : max|diff| = 0.000000 PASS
[case 2] vc=5  T=2 H=4 W=4 (非4倍数cstep): max|diff| = 0.000000 PASS
[case 3] vc=33 tc=8 T=3 H=2 W=2          : max|diff| = 0.000000 PASS
======== 结果: 4 pass, 0 fail ========
```

用例覆盖：video_channels=33/5（非 4 倍数，cstep 与 pack 边界）、W=6 非 4 对齐、
多帧/单帧。全部逐元素一致（含 int 型 patched_shape）。
