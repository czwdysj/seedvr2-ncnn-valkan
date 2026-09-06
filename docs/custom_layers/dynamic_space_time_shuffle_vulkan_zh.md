# DynamicSpaceTimeShuffle 的 Vulkan 实现

> 这是六个 SeedVR2 自定义层中**第一个**完成 Vulkan 化的算子。本文档按面试口径记录：
> 实现了什么、用了什么方法、遇到什么问题、如何解决、性能如何考量，并附真实代码。

## 1. 这个算子实现了什么

`DynamicSpaceTimeShuffle` 是 VAE Decoder 的上采样层，等价于 PyTorch 里
`learned 1×1×1 Conv3D` 接 **MAGViT 风格 channel-to-space-time 重排**：

- 输入：`Mat(w=W, h=H, d=T, c=C)`，即逻辑形状 `[C, T, H, W]` 的视频 latent；
- 输出：`[C, T_out, 2H, 2W]`，其中 `T_out = r_t·T − 1`（r_t=2 时）；
- 空间 2 倍上采样 + 时间 r_t 倍上采样，**核心是通道索引到时空坐标的映射**，
  外加一个极易漏掉的「删除重复头帧」边界规则。

三个实例：`512→4096(r_t=2)` ×2、`256→1024(r_t=1)` ×1。

## 2. 用了什么方法

**两条 forward 路径共享同一份 load_param/load_model**，保证动态语义一致：

- CPU 路径（`forward(Mat)`）：已与 PyTorch reference 逐字节对齐，作为数值基线；
- Vulkan 路径（`forward(VkMat)`）：把索引映射翻译成一个 GLSL compute shader，
  每个 work item 计算一个输出元素。

关键工程决策：

1. **shader 以内嵌字符串携带，运行时用 ncnn 的 glslang 编译**（`compile_spirv_module`），
   不改 ncnn 的静态 shader 注册表——自定义层的 shader 跟随代码一起进 git，版本可控；
2. **权重在 `upload_model` 阶段一次性上传 GPU**（`VkTransfer::record_upload`），
   forward 期间只记录一次 dispatch，不做任何主机侧搬运；
3. **通道数/倍率用 specialization constant** 烤进 SPIR-V，换取内积循环的编译期展开；
   **运行时形状（T/H/W/cstep）走 push constant**，同一份 pipeline 可处理任意尺寸。

### 核心：shader 里的索引映射（与 CPU 逐行等价）

```glsl
// buffer 用 ncnn 注入的 sfp 类型声明，读写必须走 buffer_ld1/st1 宏
layout(binding = 0) buffer bottom_data { sfp bottom_blob[]; };
layout(binding = 1) buffer top_data    { sfp top_blob[]; };
layout(binding = 2) buffer weight_data { sfp weight_blob[]; };
layout(binding = 3) buffer bias_data   { sfp bias_blob[]; };

// gi 是稠密输出索引，按 ncnn 4D 布局解码：x 最内 -> y -> t(d) -> c 最外
uint x_out = gi % p.w_out;
uint y_out = (gi / p.w_out) % p.h_out;
uint t_out = (gi / (p.w_out * p.h_out)) % p.t_out;
uint c     = gi / (p.w_out * p.h_out * p.t_out);

uint ox = x_out & 1u, x_in = x_out >> 1;   // 空间子像素反推
uint oy = y_out & 1u, y_in = y_out >> 1;

uint t_in, ot;
if (r_t == 2u) {                            // 首帧删除：raw_t=1 被跳过
    uint raw_t = (t_out == 0u) ? 0u : (t_out + 1u);
    t_in = raw_t >> 1; ot = raw_t & 1u;
} else { t_in = t_out; ot = 0u; }

uint pc = ((oy * 2u + ox) * r_t + ot) * c_out + c;   // 投影通道解码
afp sum = buffer_ld1(bias_blob, pc);                 // afp = 算术精度累加器
uint base = t_in * (p.h * p.w_in) + y_in * p.w_in + x_in;
for (uint ci = 0u; ci < c_in; ci++)
    sum += buffer_ld1(weight_blob, pc * c_in + ci)
         * buffer_ld1(bottom_blob, ci * p.cstep_in + base);
// 写回：c 在最外层，用 cstep_out 对齐，避免 padding 错位
uint dst = c * p.cstep_out + t_out * (p.h_out * p.w_out) + y_out * p.w_out + x_out;
buffer_st1(top_blob, dst, sum);
```

## 3. 遇到了什么问题（及其解决）

### 3.1 自定义层如何接入 ncnn 的 Vulkan 执行流？

ncnn 的 Vulkan 层惯例是「一个 CPU 类 + 一个 `*_vulkan` 子类」，但**自定义层通过
`register_custom_layer` 注册的是单一 creator**，Net 在 `load_param` 里依次尝试
`create_layer_vulkan → create_layer_cpu → create_custom_layer`，最后 `layer->vkdev`
被赋值。**解决**：不拆类，直接在原类上重载 `create_pipeline/upload_model/forward(VkMat)`，
并在构造函数里设 `support_vulkan = true`。这比照搬 ncnn 的双类模式更贴合自定义层场景。

### 3.2 shader 从哪来、怎么编译？

ncnn 的 shader 走编译期静态注册表（`layer_shader_registry`，由 CMake 脚本生成），
自定义层塞不进去。**解决**：发现 ncnn 公开了 `compile_spirv_module(const char*, opt, spirv)`
运行时编译入口 + `Pipeline::create(spv_data, size, spec)` 内存加载入口，两者组合即可
把 GLSL 源码字符串编译成 pipeline，完全绕开静态注册表。

### 3.3 cstep 对齐（最容易错、最隐蔽）

ncnn 的 `Mat`/`VkMat` 通道步长 `cstep` 按 16 字节对齐，`W=5` 时 `w*h*d=30` 但
`cstep=32`。shader 若想当然用 `w*h*d` 计算通道偏移就会取错地址。**解决**：把
`cstep` 作为 push constant 传入，shader 里跨通道寻址一律用 `ci * p.cstep`。

### 3.4 首帧删除的时间反向映射

CPU 侧是「正着生成 raw_t 再跳过」，GPU 侧每个线程只拿到输出索引 `t_out`，需要
**反推** `(t_in, ot)`：`raw_t = (t_out == 0) ? 0 : t_out + 1`。这个反推关系是
「5 帧 latent 解码出 4n+1 帧」的根源，也是和 PyTorch reference 对齐的关键。

### 3.5 【最隐蔽】fp16_packed 默认开启，裸 `float` 读 buffer 全是垃圾

**现象**：shader 编译成功、dispatch 正常、ReLU 对照（内置层）精确通过，但我的
shader 读 `bottom_blob[0]` 得到 `-0.000003`（应为 `-0.5`），读 bias 得到 0。

**根因**：ncnn 的 `Option` 里 `use_fp16_packed` **默认是 true**（`option.cpp` 第 37 行），
我只在测试里关了 `use_fp16_storage` 和 `use_fp16_arithmetic`，漏了这个。看
`compile_spirv_module` 的决策链（`gpu.cpp`）：

```cpp
if (use_bf16_storage) ... 
else if (use_bf16_packed) ...
else if (use_fp16_storage) ...   // false，跳过
else if (use_fp16_packed) ...    // true！→ sfp 被定义为 uint
```

于是 ncnn 注入 `sfp = uint`、`buffer_ld1(buf,i)` 展开成
`unpackHalf2x16(buf[i/2])[i%2]`——**GPU buffer 里是两个 half 打包进一个 uint32**，
而我用裸 `float[]` + `buf[i]` 读，就把 half 的位模式错当 float，得到纯垃圾。
ReLU 为什么对？因为它用 `sfpvec4` + `buffer_ld4` 宏，自动适配了 packed 布局。

**解决（双保险）**：
1. **测试侧**：显式 `opt.use_fp16_packed = false`，让 buffer 以 fp32 存储，与 CPU
   基线做严格逐位比对；
2. **shader 侧（正解）**：一律用 `sfp` 声明 buffer、`buffer_ld1/st1` 读写、`afp`
   累加——这样无论 fp32/fp16_storage/fp16_packed 哪种精度，宏都会自动展开成
   正确的 unpack/pack，shader 不依赖具体 storage 约定。

### 3.6 【索引错误】把通道 c 当成最内层维度解码

**现象**：修复 fp16 后误差从 0.42 降到 0.3655，`[0]` 元素碰巧正确、其余全错。

**根因**：我最初把稠密索引 `gi` 解码成 `c = gi % c_out`，把通道当成了最内层
维度。但 ncnn 4D `Mat`/`VkMat` 的内存布局是 **x 最内 → y → t(d) → c 最外**，
且 c 维用 `cstep`（16 字节对齐）做步长而非 `w*h*d`。`[0]` 正确只是 `gi=0` 时
任何解码都落在 `(0,0,0,0)` 的巧合。

**解决**：按真实布局解码 `x/y/t/c`，写回时用 `dst = c*cstep_out + t*(h_out*w_out)
+ y*w_out + x_out`，把 `cstep_out`（而非 `w*h*d`）作为通道步长传入 push constant。
这正是 ncnn 内置层（如 `batchnorm.comp`）用 `gi = gz*psc(cstep) + gy*psc(w) + gx`
的同一套约定。

## 4. 性能考量（当前实现 vs 后续优化）

**当前版本**是「正确性优先」的标量实现：

- 每线程一个输出元素，内积 `c_in` 次 FMA（c_in = 256 或 512）；
- 权重访问 `weight[pc*c_in+ci]` 在同一线程内连续，L1 友好；
- 输入访问 `bottom[ci*cstep + base]` 跨通道大步长，但同一 work-group 内相邻线程
  （x_out 相邻）读相邻空间位置，空间维 coalesce 尚可。

**可优化点**（记录在案，等真 GPU 再调）：

1. **shared memory 分块**：内积的 `c_in` 维和空间维都适合分块，把输入片 + 权重片
   载入 shared memory 复用，减少全局内存往返（当前每个输出元素重读一遍权重）；
2. **向量化**：用 `vec4` 一次算 4 个输出元素（c 维 4 对齐），并启用 packing
   （`support_vulkan_packing`），带宽和指令数都更优；
3. **把 1×1×1 卷积与重排拆成两段**：卷积交给 ncnn 原生 InnerProduct/Convolution
   的 Vulkan 实现，shuffle 只做纯索引搬运——但会引入一次中间 tensor 的显存往返，
   需要在 5090 上实测两种方案再做取舍。

**为什么要先写这个最简单层**：它同时覆盖了「shader 编译 → pipeline 注册 →
VkMat 输入输出 → 权重上传 → 容差比对」的完整工程链路，后续五个层（含最难的
DiTBlock）都复用这套骨架，只替换 shader 与绑定。

## 5. 面试问答（附真实代码）

**Q1：为什么这层必须自定义，不能用 ncnn 原生 PixelShuffle？**
ncnn 的 `PixelShuffle` 只做空间二维重排；这里还要把投影通道的一部分展开到时间轴
（`C·s²·r_t → C` 三轴重排），且输出 `T_out = 2T−1` 是动态表达式、删除的是 raw frame 1
而非尾帧——三者都超出原生算子的表达能力。

**Q2：GPU 路径怎么保证和 CPU/PyTorch 结果一致？**
两条路径共用 `load_param/load_model`，索引映射在 shader 里逐行复刻 CPU 实现；测试
runner 对同一输入分别跑 CPU 与 Vulkan，断言 `max|diff| < 1e-4`。

**Q3：权重和输入分别怎么进 GPU？**
权重走 `upload_model`（VkTransfer 一次性上传、forward 零搬运）；输入由 Net 的
Extractor 在 `record_upload` 阶段上传，VkMat 在层间流转不落回 CPU。

## 6. 验证结果

`tests/shuffle_vulkan_runner.cpp` 对多组参数分别跑 CPU 与 Vulkan，断言
`max|diff| < 1e-4`。实测（WSL llvmpipe，Vulkan 1.3 软件实现）：

```
[RELU-probe] identity max|diff| = 0.000000 OK       ← 对照链路保真
[case 0] C=4 T=3 r_t=2 (pack4 + 首帧删除)      : max|diff| = 0.000000 PASS
[case 1] C=5 T=2 r_t=2 (标量   + 首帧删除)      : max|diff| = 0.000000 PASS
[case 2] C=5 T=4 r_t=1 (非 2 倍率，无首帧删除)  : max|diff| = 0.000000 PASS
[case 3] C=4 T=2 r_t=1 (pack4 + 非 2 倍率)      : max|diff| = 0.000000 PASS
======== 结果: 4 pass, 0 fail ========
```

四组用例刻意覆盖了关键边界：`C=4`（触发 `record_upload` 的 pack4）、`C=5`
（标量路径）、`r_t=2`（首帧删除）、`r_t=1`（非 2 倍率）、`W=5`（cstep 16 字节
对齐后的地址计算）。全部逐元素一致，误差为 0。
