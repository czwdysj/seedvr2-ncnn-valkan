# SeedVR2 ncnn-Vulkan vs PyTorch 性能对照报告（RTX 5090 32G）

> 测试日期：2026-09-11　测试卡：NVIDIA GeForce RTX 5090（32GB，驱动 595.71）
> ncnn 版本：本项目 v0（fp16 权重 + fp32 计算，**VAE 无分块**，自定义 6 层含 Vulkan 后端）
> PyTorch 版本：官方框架 `VideoDiffusionInfer`（bf16 autocast + **conv 内存限制分块** + flash attention），torch 2.8.0+cu128
> 采样：均 1 step、cfg=1.0、seed=666；输入为随机张量（同规模，性能测试内容无关）

## 0. 测试范围与方法

- 显存峰值：`nvidia-smi` 0.2s 间隔采样取最大值；PyTorch 另记录 `torch.cuda.max_memory_allocated`
- 耗时：分阶段计时（preprocess / VAE encode / DiT / VAE decode / postprocess）
- 尺寸约束：VAE 8× 空间下采样 + DiT patch 2× → **分辨率须为 16 的倍数**（480=16×30、720=16×45 合法；360 实际被裁到 352）
- ncnn 的「resident」= 32 个 DiT block 权重常驻显存（`--resident`）；「流式」= 每个 block 用时再加载

## 1. 端到端总耗时对比（1 step）

| 规模 | ncnn 流式 | ncnn resident | PyTorch bf16 | 备注 |
|---|---|---|---|---|
| 352p × 5 帧 | 42.5 s | 22.1 s | — | ncnn 可跑通的最大帧长档 |
| 480p × 5 帧 | 60.1 s | **OOM** | ~8.1 s（推理）+18.2 s（加载） | resident 因显存冲突失败 |
| 720p × 5 帧 | **OOM** | **OOM** | ~9.5 s + 18.5 s | ncnn 在 VAE encode 即 OOM |

> PyTorch 的 ~18s「加载」含 13GB 检查点从磁盘读入+初始化，属一次性成本，不参与逐视频推理；下同。

## 2. 分阶段耗时（480p × 5 帧，单位秒）

| 阶段 | ncnn 流式 | ncnn resident | PyTorch bf16 | ncnn/PyTorch |
|---|---|---|---|---|
| VAE encode | 9.3 | 9.3 | 4.2 | 2.2× |
| DiT（1 step） | 28.9 | 4.8 | **0.33** | 14.5×（resident）|
| VAE decode | 22.0 | OOM | 3.3 | 6.7× |
| 合计（推理） | 60.1 | — | 7.9 | 7.6× |

## 3. 显存峰值（MiB）

| 规模 | ncnn 流式 | ncnn resident | PyTorch |
|---|---|---|---|
| 352p × 5 | 15 048 | 28 114 | — |
| 480p × 5 | 24 679 | 28 201（decode 时 OOM）| 29 742（含权重加载）|
| 720p × 5 | 25 693（encode OOM）| — | 30 356 |
| 480p × 9 | OOM | — | — |
| 352p × 13 | OOM | — | — |
| 240p × 61 | OOM | — | — |

关键观察：ncnn 的 OOM 峰值普遍在 25–30G，**不是总量不够，而是 VAE 单层 3D 卷积的一次性连续分配超限**（如 encoder 首层 128 通道 × 全分辨率 × 全帧 fp32 ≈ 2.4GB，多级叠加 + staging buffer 击穿 32G 上限）。

## 4. 瓶颈结论

按影响排序：

1. **VAE 无分块 → 显存爆炸（P0，阻断性）**
   ncnn VAE 把整段视频一次性全分辨率前向，激活显存随「帧数 × 分辨率」线性增长：720p 5 帧、240p 61 帧均 OOM。官方 PyTorch 靠 `vae.set_memory_limit` 做卷积分块（tile）才能跑 720p。**这是「用 ncnn 跑真实视频」的第一障碍，不解决则 720p 完全不可行。**

2. **DiT 单步慢 14.5×（P0）**
   480p 时 ncnn resident DiT 4.8s vs PyTorch 0.33s。来源：(a) 计算精度 fp32 vs bf16（理论 2×）；(b) 自定义 attention 未用 flash attention 类优化，ncnn 侧为逐 token 的 matmul + softmax 朴素实现；(c) 3D 卷积 pack1 直接卷积无 im2col 复用。剩余 ~7× 主要是算子实现差距。

3. **VAE decode/encode 慢 2–7×（P1）**
   decode 6.7×、encode 2.2×：fp32 计算 + 3D conv pack1 无优化。VAE decode 在 ncnn 总耗时中占比最高（360p 时 55%）。

4. **resident 模式的显存调度缺陷（P1）**
   resident 把 6.4GB（fp16）DiT 权重常驻显存，与 VAE decode 激活竞争：480p 流式能跑、resident OOM。需在 VAE 阶段动态 offload DiT 权重（官方 PyTorch 正是 `dit.to("cpu")` ↔ `vae.to("cuda")` 交替）。

5. **流式加载开销（P2）**
   64×64 时 resident 提速 66×，352p 时仅 5.2×——规模越大 DiT 计算占比越高、权重重载占比越低，流式模式的「每步重载 6.4GB」开销相对缩小，但仍建议显存充足时用 resident + 分块 VAE 组合。

## 5. 优化建议（按优先级）

1. **【最高】VAE 时间/空间分块（tile VAE）**
   engine 层实现：encode 按时间块（每块 5/9/13 帧，重叠 1 块上下文）分别 encode、latent 沿时间拼接；decode 按 latent 块分别 decode、帧拼接。已有一个 WIP 版本证明流程可跑通（352p 61f 从 OOM 变为可跑），但简单分块会因时间因果卷积产生 latent 数量错位，需**块间重叠 + 丢弃重叠 latent** 的正确数学，并用小规模全帧 encode 做数值校准。预计可把 720p 可行性解出来。
2. **【高】VAE/DiT 转 bf16 计算**
   ncnn Vulkan 后端已支持 fp16 存储，但计算仍 fp32。将 attention matmul、3D conv 改 fp16 计算可收 2× 速度 + 显存减半（对 32G 卡意义重大）。
3. **【高】DiT attention 算子重写**
   对齐 flash attention 思路（分块 softmax + 避免物化 QK^T 全矩阵），这是 14.5× 差距的主要来源。
4. **【中】resident 显存调度**
   VAE encode/decode 阶段把 DiT 权重 offload 回内存，decode 后重新 upload；或 VAE 分块 + 权重复用共享显存池。
5. **【中】3D 卷积 pack 优化**
   当前 pack1 逐元素卷积，可做 pack4/im2col 复用，直接压缩 VAE 2–7× 的耗时。

## 6. 附：本报告发现并修复的 bug

- **`popen(cmd, "rb")` 在 POSIX 下 EINVAL**：glibc 的 popen mode 只接受 "r"/"w"。Linux 管道不做文本换行转换，因此改用标准的 `"r"/"w"` 模式。此前 CLI 的 mp4 解码在 Linux 上从未真正跑通（冒烟只测了 usage 路径）。
- **镜像里的 `seedvr2_ema_3b.pth`（13GB）损坏**：aria2 多线程下载被中断留下空洞文件，`torch.load` 报 central directory 错误。已用 `huggingface_hub` 重新下载并 zip 校验通过。

## 7. 数据落盘位置

- 服务器（weste.seetacloud.com:21805，/root/autodl-tmp）：权重、编译产物、测试片段、`bench_*.txt`、PyTorch 环境（ptenv）
- 本仓库：本次测试相关的源代码修复（popen、pytorch_bench init_process_group）已随 git 提交
