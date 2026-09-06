# 服务器开发与代码同步规范

本文定义 SeedVR2-NCNN 项目在云服务器（RTX 5090）上的开发流程、git 同步
机制与提交规范。**所有开发会话必须遵守**。

## 1. 服务器概况

| 项目 | 值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5090 32GB（Blackwell，驱动 610.43.02 Open Kernel Module） |
| 系统 | Ubuntu 24.04.1（容器，hostname 94a42aa0a719），384 核 CPU / 48GB 内存 |
| 登录 | `ssh root+vm-f7TXgbTt9FQ1wvOK@39.145.28.59 -p 32222`（网关仅支持密码认证，公钥不可用） |
| 项目目录 | `/root/ncnn_learn/seedvr2_ncnn`（工作副本） |
| ncnn 源码 | `/root/ncnn_learn/ncnn`（自带 vendored glslang，与 WSL 本地同版本） |

### 1.1 Vulkan 运行时现状（重要）

- 容器创建时**未授予 `graphics` capability**（NVIDIA_DRIVER_CAPABILITIES 只含
  compute/utility），NVIDIA Vulkan ICD 无法初始化（`vk_icdGetInstanceProcAddr`
  返回 NULL），容器内无法修复；
- 已将 `libnvidia-gl-610` + `libnvidia-compute-610` 手工解包到 `/opt/nvidia-gl`
  （绕过 dpkg 与容器挂载文件的冲突），并写入 `/etc/profile.d/vulkan-nvidia.sh`；
  **一旦服务商开通 graphics capability，无需再改任何东西即可启用 5090 Vulkan**；
- 当前可用的 Vulkan 设备是 **llvmpipe**（Mesa 软件实现，Vulkan 1.3 一致性认证）。
  shader 的**数值正确性验证**用它完全够用（慢但正确），性能测试必须等 5090 ICD；
- 需要向服务商申请：重建/重配容器时设置
  `NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility`。

## 2. Git 三点同步架构

```text
WSL 本地（主力开发）                    云服务器（Vulkan 开发/验证）
/home/czw1/ncnn_learn/seedvr2_ncnn      /root/ncnn_learn/seedvr2_ncnn
        │    push/pull                       pull/push    │
        └────────────→ 裸仓库（同步中枢）←───────────────┘
                    /root/repos/seedvr2_ncnn.git
        （GitHub origin 仍保留，定期 push 备份）
```

- 本地 remote 名：`server`；裸仓库只做同步中枢，不直接在其上开发；
- **每次开发会话结束前必须 commit + push**，另一侧开始工作前必须 pull；
- 提交信息一律使用**简体中文**，格式：

```text
<类型>: <一句话概述>

- 改动点 1（为什么改）
- 改动点 2
- 验证方式与结果
```

类型取值：`feat`（新功能）、`fix`（修 bug）、`docs`（文档）、`refactor`（重构）、
`build`（构建/环境）、`test`（测试）、`perf`（性能）。允许组合（如 `docs+refactor`）。

## 3. 常用命令速查

### 3.1 WSL 本地侧

```bash
# 环境变量（每次会话或写入 ~/.bashrc）
export SSH_ASKPASS="$HOME/.ssh/seedvr_askpass.sh"
export SSH_ASKPASS_REQUIRE=force DISPLAY=:0
export GIT_SSH_COMMAND="ssh -p 32222 -o StrictHostKeyChecking=no"

git pull server main        # 拉取服务器侧的提交
git push server main        # 推送本地提交
ssh root+vm-f7TXgbTt9FQ1wvOK@39.145.28.59 -p 32222   # 登录服务器
```

### 3.2 服务器侧

```bash
cd /root/ncnn_learn/seedvr2_ncnn
git pull /root/repos/seedvr2_ncnn.git main
git push /root/repos/seedvr2_ncnn.git main   # 服务器侧产生的提交
```

## 4. Vulkan 开发循环

1. 写代码 / 写 shader（`.comp`）；
2. 在服务器上以 llvmpipe 做数值正确性验证（对比 CPU 基线，容差标准）；
3. 每完成一个可验证的单元：commit（中文详细说明）+ push；
4. 5090 的 graphics capability 开通后，补性能测试与 GPU 数值回归；
5. 定期 `git push origin main` 备份到 GitHub。

## 5. 环境重建备忘

容器可能被服务商重置。重建后按以下顺序恢复：

1. 挂载/重装 GPU 驱动（compute+utility 通常自动）；
2. `apt-get install -y --no-install-recommends libvulkan-dev vulkan-tools build-essential libomp-dev`
   （cmake 已预装于 /usr/local/bin）；
3. 解包 NVIDIA 图形库（若 graphics capability 已开通则容器自动挂载，可跳过）：

   ```bash
   mkdir -p /opt/nv /opt/nvidia-gl && cd /opt/nv
   apt-get download libnvidia-gl-610 libnvidia-compute-610
   dpkg-deb -x libnvidia-gl-610_*.deb /opt/nvidia-gl
   dpkg-deb -x libnvidia-compute-610_*.deb /opt/nvidia-gl
   ```

4. 确认 `/etc/profile.d/vulkan-nvidia.sh` 存在（内容见该文件）；
5. `git clone /root/repos/seedvr2_ncnn.git /root/ncnn_learn/seedvr2_ncnn`；
6. 重建 ncnn：`cmake -S . -B build -DNCNN_VULKAN=ON -DNCNN_BUILD_TOOLS=OFF ...`。

## 6. 已知问题记录

- 2026-09-06：网关 SSH 仅支持密码认证，公钥被拒绝；密码认证需
  `SSH_ASKPASS` 包装（WSL 侧已固化为 `~/.ssh/seedvr_askpass.sh` + 环境变量）；
- 2026-09-06：直接 dpkg 安装 libnvidia-gl-610 会与容器挂载的驱动文件冲突
  导致 dpkg 半安装状态，需 `dpkg --purge --force-depends` 清理后改用
  `dpkg-deb -x` 手工解包方案；
- 2026-09-06：直接层调用 ncnn 自定义层时，Option 必须显式
  `use_packing_layout=false`，否则 Reshape 输出 packed blob（ep=8），
  与 `RuntimeContext` 的标量布局契约一致（见 test_demo 实验）。
