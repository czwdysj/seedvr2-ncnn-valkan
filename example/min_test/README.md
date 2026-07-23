# 16×16 最小端到端测试

这个目录用于快速确认当前 `SeedVR2Engine` 的完整 CPU 调度能够执行，不用于评价恢复质量。
输入是 NCNN 官方 256×256 像素图标用 nearest-neighbor 缩放得到的 16×16 单帧图像。

## 文件

- `ncnn_16x16.png`：16×16 官方 NCNN 图标，来源为
  `https://raw.githubusercontent.com/Tencent/ncnn/master/images/256-ncnn.png`。
- `input_1x16x16_thwc.f32`：对应的 `[1,16,16,3]` FP32 THWC 输入，范围 `[0,1]`。
- `text_zero_1x5120.f32`：一个全零 `[1,5120]` embedding，只用于最小调度测试。
- `run_min_test.sh`：构建并执行当前正式 `seedvr2_cli`。
- `raw_output_to_ppm.py`：检查输出形状/finite，并生成可查看图片。

## 运行

```bash
cd /home/czw1/ncnn_learn/seedvr2_ncnn
chmod +x example/min_test/run_min_test.sh
SEEDVR2_THREADS=8 ./example/min_test/run_min_test.sh
```

脚本使用 `ncnn_models/vae_dynamic` 和 `ncnn_models/dit_full_fp16`。运行成功后生成：

- `example/min_test/output.f32`：Engine 的原始 FP32 THWC 输出。
- `example/min_test/output.ppm`：无额外依赖即可生成的 RGB 图片。
- `example/min_test/output.png`：安装了 FFmpeg 时自动生成。

当前 WSL2 CPU 实测输出为 `T,H,W,C=1,16,16,3`，全部数值 finite、范围 `[0,1]`，
8 线程约需 22 秒，峰值 RSS 约 1.85 GiB。耗时只用于确认运行量级，不是性能 benchmark。

16×16 会在 VAE 中下降到非常小的 latent，画质没有参考意义。正式视频仍应使用此前的
320×240、480×256 等参考尺寸，并使用真实的 positive/negative 文本 embedding。
