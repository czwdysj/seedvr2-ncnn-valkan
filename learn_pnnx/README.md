# learn_pnnx

这个目录是一个 pnnx 学习用的小例子，用来对照 PyTorch 代码、pnnx 中间表示和 ncnn `.param/.bin` 文件。

## 文件

- `toy_pnnx_net.py`: PyTorch 模型定义，输入是 `[1, 3, 32, 32]`，输出是 10 类 softmax。
- `export_and_convert.py`: 导出 TorchScript，并调用 pnnx 生成 pnnx/ncnn 文件。
- `inspect_param.py`: 把 `.param` 文件按层打印成表格，方便阅读。
- `compare_ncnn.py`: 用同一个输入比较 PyTorch 输出和 ncnn 输出。
- `out/`: 转换后的输出目录。

## 运行

当前机器可以直接复用这个虚拟环境：

```bash
cd /home/czw1/ncnn_learn/seedvr2_ncnn/learn_pnnx
/home/czw1/ncnn_learn/Penguin-VL-ncnn/.venv/bin/python export_and_convert.py
```

如果你已经把 `pnnx` 放进 PATH，也可以用自己的 Python 环境运行：

```bash
cd /home/czw1/ncnn_learn/seedvr2_ncnn/learn_pnnx
python export_and_convert.py
```

## 主要输出

- `out/toy_pnnx_net.pt`: `torch.jit.trace` 导出的 TorchScript 模型。
- `out/toy_pnnx_net.pnnx.param`: pnnx 的中间图结构。
- `out/toy_pnnx_net.pnnx.bin`: pnnx 中间权重。
- `out/toy_pnnx_net_pnnx.py`: pnnx 反生成的 PyTorch 风格代码。
- `out/toy_pnnx_net.ncnn.param`: ncnn 网络结构，重点阅读这个。
- `out/toy_pnnx_net.ncnn.bin`: ncnn 权重。
- `out/toy_pnnx_net_ncnn.py`: pnnx 生成的 Python/ncnn 调用示例。

## 阅读 param

```bash
cd /home/czw1/ncnn_learn/seedvr2_ncnn/learn_pnnx
/home/czw1/ncnn_learn/Penguin-VL-ncnn/.venv/bin/python inspect_param.py out/toy_pnnx_net.ncnn.param
```

验证 PyTorch 和 ncnn 输出：

```bash
cd /home/czw1/ncnn_learn/seedvr2_ncnn/learn_pnnx
/home/czw1/ncnn_learn/Penguin-VL-ncnn/.venv/bin/python compare_ncnn.py
```

ncnn `.param` 每一层大致是：

```text
op_type layer_name input_count output_count input_blob... output_blob... key=value...
```

例如 `Convolution` 后面的参数会描述输出通道、卷积核、stride、padding、bias、权重数量等。`BinaryOp` 通常对应 PyTorch 里的 `main + residual`，`Interp` 对应 `nn.Upsample`，`InnerProduct` 对应 `nn.Linear`。

## PyTorch 到 ncnn 的大致对照

| PyTorch 代码 | 常见 ncnn op |
| --- | --- |
| `nn.Conv2d` | `Convolution` 或 `ConvolutionDepthWise` |
| `nn.BatchNorm2d` | 通常会被 fuse 到卷积，或显示为 `BatchNorm` |
| `nn.ReLU` | `ReLU` |
| `nn.MaxPool2d` / `nn.AvgPool2d` | `Pooling` |
| `nn.SiLU` | `Swish` |
| `nn.LeakyReLU` | `ReLU` with negative slope |
| `main + residual` | `BinaryOp` |
| `nn.PReLU` | `PReLU` |
| `nn.Upsample` | `Interp` |
| `nn.GELU` | `GELU` |
| `torch.flatten` | `Flatten` 或 `Reshape` |
| `nn.Linear` | `InnerProduct` |
| `torch.softmax` | `Softmax` |

注意：pnnx/ncnn 会做图优化和层融合，所以 `.param` 不一定逐行一一对应 PyTorch 代码。例如 `Conv2d + BatchNorm2d` 很可能被合并成一个卷积层。
