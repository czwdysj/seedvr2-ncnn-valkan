# pnnx model stat
# model inputshape = [1,3,32,32]f32
# FLOPS = 1.825M
# memory OPS = 132.424K

import os
import numpy as np
import tempfile, zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torchvision
    import torchaudio
except:
    pass

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.convbn2d_0 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=3, kernel_size=(3,3), out_channels=8, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.stem_2 = nn.ReLU()
        self.stem_3 = nn.MaxPool2d(ceil_mode=False, dilation=(1,1), kernel_size=(2,2), padding=(0,0), return_indices=False, stride=(2,2))
        self.convbn2d_1 = nn.Conv2d(bias=True, dilation=(1,1), groups=8, in_channels=8, kernel_size=(3,3), out_channels=8, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.depthwise_2 = nn.SiLU()
        self.pointwise_0 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=8, kernel_size=(1,1), out_channels=16, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.pointwise_1 = nn.LeakyReLU(negative_slope=0.1)
        self.convbn2d_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=8, kernel_size=(1,1), out_channels=16, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.down = nn.AvgPool2d(ceil_mode=False, count_include_pad=True, divisor_override=None, kernel_size=(2,2), padding=(0,0), stride=(2,2))
        self.mid_conv = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=16, kernel_size=(3,3), out_channels=24, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.mid_act = nn.PReLU(num_parameters=24)
        self.upsample = nn.Upsample(mode='nearest', scale_factor=(2.0,2.0), size=None)
        self.stride_conv = nn.Conv2d(bias=False, dilation=(1,1), groups=1, in_channels=24, kernel_size=(3,3), out_channels=24, padding=(1,1), padding_mode='zeros', stride=(2,2))
        self.gelu = nn.GELU()
        self.head_pool = nn.AdaptiveAvgPool2d(output_size=(4,4))
        self.fc1 = nn.Linear(bias=True, in_features=384, out_features=32)
        self.hardswish = nn.Hardswish()
        self.fc2 = nn.Linear(bias=True, in_features=32, out_features=10)

        archive = zipfile.ZipFile('/home/czw1/ncnn_learn/seedvr2_ncnn/learn_pnnx/out/toy_pnnx_net.pnnx.bin', 'r')
        self.convbn2d_0.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_0.bias', (8), 'float32')
        self.convbn2d_0.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_0.weight', (8,3,3,3), 'float32')
        self.convbn2d_1.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_1.bias', (8), 'float32')
        self.convbn2d_1.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_1.weight', (8,1,3,3), 'float32')
        self.pointwise_0.bias = self.load_pnnx_bin_as_parameter(archive, 'pointwise.0.bias', (16), 'float32')
        self.pointwise_0.weight = self.load_pnnx_bin_as_parameter(archive, 'pointwise.0.weight', (16,8,1,1), 'float32')
        self.convbn2d_2.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_2.bias', (16), 'float32')
        self.convbn2d_2.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_2.weight', (16,8,1,1), 'float32')
        self.mid_conv.bias = self.load_pnnx_bin_as_parameter(archive, 'mid_conv.bias', (24), 'float32')
        self.mid_conv.weight = self.load_pnnx_bin_as_parameter(archive, 'mid_conv.weight', (24,16,3,3), 'float32')
        self.mid_act.weight = self.load_pnnx_bin_as_parameter(archive, 'mid_act.weight', (24), 'float32')
        self.stride_conv.weight = self.load_pnnx_bin_as_parameter(archive, 'stride_conv.weight', (24,24,3,3), 'float32')
        self.fc1.bias = self.load_pnnx_bin_as_parameter(archive, 'fc1.bias', (32), 'float32')
        self.fc1.weight = self.load_pnnx_bin_as_parameter(archive, 'fc1.weight', (32,384), 'float32')
        self.fc2.bias = self.load_pnnx_bin_as_parameter(archive, 'fc2.bias', (10), 'float32')
        self.fc2.weight = self.load_pnnx_bin_as_parameter(archive, 'fc2.weight', (10,32), 'float32')
        archive.close()

    def load_pnnx_bin_as_parameter(self, archive, key, shape, dtype, requires_grad=True):
        return nn.Parameter(self.load_pnnx_bin_as_tensor(archive, key, shape, dtype), requires_grad)

    def load_pnnx_bin_as_tensor(self, archive, key, shape, dtype):
        fd, tmppath = tempfile.mkstemp()
        with os.fdopen(fd, 'wb') as tmpf, archive.open(key) as keyfile:
            tmpf.write(keyfile.read())
        m = np.memmap(tmppath, dtype=dtype, mode='r', shape=shape).copy()
        os.remove(tmppath)
        return torch.from_numpy(m)

    def forward(self, v_0):
        v_1 = self.convbn2d_0(v_0)
        v_2 = self.stem_2(v_1)
        v_3 = self.stem_3(v_2)
        v_4 = self.convbn2d_1(v_3)
        v_5 = self.depthwise_2(v_4)
        v_6 = self.pointwise_0(v_5)
        v_7 = self.pointwise_1(v_6)
        v_8 = self.convbn2d_2(v_3)
        v_9 = (v_7 + v_8)
        v_10 = self.down(v_9)
        v_11 = self.mid_conv(v_10)
        v_12 = self.mid_act(v_11)
        v_13 = self.upsample(v_12)
        v_14 = self.stride_conv(v_13)
        v_15 = self.gelu(v_14)
        v_16 = self.head_pool(v_15)
        v_17 = torch.flatten(v_16, end_dim=-1, start_dim=1)
        v_18 = self.fc1(v_17)
        v_19 = self.hardswish(v_18)
        v_20 = self.fc2(v_19)
        v_21 = F.softmax(v_20, dim=1)
        return v_21

def export_torchscript():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, 32, 32, dtype=torch.float)

    mod = torch.jit.trace(net, v_0)
    mod.save("/home/czw1/ncnn_learn/seedvr2_ncnn/learn_pnnx/out/toy_pnnx_net_pnnx.py.pt")

def export_onnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, 32, 32, dtype=torch.float)

    torch.onnx.export(net, v_0, "/home/czw1/ncnn_learn/seedvr2_ncnn/learn_pnnx/out/toy_pnnx_net_pnnx.py.onnx", export_params=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=13, input_names=['in0'], output_names=['out0'])

def export_pnnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, 32, 32, dtype=torch.float)

    import pnnx
    pnnx.export(net, "/home/czw1/ncnn_learn/seedvr2_ncnn/learn_pnnx/out/toy_pnnx_net_pnnx.py.pt", v_0)

def export_ncnn():
    export_pnnx()

@torch.no_grad()
def test_inference():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, 32, 32, dtype=torch.float)

    return net(v_0)

if __name__ == "__main__":
    print(test_inference())
