import torch
import torch.nn as nn


class ToyPnnxNet(nn.Module):
    """A compact CNN with varied ops for reading pnnx/ncnn param files.

    Input shape: NCHW [1, 3, 32, 32]
    Output shape: [1, 10]
    """

    def __init__(self) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.depthwise = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, groups=8, bias=True),
            nn.BatchNorm2d(8),
            nn.SiLU(inplace=False),
        )

        self.pointwise = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=1, stride=1, padding=0, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
        )

        self.skip = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(16),
        )

        self.down = nn.AvgPool2d(kernel_size=2, stride=2)
        self.mid_conv = nn.Conv2d(16, 24, kernel_size=3, stride=1, padding=1, bias=True)
        self.mid_act = nn.PReLU(num_parameters=24)
        self.upsample = nn.Upsample(scale_factor=2.0, mode="nearest")
        self.stride_conv = nn.Conv2d(24, 24, kernel_size=3, stride=2, padding=1, bias=False)
        self.gelu = nn.GELU()
        self.head_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(24 * 4 * 4, 32)
        self.hardswish = nn.Hardswish()
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        main = self.depthwise(x)
        main = self.pointwise(main)
        residual = self.skip(x)
        x = main + residual

        x = self.down(x)
        x = self.mid_conv(x)
        x = self.mid_act(x)
        x = self.upsample(x)
        x = self.stride_conv(x)
        x = self.gelu(x)
        x = self.head_pool(x)

        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.hardswish(x)
        x = self.fc2(x)
        return torch.softmax(x, dim=1)


def build_model() -> ToyPnnxNet:
    torch.manual_seed(20260716)
    model = ToyPnnxNet()
    model.eval()
    return model


if __name__ == "__main__":
    net = build_model()
    example = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        y = net(example)
    print(net)
    print("output shape:", tuple(y.shape))
    print("output sum:", float(y.sum()))
