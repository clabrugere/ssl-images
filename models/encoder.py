import torch.nn.functional as F
from torch import Tensor
from torch.nn import AdaptiveAvgPool2d, BatchNorm2d, Conv2d, Identity, MaxPool2d, Module, ReLU, Sequential
from torch.nn.init import constant_, kaiming_normal_


class ShortcutProjection(Module):
    def __init__(self, in_channels, out_channels, stride) -> None:
        super().__init__()
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn = BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.bn(self.conv1(x))


class ResidualBlock(Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self._layers = Sequential(
            Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            BatchNorm2d(out_channels),
            ReLU(inplace=True),
            Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
            BatchNorm2d(out_channels),
        )

        if stride != 1 or in_channels != out_channels:
            self.shortcut = ShortcutProjection(in_channels, out_channels, stride)
        else:
            self.shortcut = Identity()

    def forward(self, x: Tensor) -> Tensor:
        out = self._layers(x)

        return F.relu(self.shortcut(x) + out)


class ResNet(Module):
    def __init__(
        self,
        in_channels: int,
        num_stage: int,
        num_block_per_stage: int,
        out_channels_first: int = 64,
        kernel_size_first: int = 7,
    ) -> None:
        super().__init__()
        self.conv1 = Conv2d(
            in_channels, out_channels_first, kernel_size_first, stride=2, padding=kernel_size_first // 2, bias=False
        )
        self.bn = BatchNorm2d(out_channels_first)
        self.max_pool = MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.residual_blocks = Sequential()
        in_channels_stage = out_channels_first
        for i in range(num_stage):
            out_channels_stage = out_channels_first * 2**i
            for _ in range(num_block_per_stage):
                self.residual_blocks.append(ResidualBlock(in_channels_stage, out_channels_stage))
                in_channels_stage = out_channels_stage

        self.avgpool = AdaptiveAvgPool2d((1, 1))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        # input shape: (B, C, H, W)
        batch_size = x.size(0)

        x = self.bn(self.conv1(x))
        x = self.max_pool(x)
        x = self.residual_blocks(x)  # (B, C_o, H_o, W_o)
        x = self.avgpool(x)
        x = x.view(batch_size, -1)

        return x
