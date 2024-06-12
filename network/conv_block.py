import torch.nn as nn

activation_layer = nn.ReLU(inplace=True)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(DepthwiseSeparableConv, self).__init__()

        # 深度卷积层
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size,
                      stride=stride, padding=(kernel_size - 1) // 2,
                      groups=in_channels,
                      bias=False),
            nn.BatchNorm2d(in_channels),
            activation_layer
        )
        # 逐点卷积层
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, stride, bias=False),
            nn.BatchNorm2d(out_channels),
            activation_layer
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DeformConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DeformConv, self).__init__()
        from network.deform_conv import DeformConv2d
        self.conv = nn.Sequential(
            DeformConv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
