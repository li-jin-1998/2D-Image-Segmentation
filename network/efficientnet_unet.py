from collections import OrderedDict
from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor
from torchvision import ops
from torchvision.models import efficientnet

from convert_onnx import is_convert_onnx


class IntermediateLayerGetter(nn.ModuleDict):
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}

        # 重新构建backbone，将没有使用到的模块全部删掉
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


activation_layer = nn.ReLU(inplace=True)


# activation_layer = nn.LeakyReLU(0.1, inplace=True)


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__()
        middle_channels = in_channels // 2
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, middle_channels, 4, 2, 1, 0, bias=False),
            nn.BatchNorm2d(middle_channels),
            activation_layer,
            nn.Conv2d(middle_channels, num_classes, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return self.conv(x)


class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            activation_layer
        )

    def forward(self, x):
        return self.conv(x)


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DownConv, self).__init__()
        if mid_channels is None:
            mid_channels = in_channels * 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            activation_layer,
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            activation_layer
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            activation_layer
        )
        self.maxpool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        residual = self.conv2(x)
        x = self.conv(x) + residual
        x = self.maxpool(x)
        return x


class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpConv, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, padding=0, output_padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            activation_layer
        )

    def forward(self, x):
        return self.conv(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(DepthwiseSeparableConv, self).__init__()

        # 深度卷积层
        self.depthwise = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size,
                                                 stride=stride, padding=(kernel_size - 1) // 2, groups=in_channels,
                                                 bias=False),
                                       nn.BatchNorm2d(in_channels),
                                       activation_layer
                                       )
        # 逐点卷积层
        self.pointwise = nn.Sequential(nn.Conv2d(in_channels, out_channels, stride, bias=False),
                                       nn.BatchNorm2d(out_channels),
                                       activation_layer
                                       )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, p):
        super(DecoderBlock, self).__init__()

        middle_channels = int(in_channels * 2)

        self.up = UpConv(in_channels, middle_channels)

        self.conv1 = Conv(middle_channels, middle_channels, kernel_size=3)
        self.conv2 = Conv(middle_channels, out_channels, kernel_size=3)

        self.drop = ops.DropBlock2d(p=p, block_size=3, inplace=False)

    def forward(self, x, y):
        x = self.up(x)

        x = self.conv1(x)
        x = self.conv2(x)

        x = torch.cat([y, x], dim=1)
        x = self.drop(x)
        return x


class EfficientUNet(nn.Module):
    def __init__(self, num_classes, pretrain_backbone: bool = True, model_name: str = None, with_depth: bool = False):
        super(EfficientUNet, self).__init__()
        if model_name == 'efficientnet_b0':
            backbone = efficientnet.efficientnet_b0(pretrained=pretrain_backbone)
            self.stage_out_channels = [16, 24, 40, 112, 320]
        elif model_name == 'efficientnet_b1':
            backbone = efficientnet.efficientnet_b1(pretrained=pretrain_backbone)
            self.stage_out_channels = [16, 24, 40, 112, 320]
        elif model_name == 'efficientnet_b2':
            backbone = efficientnet.efficientnet_b2(pretrained=pretrain_backbone)
            self.stage_out_channels = [16, 24, 48, 120, 352]
        else:
            exit(1)
        stage_indices = [1, 2, 3, 5, 7]
        return_layers = dict([(str(j), f"stage{i}") for i, j in enumerate(stage_indices)])
        self.backbone = IntermediateLayerGetter(backbone.features, return_layers=return_layers)
        drop = [0.2, 0.2, 0.2, 0.2]
        # print(drop)

        self.up1 = DecoderBlock(self.stage_out_channels[4], self.stage_out_channels[3], drop[0])
        self.up2 = DecoderBlock(self.stage_out_channels[3] * 2, self.stage_out_channels[2], drop[1])
        self.up3 = DecoderBlock(self.stage_out_channels[2] * 2, self.stage_out_channels[1], drop[2])
        self.up4 = DecoderBlock(self.stage_out_channels[1] * 2, self.stage_out_channels[0], drop[3])
        self.outconv = OutConv(self.stage_out_channels[0] * 2, num_classes=num_classes)

        from network.PSAM import PSAModule
        self.PPM = PSAModule(self.stage_out_channels[4], self.stage_out_channels[4])
        # self.PPM = PPM(self.stage_out_channels[4], self.stage_out_channels[4] // 4, [2, 3, 5, 6])
        self.with_depth = with_depth
        if self.with_depth:
            self.conv = nn.Conv2d(1, self.stage_out_channels[0], kernel_size=7, stride=2, padding=3, bias=False)
            self.bn = nn.BatchNorm2d(self.stage_out_channels[0])
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
            self.encoder1 = DownConv(self.stage_out_channels[0], self.stage_out_channels[1])
            self.encoder2 = DownConv(self.stage_out_channels[1], self.stage_out_channels[2])
            self.encoder3 = DownConv(self.stage_out_channels[2], self.stage_out_channels[3])
            self.encoder4 = DownConv(self.stage_out_channels[3], self.stage_out_channels[4])

    def forward(self, x: torch.Tensor, depth) -> Dict[str, torch.Tensor]:
        if is_convert_onnx:
            x = x.permute(0, 3, 1, 2)  # rgb
        backbone_out = self.backbone(x)

        if self.with_depth:
            depth0 = self.conv(depth)
            depth0 = self.bn(depth0)
            depth0 = self.relu(depth0)
            depth0 = self.maxpool(depth0)
            depth1 = self.encoder1(depth0)
            depth2 = self.encoder2(depth1)
            depth3 = self.encoder3(depth2)
            depth4 = self.encoder4(depth3)
            # print(d0.shape)
            # for i in range(8):
            #     print(i,backbone_out['stage{}'.format(str(i))].shape)
            # encoder
            e0 = backbone_out['stage0'] + depth0
            e1 = backbone_out['stage1'] + depth1
            e2 = backbone_out['stage2'] + depth2
            e3 = backbone_out['stage3'] + depth3
            e4 = backbone_out['stage4']  # + depth4
        else:
            e0 = backbone_out['stage0']
            e1 = backbone_out['stage1']
            e2 = backbone_out['stage2']
            e3 = backbone_out['stage3']
            e4 = backbone_out['stage4']

        # center
        e4 = self.PPM(e4)

        # decoder
        d4 = self.up1(e4, e3)
        d3 = self.up2(d4, e2)
        d2 = self.up3(d3, e1)
        d1 = self.up4(d2, e0)
        out = self.outconv(d1)
        if is_convert_onnx:
            out = out.permute(0, 2, 3, 1)
        return {'out': out}


if __name__ == '__main__':
    from torchsummary import summary

    model = EfficientUNet(num_classes=3, pretrain_backbone=True,
                          model_name='efficientnet_b1', with_depth=True).to("cuda")
    summary(model, [(3, 192, 192), (1, 192, 192)])  # 16,903,683   16,604,147
