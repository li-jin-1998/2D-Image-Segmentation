from collections import OrderedDict
from functools import partial
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchsummary import summary
from torchvision import models, ops
from torchvision.models import efficientnet
from network.ACBlock import ACBlock

from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation as SElayer

nonlinearity = partial(F.relu, inplace=True)


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


# activation_layer = nn.GELU()
activation_layer = nn.LeakyReLU(0.1, inplace=True)


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__()
        middle_channels = in_channels // 2
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, middle_channels, 4, 2, 1, 0),
            nn.BatchNorm2d(middle_channels),
            activation_layer,
            nn.Conv2d(middle_channels, num_classes, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, input):
        return self.conv(input)


class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2),
            nn.BatchNorm2d(out_ch),
            activation_layer
        )

    def forward(self, input):
        return self.conv(input)


class Conv2(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(Conv2, self).__init__()
        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(out_ch),
        #     activation_layer
        # )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch // 2, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False),
            nn.BatchNorm2d(out_ch // 2),
            activation_layer)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch // 2, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=False),
            nn.BatchNorm2d(out_ch // 2),
            activation_layer
        )

    def forward(self, input):
        # x = self.conv(input)
        x1 = self.conv1(input)
        x2 = self.conv2(input)
        return torch.cat([x1, x2], dim=1)


class DeformConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DeformConv, self).__init__()
        from network.deform_conv import DeformConv2d
        self.conv = nn.Sequential(
            DeformConv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            activation_layer
        )

    def forward(self, x):
        x = self.up(x)
        return x


class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpConv, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_ch),
            activation_layer
        )

    def forward(self, x):
        return self.conv(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DepthwiseSeparableConv, self).__init__()

        # 深度卷积层
        self.depthwise = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size,
                                                 stride=1, padding=(kernel_size - 1) // 2, groups=in_channels),
                                       nn.BatchNorm2d(in_channels),
                                       activation_layer
                                       # nn.LeakyReLU(0.1, inplace=True)
                                       )
        # 逐点卷积层
        self.pointwise = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1),
                                       nn.BatchNorm2d(out_channels),
                                       activation_layer
                                       # nn.LeakyReLU(0.1, inplace=True)
                                       )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# from network.FasterNet import Partial_conv3

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, p):
        super(DecoderBlock, self).__init__()

        middle_channels = int(in_channels * 2)

        # self.conv1 = Partial_conv3(in_channels, 2)
        self.conv1 = Conv(in_channels, middle_channels, kernel_size=3)
        self.deconv2 = UpConv(middle_channels, middle_channels)
        self.conv3 = Conv(middle_channels, out_channels, kernel_size=3)

        self.drop = ops.DropBlock2d(p=p, block_size=3, inplace=True)

    def forward(self, x, y):
        x = self.conv1(x)
        x = self.deconv2(x)
        x = self.conv3(x)

        x = torch.cat([y, x], dim=1)
        x = self.drop(x)
        return x


class EfficientUNet(nn.Module):
    def __init__(self, num_classes, pretrain_backbone: bool = True, model_name: str = None):
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
        elif model_name == 'efficientnet_b3':
            backbone = efficientnet.efficientnet_b3(pretrained=pretrain_backbone)
            self.stage_out_channels = [24, 32, 48, 136, 384]
        elif model_name == 'efficientnet_b4':
            backbone = efficientnet.efficientnet_b4(pretrained=pretrain_backbone)
            self.stage_out_channels = [24, 32, 56, 160, 448]
        elif model_name == 'efficientnet_b5':
            backbone = efficientnet.efficientnet_b5(pretrained=pretrain_backbone)
            self.stage_out_channels = [24, 40, 64, 176, 512]
        elif model_name == 'efficientnet_b6':
            backbone = efficientnet.efficientnet_b6(pretrained=pretrain_backbone)
            self.stage_out_channels = [32, 40, 72, 200, 576]
        elif model_name == 'efficientnet_b7':
            backbone = efficientnet.efficientnet_b7(pretrained=pretrain_backbone)
            self.stage_out_channels = [32, 48, 80, 224, 640]
        else:
            exit(1)
        stage_indices = [1, 2, 3, 5, 7]
        # stage_indices = [i for i in range(8)]
        return_layers = dict([(str(j), f"stage{i}") for i, j in enumerate(stage_indices)])
        self.backbone = IntermediateLayerGetter(backbone.features, return_layers=return_layers)

        drop_dict = {0: [0.1, 0.15, 0.15, 0.2],  # dice:96.67 miou:95.19 b1
                     1: [0.1, 0.15, 0.2, 0.3],  # dice:96.81 miou:95.15
                     2: [0.1, 0.2, 0.3, 0.1],  # dice:96.89 miou:95.20
                     3: [0.2, 0.2, 0.2, 0.2],  # dice:96.79 miou:95.24
                     4: [0.25, 0.25, 0.25, 0.25],  # dice:96.56 miou:95.19
                     5: [0.1, 0.1, 0.2, 0.2],  # dice:96.68 miou:95.12
                     6: [0.15, 0.15, 0.2, 0.2],  # dice:96.64 miou:95.11
                     7: [0.25, 0.25, 0.2, 0.2],  # dice:96.67 miou:95.21
                     8: [0.3, 0.3, 0.5, 0.5]  # dice:96.72 miou:95.18
                     }
        drop = drop_dict[3]
        print(drop)
        self.up1 = DecoderBlock(self.stage_out_channels[4], self.stage_out_channels[3], drop[0])
        self.up2 = DecoderBlock(self.stage_out_channels[3] * 2, self.stage_out_channels[2], drop[1])
        self.up3 = DecoderBlock(self.stage_out_channels[2] * 2, self.stage_out_channels[1], drop[2])
        self.up4 = DecoderBlock(self.stage_out_channels[1] * 2, self.stage_out_channels[0], drop[3])
        self.outconv = OutConv(self.stage_out_channels[0] * 2, num_classes=num_classes)

        from network.PSAM import PPM, PSAModule, PSAModule_initial
        self.PPM = PSAModule(self.stage_out_channels[4], self.stage_out_channels[4])
        # self.PPM = PPM(self.stage_out_channels[4], self.stage_out_channels[4] // 4, [2, 3, 5, 6])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x = x.permute(0, 3, 1, 2)  # rgb
        backbone_out = self.backbone(x)
        # for i in range(8):
        #     print(i,backbone_out['stage{}'.format(str(i))].shape)
        # encoder
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
        # out = out.permute(0, 2, 3, 1)
        return {'out': out}


if __name__ == '__main__':
    model = EfficientUNet(num_classes=3, pretrain_backbone=True,
                          model_name='efficientnet_b1').to("cuda")
    summary(model, (3, 192, 192))
