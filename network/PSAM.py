from functools import partial

import torch
import torch.nn as nn
import math
import torch.nn.functional as F

nonlinearity = partial(F.relu, inplace=True)


class channel_attention(nn.Module):
    # 初始化, in_channel代表输入特征图的通道数, ratio代表第一个全连接的通道下降倍数
    def __init__(self, in_channel, ratio=4):
        # 继承父类初始化方法
        super(channel_attention, self).__init__()

        # 全局最大池化 [b,c,h,w]==>[b,c,1,1]
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=1)
        # 全局平均池化 [b,c,h,w]==>[b,c,1,1]
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

        # 第一个全连接层, 通道数下降4倍
        self.fc1 = nn.Linear(in_features=in_channel, out_features=in_channel // ratio, bias=False)
        # 第二个全连接层, 恢复通道数
        self.fc2 = nn.Linear(in_features=in_channel // ratio, out_features=in_channel, bias=False)

        # relu激活函数
        self.relu = nn.ReLU()
        # sigmoid激活函数
        self.sigmoid = nn.Sigmoid()

    # 前向传播
    def forward(self, inputs):
        # 获取输入特征图的shape
        b, c, h, w = inputs.shape

        # 输入图像做全局最大池化 [b,c,h,w]==>[b,c,1,1]
        max_pool = self.max_pool(inputs)
        # 输入图像的全局平均池化 [b,c,h,w]==>[b,c,1,1]
        avg_pool = self.avg_pool(inputs)

        # 调整池化结果的维度 [b,c,1,1]==>[b,c]
        max_pool = max_pool.view([b, c])
        avg_pool = avg_pool.view([b, c])

        # 第一个全连接层下降通道数 [b,c]==>[b,c//4]
        x_maxpool = self.fc1(max_pool)
        x_avgpool = self.fc1(avg_pool)

        # 激活函数
        x_maxpool = self.relu(x_maxpool)
        x_avgpool = self.relu(x_avgpool)

        # 第二个全连接层恢复通道数 [b,c//4]==>[b,c]
        x_maxpool = self.fc2(x_maxpool)
        x_avgpool = self.fc2(x_avgpool)

        # 将这两种池化结果相加 [b,c]==>[b,c]
        x = x_maxpool + x_avgpool
        # sigmoid函数权值归一化
        x = self.sigmoid(x)
        # 调整维度 [b,c]==>[b,c,1,1]
        x = x.view([b, c, 1, 1])
        # 输入特征图和通道权重相乘 [b,c,h,w]
        # outputs = inputs * x
        outputs = x
        return outputs


class SEWeightModule(nn.Module):

    def __init__(self, channels, reduction=16):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)

        return weight


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class PSAModule(nn.Module):

    def __init__(self, inplans, planes, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16]):
        # groups 保证相同的计算量大小  不同的卷积核  空洞卷积应该可以省略这一步
        super(PSAModule, self).__init__()
        conv_groups = [1, 1, 1, 1]
        self.conv_1 = conv(inplans, planes // 4, kernel_size=3, padding=3 // 2, dilation=1,
                           stride=stride, groups=conv_groups[0])
        self.conv_2 = conv(inplans, planes // 4, kernel_size=3, padding=5 // 2, dilation=2,
                           stride=stride, groups=conv_groups[1])
        self.conv_3 = conv(inplans, planes // 4, kernel_size=3, padding=7 // 2, dilation=3,
                           stride=stride, groups=conv_groups[2])
        self.conv_4 = conv(inplans, planes // 4, kernel_size=3, padding=9 // 2, dilation=4,
                           stride=stride, groups=conv_groups[3])
        # self.conv_1 = conv(inplans, planes // 4, kernel_size=conv_kernels[0], padding=conv_kernels[0] // 2,
        #                    stride=stride, groups=conv_groups[0])
        # self.conv_2 = conv(inplans, planes // 4, kernel_size=conv_kernels[1], padding=conv_kernels[1] // 2,
        #                    stride=stride, groups=conv_groups[1])
        # self.conv_3 = conv(inplans, planes // 4, kernel_size=conv_kernels[2], padding=conv_kernels[2] // 2,
        #                    stride=stride, groups=conv_groups[2])
        # self.conv_4 = conv(inplans, planes // 4, kernel_size=conv_kernels[3], padding=conv_kernels[3] // 2,
        #                    stride=stride, groups=conv_groups[3])
        # self.se = SEWeightModule(planes // 4)
        self.se = channel_attention(planes // 4)
        self.split_channel = planes // 4
        self.softmax = nn.Softmax(dim=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch_size = x.shape[0]
        # print(x.shape)
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)
        # print(x1.shape,x2.shape,x3.shape,x4.shape)
        feats = torch.cat((x1, x2, x3, x4), dim=1)
        # feats=self.relu(feats)
        # return feats
        feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])

        x1_se = self.se(x1)
        x2_se = self.se(x2)
        x3_se = self.se(x3)
        x4_se = self.se(x4)
        # print(x1_se.shape, x2_se.shape, x3_se.shape, x4_se.shape)
        x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)
        attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors
        for i in range(4):
            x_se_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), 1)

        return out


class PSAModule_initial(nn.Module):

    def __init__(self, inplans, planes, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16]):
        # groups 保证相同的计算量大小  不同的卷积核  空洞卷积应该可以省略这一步
        super(PSAModule_initial, self).__init__()
        # conv_groups = [1, 1, 1, 1]
        # self.conv_1 = conv(inplans, planes // 4, kernel_size=3, padding=3 // 2, dilation=1,
        #                    stride=stride, groups=conv_groups[0])
        # self.conv_2 = conv(inplans, planes // 4, kernel_size=3, padding=5 // 2, dilation=2,
        #                    stride=stride, groups=conv_groups[1])
        # self.conv_3 = conv(inplans, planes // 4, kernel_size=3, padding=7 // 2, dilation=3,
        #                    stride=stride, groups=conv_groups[2])
        # self.conv_4 = conv(inplans, planes // 4, kernel_size=3, padding=9 // 2, dilation=4,
        #                    stride=stride, groups=conv_groups[3])
        self.conv_1 = conv(inplans, planes // 4, kernel_size=conv_kernels[0], padding=conv_kernels[0] // 2,
                           stride=stride, groups=conv_groups[0])
        self.conv_2 = conv(inplans, planes // 4, kernel_size=conv_kernels[1], padding=conv_kernels[1] // 2,
                           stride=stride, groups=conv_groups[1])
        self.conv_3 = conv(inplans, planes // 4, kernel_size=conv_kernels[2], padding=conv_kernels[2] // 2,
                           stride=stride, groups=conv_groups[2])
        self.conv_4 = conv(inplans, planes // 4, kernel_size=conv_kernels[3], padding=conv_kernels[3] // 2,
                           stride=stride, groups=conv_groups[3])
        self.se = SEWeightModule(planes // 4)
        # self.se = channel_attention(planes // 4)
        self.split_channel = planes // 4
        self.softmax = nn.Softmax(dim=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch_size = x.shape[0]
        # print(x.shape)
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)
        # print(x1.shape,x2.shape,x3.shape,x4.shape)
        feats = torch.cat((x1, x2, x3, x4), dim=1)
        # feats=self.relu(feats)
        # return feats
        feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])

        x1_se = self.se(x1)
        x2_se = self.se(x2)
        x3_se = self.se(x3)
        x4_se = self.se(x4)
        # print(x1_se.shape, x2_se.shape, x3_se.shape, x4_se.shape)
        x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)
        attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors
        for i in range(4):
            x_se_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), 1)

        return out


# class SPPblock(nn.Module):
#     def __init__(self, in_channels):
#         super(SPPblock, self).__init__()
#         self.pool1 = nn.MaxPool2d(kernel_size=[2, 2])
#         self.pool2 = nn.MaxPool2d(kernel_size=[3, 3])
#         self.pool3 = nn.MaxPool2d(kernel_size=[5, 5])
#         self.pool4 = nn.MaxPool2d(kernel_size=[6, 6])
#         self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1, padding=0)
#
#         self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=1, padding=0)
#
#     # self.se = channel_attention(in_channels // 4)
#     # self.split_channel = in_channels // 4
#     # self.softmax = nn.Softmax(dim=1)
#     def forward(self, x):
#         self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
#         # batch_size = x.shape[0]
#         self.layer1 = F.upsample(self.conv(self.pool1(x)), size=(h, w), mode='bilinear')
#         self.layer2 = F.upsample(self.conv(self.pool2(x)), size=(h, w), mode='bilinear')
#         self.layer3 = F.upsample(self.conv(self.pool3(x)), size=(h, w), mode='bilinear')
#         self.layer4 = F.upsample(self.conv(self.pool4(x)), size=(h, w), mode='bilinear')
#         self.layer5 = self.conv2(x)
#         out = torch.cat([self.layer5, self.layer1, self.layer2, self.layer3, self.layer4], 1)
#         # print(x.shape,out.shape)
#         return out


class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                # nn.MaxPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1),
                nn.BatchNorm2d(reduction_dim),
                nn.LeakyReLU(0.1, inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

        # self.conv = nn.Conv2d(in_channels=in_dim * 2, out_channels=in_dim, kernel_size=3, stride=1, padding=1)
        # self.relu = nn.LeakyReLU(0.1, inplace=True)
        # self.BN = nn.BatchNorm2d(in_dim)

    def forward(self, x):
        x_size = x.size()
        # x1 = self.conv(x)
        # x1 = self.BN(x1)
        # x1 = self.relu(x1)
        out = []
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        # print(torch.cat(out, 1).shape)
        outputs = torch.cat(out, 1)
        # outputs = self.conv(outputs)
        # outputs = self.BN(outputs)
        # outputs = self.relu(outputs)
        return outputs


class DACblock(nn.Module):
    def __init__(self, channel):
        super(DACblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


class SPPblock(nn.Module):
    def __init__(self, in_channels):
        super(SPPblock, self).__init__()
        # self.pool1 = nn.MaxPool2d(kernel_size=[3, 3], stride=1,padding=1)
        # self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=1,padding=1)
        # self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=1,padding=2)
        # self.pool4 = nn.MaxPool2d(kernel_size=[5, 5], stride=1,padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2])
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3])
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5])
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6])
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)

        self.layer1 = F.upsample(self.conv(self.pool1(x)), size=(h, w), mode='bilinear')
        self.layer2 = F.upsample(self.conv(self.pool2(x)), size=(h, w), mode='bilinear')
        self.layer3 = F.upsample(self.conv(self.pool3(x)), size=(h, w), mode='bilinear')
        self.layer4 = F.upsample(self.conv(self.pool4(x)), size=(h, w), mode='bilinear')

        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)

        return out
# model = PPM(512,512//8,(2,3,5,6)).to('cuda')
# # model = SPPblock(512).to('cuda')
# from torchsummary import summary
# summary(model, (512,16,16))
