import ml_collections
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import efficientnet

from network.UDTransNet.DAT import DAT
from network.efficientnet_unet import IntermediateLayerGetter, OutConv

# 改进UDTransNet,换backbone
def get_model_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.transformer.num_heads = 4
    config.transformer.num_layers = 4
    config.expand_ratio = 2
    config.transformer.embedding_channels = 32 * config.transformer.num_heads
    config.KV_size = config.transformer.embedding_channels * 4
    config.KV_size_S = config.transformer.embedding_channels
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0.1
    config.patch_sizes = [16, 8, 4, 2]
    config.base_channel = 32
    config.decoder_channels = [16, 24, 40, 112, 320]
    return config


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Reconstruct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Reconstruct, self).__init__()
        if kernel_size == 3:
            padding = 1
        else:
            padding = 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        B, n_patch, hidden = x.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        if self.scale_factor[0] > 1:
            x = nn.Upsample(scale_factor=self.scale_factor)(x)
        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out


class Down_block(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(Down_block, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.Maxpool(x)
        x = self.conv(x)
        return x


class DRA_C(nn.Module):
    """ Channel-wise DRA Module"""

    def __init__(self, skip_dim, decoder_dim, img_size, config):
        super().__init__()
        self.patch_size = img_size // 14
        self.ft_size = img_size
        self.patch_embeddings = nn.Conv2d(in_channels=decoder_dim,
                                          out_channels=decoder_dim,
                                          kernel_size=self.patch_size,
                                          stride=self.patch_size)
        self.conv = nn.Sequential(
            nn.Conv2d(decoder_dim, skip_dim, kernel_size=(1, 1), bias=True),
            nn.BatchNorm2d(skip_dim),
            nn.ReLU(inplace=True))
        self.query = nn.Linear(decoder_dim, skip_dim, bias=False)
        self.key = nn.Linear(config.transformer.embedding_channels, skip_dim, bias=False)
        self.value = nn.Linear(config.transformer.embedding_channels, skip_dim, bias=False)
        self.out = nn.Linear(skip_dim, skip_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.psi = nn.InstanceNorm2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.reconstruct = Reconstruct(skip_dim, skip_dim, kernel_size=1,
                                       scale_factor=(self.patch_size, self.patch_size))

    def forward(self, decoder, trans):
        decoder_mask = self.conv(decoder)
        decoder_L = self.patch_embeddings(decoder).flatten(2).transpose(-1, -2)
        query = self.query(decoder_L).transpose(-1, -2)
        key = self.key(trans)
        value = self.value(trans).transpose(-1, -2)
        ch_similarity_matrix = torch.matmul(query, key)
        ch_similarity_matrix = self.softmax(self.psi(ch_similarity_matrix.unsqueeze(1)).squeeze(1))
        out = torch.matmul(ch_similarity_matrix, value).transpose(-1, -2)
        out = self.out(out)
        out = self.reconstruct(out)
        out = out * decoder_mask
        return out


class DRA_S(nn.Module):
    """ Spatial-wise DRA Module"""

    def __init__(self, skip_dim, decoder_dim, img_size, config):
        super().__init__()
        self.patch_size = img_size // 14
        self.ft_size = img_size
        self.patch_embeddings = nn.Conv2d(in_channels=decoder_dim,
                                          out_channels=decoder_dim,
                                          kernel_size=self.patch_size,
                                          stride=self.patch_size)
        self.conv = nn.Sequential(
            nn.Conv2d(decoder_dim, skip_dim, kernel_size=(1, 1), bias=True),
            nn.BatchNorm2d(skip_dim),
            nn.ReLU(inplace=True))
        self.query = nn.Linear(decoder_dim, skip_dim, bias=False)
        self.key = nn.Linear(config.transformer.embedding_channels, skip_dim, bias=False)
        self.value = nn.Linear(config.transformer.embedding_channels, skip_dim, bias=False)
        self.out = nn.Linear(skip_dim, skip_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.psi = nn.InstanceNorm2d(1)
        self.reconstruct = Reconstruct(skip_dim, skip_dim, kernel_size=1,
                                       scale_factor=(self.patch_size, self.patch_size))

    def forward(self, decoder, trans):
        decoder_mask = self.conv(decoder)
        decoder_L = self.patch_embeddings(decoder).flatten(2).transpose(-1, -2)
        query = self.query(decoder_L)
        key = self.key(trans).transpose(-1, -2)
        value = self.value(trans)
        sp_similarity_matrix = torch.matmul(query, key)
        sp_similarity_matrix = self.softmax(self.psi(sp_similarity_matrix.unsqueeze(0)).squeeze(0))
        out = torch.matmul(sp_similarity_matrix, value)
        out = self.out(out)
        out = self.reconstruct(out)
        out = out * decoder_mask
        return out


class Up_Block(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, img_size, config):
        super().__init__()
        self.scale_factor = (img_size // 14, img_size // 14)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(in_ch // 2),
            nn.ReLU(inplace=True))
        self.pam = DRA_C(skip_ch, in_ch // 2, img_size, config)  # # channel_wise_DRA
        # self.pam = DRA_S(skip_ch, in_ch//2, img_size, config) # spatial_wise_DRA
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch // 2 + skip_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, decoder, o_i):
        d_i = self.up(decoder)
        o_hat_i = self.pam(d_i, o_i)
        x = torch.cat((o_hat_i, d_i), dim=1)
        x = self.conv(x)
        return x


class ETransUNet(nn.Module):

    def __init__(self, config=get_model_config(), n_channels=3, n_classes=1, img_size=224):
        super().__init__()
        self.n_classes = n_classes

        backbone = efficientnet.efficientnet_b1(pretrained=True)
        self.stage_out_channels = [16, 24, 40, 112, 320]
        stage_indices = [1, 2, 3, 5, 7]
        return_layers = dict([(str(j), f"stage{i}") for i, j in enumerate(stage_indices)])
        self.backbone = IntermediateLayerGetter(backbone.features, return_layers=return_layers)

        filters_resnet = [16, 24, 40, 112, 320]
        filters_decoder = [16, 24, 40, 112, 320]

        # =====================================================
        # DAT Module
        # =====================================================
        self.mtc = DAT(config, img_size, channel_num=filters_resnet[0:4], patchSize=config.patch_sizes)

        # =====================================================
        # DRA & Decoder
        # =====================================================
        self.Up5 = Up_Block(filters_resnet[4], filters_resnet[3], filters_decoder[3], 28, config)
        self.Up4 = Up_Block(filters_decoder[3], filters_resnet[2], filters_decoder[2], 56, config)
        self.Up3 = Up_Block(filters_decoder[2], filters_resnet[1], filters_decoder[1], 112, config)
        self.Up2 = Up_Block(filters_decoder[1], filters_resnet[0], filters_decoder[0], 224, config)

        self.outconv = OutConv(filters_decoder[0], num_classes=n_classes)

    def forward(self, x):
        backbone_out = self.backbone(x)
        # encoder
        e1 = backbone_out['stage0']
        e2 = backbone_out['stage1']
        e3 = backbone_out['stage2']
        e4 = backbone_out['stage3']
        e5 = backbone_out['stage4']

        o1, o2, o3, o4 = self.mtc(e1, e2, e3, e4)

        d4 = self.Up5(e5, o4)
        d3 = self.Up4(d4, o3)
        d2 = self.Up3(d3, o2)
        d1 = self.Up2(d2, o1)
        out = self.outconv(d1)
        return out


if __name__ == '__main__':
    from torchsummary import summary

    vit_config = get_model_config()
    model = ETransUNet(vit_config, n_channels=3, n_classes=3, img_size=224).to("cuda")
    summary(model, (3, 224, 224))
