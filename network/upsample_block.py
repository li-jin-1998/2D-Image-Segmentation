import torch
import torch.nn as nn


class UpsampleModule(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, mode='transposed_conv'):
        super(UpsampleModule, self).__init__()
        self.mode = mode
        if mode == 'transposed_conv':
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=scale_factor),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(out_channels))
        elif mode == 'bilinear':
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(out_channels)
            )
        else:
            raise ValueError("Invalid mode: choose 'transposed_conv' or 'bilinear'")

    def forward(self, x):
        return self.upsample(x)


# Test the UpsampleModule
if __name__ == "__main__":
    x = torch.randn(1, 64, 32, 32)  # Example input tensor with shape (batch_size, channels, height, width)

    # Test transposed convolution upsampling
    upsample_transposed_conv = UpsampleModule(in_channels=64, out_channels=64, scale_factor=2, mode='transposed_conv')
    output_transposed_conv = upsample_transposed_conv(x)
    print(f"Output shape using transposed convolution: {output_transposed_conv.shape}")

    # Test bilinear upsampling
    upsample_bilinear = UpsampleModule(in_channels=64, out_channels=64, scale_factor=2, mode='bilinear')
    output_bilinear = upsample_bilinear(x)
    print(f"Output shape using bilinear upsampling: {output_bilinear.shape}")
