import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    kernel_size = (3, 3)

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        kernel_height, kernel_width = self.kernel_size
        height_padding, width_padding = kernel_height // 2, kernel_width // 2

        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.ReflectionPad2d(padding=(width_padding, width_padding, height_padding, height_padding)),
            # 反射法padding填充
            nn.Conv2d(in_channels, mid_channels, kernel_size=(kernel_height, kernel_width)),
            nn.InstanceNorm2d(mid_channels),
            # 实例归一化
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=(width_padding, width_padding, height_padding, height_padding)),
            nn.Conv2d(mid_channels, out_channels, kernel_size=(kernel_height, kernel_width)),
            nn.InstanceNorm2d(out_channels)
        )
        if in_channels != out_channels:
            self.conv1x1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        else:
            self.conv1x1 = lambda x: x
        self.relu = nn.ReLU(inplace=True)

        for m in self.double_conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in")
                # 对卷积层进行凯明初始化

    def forward(self, x):
        return self.relu(self.conv1x1(x) + self.double_conv(x))


class Down(nn.Module):
    """
    Downscaling with maxpool then double conv
    使用maxpooling和double conv进行两倍下采样
    """

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling then double conv
    使用double conv进行两倍上采样
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        # 如果使用双线性，则使用正常卷积来减少通道的数量，否则则使用反卷积
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        nn.init.kaiming_uniform_(self.conv.weight, mode="fan_in")

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self,
                 in_channels,  # 输入通道
                 out_channels,  # 输出通道
                 conv_kernel_size,  # 卷积核size
                 ngf,  # 生成器的卷积核个数
                 nlayers,  # 层数
                 bilinear  # 双线性
                 ):
        super(UNet, self).__init__()
        self.n_channels = in_channels
        self.n_classes = out_channels
        self.bilinear = bilinear
        DoubleConv.kernel_size = conv_kernel_size

        self.inc = DoubleConv(in_channels, ngf)
        self.downs = nn.ModuleList(modules=[])
        factor = 2 if bilinear else 1
        for i in range(nlayers):
            if i != (nlayers - 1):
                self.downs.append(Down(ngf * 2 ** i, ngf * 2 ** (i + 1)))
            else:
                self.downs.append(Down(ngf * 2 ** i, ngf * 2 ** (i + 1) // factor))
        self.ups = nn.ModuleList(modules=[])
        for i in np.arange(nlayers - 1, -1, -1):
            if i != 0:
                self.ups.append(Up(ngf * 2 ** (i + 1), ngf * 2 ** i // factor, bilinear))
            else:
                self.ups.append(Up(ngf * 2 ** (i + 1), ngf * 2 ** i, bilinear))
        self.outc = OutConv(ngf, out_channels)

    def forward(self, x):
        x = self.inc(x)  # 输入层双卷积

        downs = [x]
        for down in self.downs:
            x = down(x)
            downs.append(x)  # 下采样

        for down_out, up in zip(reversed(downs[:-1]), self.ups):
            x = up(x, down_out)  # 上采样

        logits = self.outc(x)  # 输出

        return logits
