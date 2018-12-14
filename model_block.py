import torch
import torch.nn as nn

class double_conv(nn.Module):
    '''(conv=> BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x = self.conv(x)
        return x


class down(nn.Module):
    '''maxpooling + double_conv'''
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()

        self.mplconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch,out_ch)
        )

    def forward(self, x):
        x = self.mplconv(x)
        return x

class up(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        )
        self.conv = double_conv(in_ch,out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2,x1], dim =1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv(x)
        return x


















