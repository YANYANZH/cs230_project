
"""
    Define the neural network.
    Written by Yanyan Zhao.
"""

from model_block import *


class Net(nn.Module):

    def __init__(self, n_channels, n_classes):

        super(Net, self).__init__()
        self.inc = double_conv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256,512)
        self.down3_drop = nn.Dropout2d(0.5)
        self.down4 = down(512, 1024)
        self.down4_drop = nn.Dropout2d(0.5)
        self.up1 = up(1024, 512)
        self.up2 = up(512, 256)
        self.up3 = up(256, 128)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x3 = self.down3_drop(x3)
        x4 = self.down3(x3)
        x4 = self.down4_drop(x4)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        return x




