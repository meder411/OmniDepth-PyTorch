# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
__author__ = "Marc Eder"
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

import torch
import torch.nn as nn
import torch.nn.functional as F

from util import xavier_init


class RectNet(nn.Module):

    def __init__(self):
        super(RectNet, self).__init__()

        # Network definition
        self.input0_0 = ConvELUBlock(3, 8, (3, 9), padding=(1, 4))
        self.input0_1 = ConvELUBlock(3, 8, (5, 11), padding=(2, 5))
        self.input0_2 = ConvELUBlock(3, 8, (5, 7), padding=(2, 3))
        self.input0_3 = ConvELUBlock(3, 8, 7, padding=3)

        self.input1_0 = ConvELUBlock(32, 16, (3, 9), padding=(1, 4))
        self.input1_1 = ConvELUBlock(32, 16, (3, 7), padding=(1, 3))
        self.input1_2 = ConvELUBlock(32, 16, (3, 5), padding=(1, 2))
        self.input1_3 = ConvELUBlock(32, 16, 5, padding=2)

        self.encoder0_0 = ConvELUBlock(64, 128, 3, stride=2, padding=1)
        self.encoder0_1 = ConvELUBlock(128, 128, 3, padding=1)
        self.encoder0_2 = ConvELUBlock(128, 128, 3, padding=1)

        self.encoder1_0 = ConvELUBlock(128, 256, 3, stride=2, padding=1)
        self.encoder1_1 = ConvELUBlock(256, 256, 3, padding=2, dilation=2)
        self.encoder1_2 = ConvELUBlock(256, 256, 3, padding=4, dilation=4)
        self.encoder1_3 = ConvELUBlock(512, 256, 1)

        self.encoder2_0 = ConvELUBlock(256, 256, 3, padding=8, dilation=8)
        self.encoder2_1 = ConvELUBlock(256, 512, 3, padding=16, dilation=16)
        self.encoder2_2 = ConvELUBlock(768, 512, 1)

        self.decoder0_0 = ConvTransposeELUBlock(
            512, 256, 4, stride=2, padding=1)
        self.decoder0_1 = ConvELUBlock(256, 256, 5, padding=2)

        self.prediction0 = nn.Conv2d(256, 1, 3, padding=1)

        self.decoder1_0 = ConvTransposeELUBlock(
            256, 128, 4, stride=2, padding=1)
        self.decoder1_1 = ConvELUBlock(128, 128, 5, padding=2)
        self.decoder1_2 = ConvELUBlock(129, 64, 1)

        self.prediction1 = nn.Conv2d(64, 1, 3, padding=1)

        # Initialize the network weights
        self.apply(xavier_init)

    def forward(self, x):

        # First filter bank
        input0_0_out = self.input0_0(x)
        input0_1_out = self.input0_1(x)
        input0_2_out = self.input0_2(x)
        input0_3_out = self.input0_3(x)
        input0_out_cat = torch.cat(
            (input0_0_out, input0_1_out, input0_2_out, input0_3_out), 1)

        # Second filter bank
        input1_0_out = self.input1_0(input0_out_cat)
        input1_1_out = self.input1_1(input0_out_cat)
        input1_2_out = self.input1_2(input0_out_cat)
        input1_3_out = self.input1_3(input0_out_cat)

        # First encoding block
        encoder0_0_out = self.encoder0_0(
            torch.cat((input1_0_out, input1_1_out, input1_2_out, input1_3_out),
                      1))
        encoder0_1_out = self.encoder0_1(encoder0_0_out)
        encoder0_2_out = self.encoder0_2(encoder0_1_out)

        # Second encoding block
        encoder1_0_out = self.encoder1_0(encoder0_2_out)
        encoder1_1_out = self.encoder1_1(encoder1_0_out)
        encoder1_2_out = self.encoder1_2(encoder1_1_out)
        encoder1_3_out = self.encoder1_3(
            torch.cat((encoder1_1_out, encoder1_2_out), 1))

        # Third encoding block
        encoder2_0_out = self.encoder2_0(encoder1_3_out)
        encoder2_1_out = self.encoder2_1(encoder2_0_out)
        encoder2_2_out = self.encoder2_2(
            torch.cat((encoder2_0_out, encoder2_1_out), 1))

        # First decoding block
        decoder0_0_out = self.decoder0_0(encoder2_2_out)
        decoder0_1_out = self.decoder0_1(decoder0_0_out)

        # 2x downsampled prediction
        pred_2x = self.prediction0(decoder0_1_out)
        upsampled_pred_2x = F.interpolate(pred_2x, scale_factor=2)

        # Second decoding block
        decoder1_0_out = self.decoder1_0(decoder0_1_out)
        decoder1_1_out = self.decoder1_1(decoder1_0_out)
        decoder1_2_out = self.decoder1_2(
            torch.cat((decoder1_1_out, upsampled_pred_2x), 1))

        # Second prediction output (original scale)
        pred_1x = self.prediction1(decoder1_2_out)

        return [pred_1x, pred_2x]


# -----------------------------------------------------------------------------
class UResNet(nn.Module):

    def __init__(self):
        super(UResNet, self).__init__()

        self.input0 = ConvELUBlock(
            in_channels=3, out_channels=32, kernel_size=7, stride=1, padding=3)
        self.input1 = ConvELUBlock(
            in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)

        self.encoder0 = SkipBlock(64, 128)
        self.encoder1 = SkipBlock(128, 256)
        self.encoder2 = SkipBlock(256, 512)
        self.encoder3 = SkipBlock(512, 1024)

        self.decoder0_0 = ConvTransposeELUBlock(
            in_channels=1024,
            out_channels=512,
            kernel_size=4,
            stride=2,
            padding=1)
        self.decoder0_1 = ConvELUBlock(
            in_channels=512,
            out_channels=512,
            kernel_size=5,
            stride=1,
            padding=2)
        self.decoder1_0 = ConvTransposeELUBlock(
            in_channels=512,
            out_channels=256,
            kernel_size=4,
            stride=2,
            padding=1)
        self.decoder1_1 = ConvELUBlock(
            in_channels=256,
            out_channels=256,
            kernel_size=5,
            stride=1,
            padding=2)
        self.decoder2_0 = ConvTransposeELUBlock(
            in_channels=256,
            out_channels=128,
            kernel_size=4,
            stride=2,
            padding=1)
        self.decoder2_1 = ConvELUBlock(
            in_channels=128 + 1,
            out_channels=128,
            kernel_size=5,
            stride=1,
            padding=2)
        self.decoder3_0 = ConvTransposeELUBlock(
            in_channels=128,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=1)
        self.decoder3_1 = ConvELUBlock(
            in_channels=64 + 1,
            out_channels=64,
            kernel_size=5,
            stride=1,
            padding=2)

        self.prediction0 = nn.Conv2d(256, 1, 3, padding=1)
        self.prediction1 = nn.Conv2d(128, 1, 3, padding=1)
        self.prediction2 = nn.Conv2d(64, 1, 3, padding=1)

        self.apply(xavier_init)

    def forward(self, x):

        # Encode down to 4x
        x = self.input0(x)
        x = self.input1(x)
        x = self.encoder0(x)
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        x = self.decoder0_0(x)
        x = self.decoder0_1(x)
        x = self.decoder1_0(x)
        x = self.decoder1_1(x)

        # Predict at 4x downsampled
        pred_4x = self.prediction0(x)

        # Upsample through convolution to 2x
        x = self.decoder2_0(x)
        upsampled_pred_4x = F.interpolate(pred_4x.detach(), scale_factor=2)

        # Predict at 2x downsampled
        x = self.decoder2_1(torch.cat((x, upsampled_pred_4x), 1))
        pred_2x = self.prediction1(x)

        # Upsample through convolution to 1x
        x = self.decoder3_0(x)
        upsampled_pred_2x = F.interpolate(pred_2x.detach(), scale_factor=2)

        # Predict at 1x
        x = self.decoder3_1(torch.cat((x, upsampled_pred_2x), 1))
        pred_1x = self.prediction2(x)

        return [pred_1x, pred_2x, pred_4x]


# -----------------------------------------------------------------------------
class ConvELUBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1):
        super(ConvELUBlock, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation)

    def forward(self, x):
        return F.elu(self.conv(x), inplace=True)


# -----------------------------------------------------------------------------
class ConvTransposeELUBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1):
        super(ConvTransposeELUBlock, self).__init__()

        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation)

    def forward(self, x):
        return F.elu(self.conv(x), inplace=True)


# -----------------------------------------------------------------------------
class SkipBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(SkipBlock, self).__init__()

        self.conv1 = ConvELUBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1)
        self.conv2 = ConvELUBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1)
        self.conv3 = ConvELUBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1)

    def forward(self, x):

        # First convolutional block
        out1 = self.conv1(x)

        # Second and third convolutional blocks
        out3 = self.conv3(self.conv2(out1))

        # Return the sum of the outputs of the first block and the third block
        return out1 + out3