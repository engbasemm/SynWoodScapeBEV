"""
BEV Stitching Decoder for OmniDet.

# author: Basem Barakat

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.
"""

import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

# Helper layers (assuming they are defined in a layers.py or similar, included here for completeness)
class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)
    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out

class Conv3x3(nn.Module):
    """Layer to pad and convolve images"""
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()
        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)
    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

def upsample(x):
    """Upsample input tensor by a factor of 2"""
    return nn.functional.interpolate(x, scale_factor=2, mode="nearest")


class StitchingDecoder(nn.Module):
    """
    A U-Net-like decoder to stitch fused multi-view features into a BEV image.
    """
    def __init__(self, encoder_channels, n_classes=3):
        super(StitchingDecoder, self).__init__()
        self.num_ch_enc = encoder_channels
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.n_classes = n_classes # BEV GT classes
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)
            num_ch_in = self.num_ch_dec[i]
            if i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
        self.convs[("bev_stitch", 0)] = Conv3x3(self.num_ch_dec[0], self.n_classes)
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        outputs = {}
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            if i > 0:
                x = upsample(x)
                skip_connection = input_features[i - 1]
                # Resize x to match the spatial dimensions of the skip connection to handle odd sizes
                if x.shape != skip_connection.shape:
                    x = nn.functional.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=False)
                x = torch.cat([x, skip_connection], 1)
            x = self.convs[("upconv", i, 1)](x)
        bev_output = self.convs[("bev_stitch", 0)](x)
        outputs[("bev_stitch", 0)] = self.sigmoid(bev_output)  # Apply sigmoid here
        return outputs