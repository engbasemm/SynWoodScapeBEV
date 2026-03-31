"""
BEV Neck module for OmniDet.

# author: Basem Barakat

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.
"""

import torch
import torch.nn as nn

class BEVNeck(nn.Module):
    """
    A simple neck for BEV feature maps.
    It takes a single BEV feature map and generates a multi-scale feature pyramid
    by downsampling. This mimics the output of an image-based FPN, making it
    compatible with existing decoders.
    """
    def __init__(self, in_channels, out_channels_pyramid):
        """
        Args:
            in_channels (int): Number of channels of the input BEV feature map.
            out_channels_pyramid (list[int]): A list of channel numbers for each level
                                              of the output pyramid. The length of the list
                                              determines the number of pyramid levels.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels_pyramid = out_channels_pyramid
        
        self.convs = nn.ModuleList()
        
        current_channels = self.in_channels
        for i, out_ch in enumerate(self.out_channels_pyramid):
            if i == 0:
                # First layer just adjusts channels without downsampling
                stride = 1
            else:
                # Subsequent layers downsample
                stride = 2
            
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(current_channels, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                )
            )
            current_channels = out_ch

    def forward(self, x):
        pyramid_features = []
        for conv in self.convs:
            x = conv(x)
            pyramid_features.append(x)
        return pyramid_features