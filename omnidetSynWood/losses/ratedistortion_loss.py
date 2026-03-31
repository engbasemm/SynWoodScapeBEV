"""
Compression loss functions for OmniDet.

# author: Basem Barakat <eng.basem.ahmed@gmail.com>

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.
"""

import torch
from torch import nn
import torch.nn.functional as F
import math

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = 0.08 #0.01 #0.05 #0.08 #0.13 #0.03 #lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W
        if type(output) is dict:
            if 'likelihoods' in output:
                out["bpp_loss"] = sum(
                    (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                    for likelihoods in output["likelihoods"].values()
                )
                bpp_loss = out["bpp_loss"]
            else:
                bpp_loss = 0

            recons = output["x_hat"]

        else:
            bpp_loss =  0
            recons = output
        out["mse_loss"] = self.mse(recons, target)
        out["compression_loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + bpp_loss

        return out