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

class CompressionLoss(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self,recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

