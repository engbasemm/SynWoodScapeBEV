# Imports
from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)

import torchvision
from torchvision.transforms import transforms
#from omnidet.models.compressAI.priors import FactorizedPrior



class ResBlock(nn.Module):
    """
    A two-convolutional layer residual block.
    """

    def __init__(self, c_in, c_out, k, s=1, p=1, mode='encode'):
        assert mode in ['encode', 'decode'], "Mode must be either 'encode' or 'decode'."
        super(ResBlock, self).__init__()
        if mode == 'encode':
            self.conv1 = nn.Conv2d(c_in, c_out, k, s, p)
            self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1)
        elif mode == 'decode':
            self.conv1 = nn.ConvTranspose2d(c_in, c_out, k, s, p)
            self.conv2 = nn.ConvTranspose2d(c_out, c_out, 3, 1, 1)
        self.relu = nn.ReLU()
        #self.BN = nn.GroupNorm (c_out,c_out)
        self.BN = nn.BatchNorm2d(c_out, track_running_stats=False)

        self.resize = s > 1 or (s == 1 and p == 0) or c_out != c_in

    def forward(self, x):
        conv1 = self.BN(self.conv1(x))
        relu = self.relu(conv1)
        conv2 = self.BN(self.conv2(relu))
        if self.resize:
            x = self.BN(self.conv1(x))
        return self.relu(x + conv2)


"""
class Encoder(nn.Module):
    "
    Encoder class, mainly consisting of three residual blocks.
    "

    def __init__(self):
        super(Encoder, self).__init__()
        self.init_conv = nn.Conv2d(3, 48, 3, 3, 0) # 48 10 10
        self.BN = nn.BatchNorm2d(48)
        self.rb1 = ResBlock(48, 48, 3, 2, 1, 'encode') # 48 5 5
        self.rb2 = ResBlock(48, 24, 3, 2, 1, 'encode') # 24 3 3
        self.rb3 = ResBlock(24, 24, 2, 1, 0, 'encode') # 24 2 2
        self.relu = nn.ReLU()

    def forward(self, inputs):
        init_conv = self.relu(self.BN(self.init_conv(inputs)))
        rb1 = self.rb1(init_conv)
        rb2 = self.rb2(rb1)
        rb3 = self.rb3(rb2)
        return rb3
"""


class Encoder(nn.Module):
    """
    Encoder class, mainly consisting of three residual blocks.
    """

    def __init__(self):
        super(Encoder, self).__init__()
        self.init_conv = nn.Conv2d(3, 16, 3, 1, 1)  # 16 32 32
        self.BN = nn.BatchNorm2d(16,track_running_stats=False)
        #self.BN = nn.GroupNorm (16,16)
        self.rb1 = ResBlock(16, 16, 3, 2, 1, 'encode')  # 16 16 16
        self.rb2 = ResBlock(16, 32, 3, 1, 1, 'encode')  # 32 16 16
        self.rb3 = ResBlock(32, 32, 3, 2, 1, 'encode')  # 32 8 8
        self.rb4 = ResBlock(32, 48, 3, 1, 1, 'encode')  # 48 8 8
        self.rb5 = ResBlock(48, 48, 3, 2, 1, 'encode')  # 48 4 4
        self.rb6 = ResBlock(48, 64, 3, 2, 1, 'encode')  # 64 2 2
        self.relu = nn.ReLU()

    def forward(self, inputs):
        init_conv = self.relu(self.BN(self.init_conv(inputs)))
        rb1 = self.rb1(init_conv)
        rb2 = self.rb2(rb1)
        rb3 = self.rb3(rb2)
        rb4 = self.rb4(rb3)
        rb5 = self.rb5(rb4)
        rb6 = self.rb6(rb5)
        return rb6


"""
class Decoder(nn.Module):
    "
    Decoder class, mainly consisting of two residual blocks.
    "

    def __init__(self):
        super(Decoder, self).__init__()
        self.relu = nn.ReLU()
        self.rb1 = ResBlock(24, 48, 3, 2, 0, 'decode') # 48 5 5
        self.rb2 = ResBlock(48, 24, 5, 3, 0, 'decode') # 24 17 17
        self.out_conv = nn.ConvTranspose2d(24, 3, 2, 2, 1) # 3 32 32
        self.tanh = nn.Tanh()

    def forward(self, inputs):
        relu = self.relu(inputs)
        rb1 = self.rb1(relu)
        rb2 = self.rb2(rb1)
        out_conv = self.out_conv(rb2)
        output = self.tanh(out_conv)
        return output
"""


class DecoderIn64(nn.Module):
    """
    Decoder class, mainly consisting of two residual blocks.
    """

    def __init__(self):
        super(DecoderIn64, self).__init__()
        self.rb1 = ResBlock(64, 48, 2, 2, 0, 'decode')  # 48 4 4
        self.rb2 = ResBlock(48, 48, 2, 2, 0, 'decode')  # 48 8 8
        self.rb3 = ResBlock(48, 16, 3, 1, 1, 'decode')  # 32 8 8
        self.rb4 = ResBlock(16, 16, 3, 1, 1, 'decode')  # 32 16 16
        #self.rb5 = ResBlock(32, 16, 3, 1, 1, 'decode')  # 16 16 16
        #self.rb6 = ResBlock(16, 16, 2, 2, 0, 'decode')  # 16 32 32
        self.out_conv = nn.ConvTranspose2d(16, 3, 3, 1, 1)  # 3 32 32
        self.tanh = nn.Tanh()

    def forward(self, inputs):
        rb1 = self.rb1(inputs[1])
        rb2 = self.rb2(rb1)
        rb3 = self.rb3(rb2)
        rb4 = self.rb4(rb3)
        #rb5 = self.rb5(rb4)
        #rb6 = self.rb6(rb5)
        out_conv = self.out_conv(rb4)
        output = self.tanh(out_conv)
        return output

class DecoderIn512(nn.Module):
    """
    Decoder class, mainly consisting of two residual blocks.
    """

    def __init__(self):
        super(DecoderIn512, self).__init__()
        self.rb1 = ResBlock(512, 192, 2, 2, 0, 'decode')  # 48 4 4
        self.rb2 = ResBlock(192, 192, 2, 2, 0, 'decode')  # 48 4 4
        self.rb3 = ResBlock(192, 128, 2, 2, 0, 'decode')  # 48 8 8
        self.rb4 = ResBlock(128, 64, 2, 2, 0, 'decode')  # 48 8 8
        self.rb5 = ResBlock(64, 32, 3, 1, 1, 'decode')  # 32 8 8
        self.rb6 = ResBlock(32, 16, 2, 2, 0, 'decode')  # 32 16 16
        self.out_conv = nn.ConvTranspose2d(16, 3, 3, 1, 1)  # 3 32 32
        self.tanh = nn.Tanh()

    def forward(self, inputs):
        rb1 = self.rb1(inputs[4])
        rb2 = self.rb2(rb1)
        rb3 = self.rb3(rb2)
        rb4 = self.rb4(rb3)
        rb5 = self.rb5(rb4)
        rb6 = self.rb6(rb5)
        out_conv = self.out_conv(rb6)
        output = self.tanh(out_conv)
        return output

class DecoderIn256(nn.Module):
    """
    Decoder class, mainly consisting of two residual blocks.
    """

    def __init__(self):
        super(DecoderIn256, self).__init__()
        self.rb1 = ResBlock(256, 192, 2, 2, 0, 'decode')  # 48 4 4
        self.rb2 = ResBlock(192, 192, 2, 2, 0, 'decode')  # 48 4 4
        self.rb3 = ResBlock(192, 128, 3, 1, 1, 'decode')  # 48 8 8
        self.rb4 = ResBlock(128, 64, 3, 1, 1, 'decode')  # 48 8 8
        self.rb5 = ResBlock(64, 32, 3, 1, 1, 'decode')  # 32 8 8
        self.rb6 = ResBlock(32, 16, 2, 2, 0, 'decode')  # 32 16 16
        self.rb7 = ResBlock(16, 16, 2, 2, 0, 'decode')  # 32 16 16
        self.out_conv = nn.ConvTranspose2d(16, 3, 3, 1, 1)  # 3 32 32
        self.tanh = nn.Tanh()

    def forward(self, inputs):
        rb1 = self.rb1(inputs[3])
        rb2 = self.rb2(rb1)
        rb3 = self.rb3(rb2)
        rb4 = self.rb4(rb3)
        rb5 = self.rb5(rb4)
        rb6 = self.rb6(rb5)
        rb7 = self.rb7(rb6)
        out_conv = self.out_conv(rb7)
        output = self.tanh(out_conv)
        return output

class Block(nn.Module):
    """
    U-Net block, consisting mainly of three convolutional filters with
    batch normalisation and ReLU activation functions.

    'down' blocks downsample at the output layer and outputs both downsampled
    and non-downsampled activations.

    'up' blocks concatenate non-downsampled corresponding feature maps and
    upsample at the output layer.

    'out' blocks concatenate non-downsampled corresponding feature maps and
    outputs the final feature maps, representing the final output layer of
    the model.
    """

    def __init__(self, in_channels, out_channels, direction='down'):
        assert direction in ['down', 'up', 'out'], "Direction must be either 'down', 'up' or 'out'."
        super(Block, self).__init__()
        if direction == 'down':
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
            self.out = nn.Conv2d(out_channels, out_channels, 2, 2, 0)
        else:
            self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, 3, 1, 1)
            self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, 3, 1, 1)
            if direction == 'up':
                self.out = nn.ConvTranspose2d(out_channels, out_channels // 2, 2, 2, 0)
            elif direction == 'out':
                self.out = nn.ConvTranspose2d(out_channels, 3, 2, 2, 0)

        self.BN1 = nn.BatchNorm2d(out_channels, track_running_stats=False)
        self.relu1 = nn.ReLU()
        self.BN2 = nn.BatchNorm2d(out_channels, track_running_stats=False)
        self.relu2 = nn.ReLU()
        self.direction = direction

    def forward(self, inputs, cat_layers=None):
        if self.direction != 'down':
            assert cat_layers is not None, "'up' and 'out' directions must have concatenated layers."
            assert inputs.shape == cat_layers.shape, "Shape of both inputs and concatenated layers must be equal."
            inputs = torch.cat((inputs, cat_layers), dim=1)

        conv1 = self.conv1(inputs)
        BN1 = self.BN1(conv1)
        relu1 = self.relu1(BN1)
        conv2 = self.conv2(relu1)
        BN2 = self.BN2(conv2)
        relu2 = self.relu2(BN2)
        out = self.out(relu2)
        if self.direction == 'down':
            return out, relu2
        else:
            return out

class UnetDecod(nn.Module):
    """
    Decoder class, consists of two 'up' blocks and a final 'out' block.
    """

    def __init__(self,size=1):
        super(UnetDecod, self).__init__()
        self.block1 = Block(128*size, 64*size, 'up')
        self.block2 = Block(64*size, 32*size, 'up')
        self.block3 = Block(32*size, 16*size, 'out')

    def forward(self, inputs, concats):
        block1 = self.block1(inputs, concats[-1])
        block2 = self.block2(block1, concats[-2])
        block3 = self.block3(block2, concats[-3])
        return block3


class UnetDecoder(nn.Module):
    """
    Autoencoder class, combines encoder and decoder model with a bottleneck
    layer in between.
    """

    def __init__(self , size=1):
        super(UnetDecoder, self).__init__()
        self.decoder = UnetDecod(size)
        self.adjustFirstLayer =  nn.ConvTranspose2d(32*size, 16*size, 3, 1, 1)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64*size, 128*size, 3, 1, 1),
            nn.BatchNorm2d(128*size, track_running_stats=False),
            nn.ReLU(),
            nn.Conv2d(128*size, 128*size, 3, 1, 1),
            nn.BatchNorm2d(128*size, track_running_stats=False),
            nn.ReLU(),
            nn.ConvTranspose2d(128*size, 64*size, 3, 1, 1)
        )

    @property
    def num_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        num_p = sum([np.prod(p.size()) for p in model_parameters])
        return num_p

    def forward(self, main_concatted ):
        concatted = main_concatted[0:3]
        first_layer = self.adjustFirstLayer(concatted[0])
        concatted[0] = first_layer
        bottlenecked = self.bottleneck(concatted[-1])
        decoded = self.decoder(bottlenecked, concatted)
        #added = inputs + decoded
        return decoded#added

class gAutoencoder(nn.Module):
    """
    Autoencoder class, combines encoder and decoder model.
    """
    def __init__(self , encoder,decoder):
        super(gAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    @property
    def num_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        num_p = sum([np.prod(p.size()) for p in model_parameters])
        return num_p

    def forward(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

class Autoencoder(nn.Module):
    """
    Autoencoder class, combines encoder and decoder model.
    """

    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    @property
    def num_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        num_p = sum([np.prod(p.size()) for p in model_parameters])
        return num_p

    def forward(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded


