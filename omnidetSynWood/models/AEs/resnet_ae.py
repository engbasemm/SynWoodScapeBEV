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

from utils import *

print(torch.__version__)
print(torchvision.__version__)

# Define transform
transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

CIFAR_set = torchvision.datasets.CIFAR10(root='../data/CIFAR10', train=True,
                                         download=True, transform=transform)

test_set = torchvision.datasets.CIFAR10(root='../data/CIFAR10', train=False,
                                        download=True, transform=transform)

train_set, val_set = train_val_split(CIFAR_set, 0.1)


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
        self.BN = nn.BatchNorm2d(c_out)
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
        self.BN = nn.BatchNorm2d(16)
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


class Decoder(nn.Module):
    """
    Decoder class, mainly consisting of two residual blocks.
    """

    def __init__(self):
        super(Decoder, self).__init__()
        self.rb1 = ResBlock(64, 48, 2, 2, 0, 'decode')  # 48 4 4
        self.rb2 = ResBlock(48, 48, 2, 2, 0, 'decode')  # 48 8 8
        self.rb3 = ResBlock(48, 32, 3, 1, 1, 'decode')  # 32 8 8
        self.rb4 = ResBlock(32, 32, 2, 2, 0, 'decode')  # 32 16 16
        self.rb5 = ResBlock(32, 16, 3, 1, 1, 'decode')  # 16 16 16
        self.rb6 = ResBlock(16, 16, 2, 2, 0, 'decode')  # 16 32 32
        self.out_conv = nn.ConvTranspose2d(16, 3, 3, 1, 1)  # 3 32 32
        self.tanh = nn.Tanh()

    def forward(self, inputs):
        rb1 = self.rb1(inputs)
        rb2 = self.rb2(rb1)
        rb3 = self.rb3(rb2)
        rb4 = self.rb4(rb3)
        rb5 = self.rb5(rb4)
        rb6 = self.rb6(rb5)
        out_conv = self.out_conv(rb6)
        output = self.tanh(out_conv)
        return output


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

print("Number of parameters in model: {0}".format(Autoencoder().num_params))

param_names = ('init_lr', 'batch_size', 'weight_decay')
parameters = OrderedDict(
    run = [0.05, 256, 0.001],
)

m = RunManager()
num_epochs = 3

for hparams in RunBuilder.get_runs_from_params(param_names, parameters):

    # Instantiate a network model
    ae = Autoencoder()

    # Construct a DataLoader object with training data
    train_loader = DataLoader(train_set, batch_size=hparams.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=hparams.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=10, shuffle=False)
    test_images, _ = next(iter(test_loader))
    test_noisy_images = image_noiser(test_images)
    # Define optimizer
    optimizer = optim.SGD(ae.parameters(), lr=hparams.init_lr, momentum=0.9, weight_decay=hparams.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 60, 0.1)

    # Setup run instance
    m.begin_run(hparams, ae, test_images,test_noisy_images)
    print('Now training model with hyperparameters: init_lr={0}, batch_size={1}, weight_decay={2}'
          .format(hparams.init_lr, hparams.batch_size, hparams.weight_decay))

    # Start training loop
    for epoch in range(num_epochs):
        m.begin_epoch()

        # Train the model
        for i, batch in enumerate(train_loader):
            images, _ = batch

            # Zero all gradients
            optimizer.zero_grad()

            # Calculating the loss
            preds = ae(images)
            loss = F.mse_loss(preds, images)

            if i % 10 == 0:
                with torch.no_grad():
                    val_images, _ = next(iter(val_loader))
                    val_preds = ae(val_images)
                    val_loss = F.mse_loss(val_preds, val_images)
                    m.track_loss(val_loss, val_images.size(0), mode='val')
                print('Epoch {0}, iteration {1}: train loss {2}, val loss {3}'.format(epoch + 1,
                                                                                      i * hparams.batch_size,
                                                                                      round(loss.item(), 6),
                                                                                      round(val_loss.item(), 6)))

            # Backpropagate
            loss.backward()

            # Update the weights
            optimizer.step()

            m.track_loss(loss, images.size(0), mode='train')

        m.end_epoch()

    # torch.save(ae, './models/150epochs_' + str(hparams) + '.pth')
    m.end_run()
    print("Model has finished training.\n")
    scheduler.step()

m.save('results_final')
print("Training completed.")

# Load best model
ae = torch.load('./../final_best_Run(init_lr=0.05, batch_size=256, weight_decay=0.001).pth')

print("Visualising test images...\n")

test_loader = DataLoader(test_set, batch_size=10, shuffle=True)
images, _ = next(iter(test_loader))
print("Original images:")
imgviz(images)
print("Reconstructed images:")
with torch.no_grad():
    preds = ae(images)
    imgviz(preds)

# Test loss
test_loader = DataLoader(test_set, batch_size=len(test_set))
for i, batch in enumerate(test_loader):
    images, _ = batch
    with torch.no_grad():
        preds = ae(images)
        loss = F.mse_loss(preds, images)  # calculates the loss
print('Test loss:', loss.item())

transform = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(256),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

kodak_set = torchvision.datasets.ImageFolder(root='../data/Kodak', transform=transform)
kodak_loader = DataLoader(kodak_set, batch_size=24)

# Validate with Kodak
kodak, _ = next(iter(kodak_loader))
print("Original Kodak images:")
imgviz(kodak, save_fname='kodak_imgs.png', nrow=8)
print("Reconstructed Kodak images:")
with torch.no_grad():
    kodak_preds = ae(kodak)
    imgviz(kodak_preds, save_fname='reconstruced_kodak_imgs.png', nrow=8)
    kodak_loss = F.mse_loss(kodak_preds, kodak)
    print('Kodak MSE loss:', kodak_loss.item())