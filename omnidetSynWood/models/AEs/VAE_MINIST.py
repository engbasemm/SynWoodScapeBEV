import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import os
from PIL import Image
import numpy as np
from torchvision.datasets import ImageFolder
import torchvision


class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x


class ResNet18Enc(nn.Module):
    def __init__(self, z_dim=32):
        super(ResNet18Enc, self).__init__()
        self.z_dim = z_dim
        self.ResNet18 = models.resnet18(pretrained=True)
        self.num_feature = self.ResNet18.fc.in_features
        self.ResNet18.fc = nn.Linear(self.num_feature, 2 * self.z_dim)

    def forward(self, x):
        x = self.ResNet18(x)
        mu = x[:, :self.z_dim]
        logvar = x[:, self.z_dim:]
        return mu, logvar


class BasicBlockDec(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes / stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet18Dec(nn.Module):

    def __init__(self, num_Blocks=[2, 2, 2, 2], z_dim=32, nc=3):
        super().__init__()
        self.in_planes = 512

        self.linear = nn.Linear(z_dim, 512)

        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)
        self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=2)

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        x = self.linear(z)
        x = x.view(z.size(0), 512, 1, 1)
        x = F.interpolate(x, scale_factor=7)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = F.interpolate(x, size=(112, 112), mode='bilinear')
        x = torch.sigmoid(self.conv1(x))
        x = x.view(x.size(0), 3, 224, 224)
        return x


class VAE(nn.Module):
    def __init__(self, z_dim):
        super(VAE, self).__init__()
        self.encoder = ResNet18Enc(z_dim=z_dim)
        self.decoder = ResNet18Dec(z_dim=z_dim)

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        x = self.decoder(z)
        return x, mean, logvar

    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(logvar / 2)  # in log-space, squareroot is divide by two
        epsilon = torch.randn_like(std).cuda()
        return epsilon * std + mean


def loss_func(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD




epoch_num = 1
batch_size = 16
vae = VAE(z_dim=256).cuda()
optimizer = optim.Adam(vae.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

root = "./dataset/"



transform = transforms.Compose([transforms.Resize([224, 224]),
                                transforms.ToTensor(),
                                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                # gray -> GRB 3 channel (lambda function)
                                transforms.Normalize(mean=[0.0, 0.0, 0.0],
                                                     std=[1.0, 1.0, 1.0])])  # for grayscale images

# MNIST dataset (images and labels)
MNIST_train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
MNIST_test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)

# Data loader (input pipeline)
train_iter = torch.utils.data.DataLoader(dataset=MNIST_train_dataset, batch_size=batch_size, shuffle=True)
test_iter = torch.utils.data.DataLoader(dataset=MNIST_test_dataset, batch_size=batch_size, shuffle=False)

for epoch in range(0, epoch_num):
    l_sum = 0
    scheduler.step()
    print(epoch, "\n")
    for x, y in train_iter:
        # x = torch.sigmoid(x).cuda()
        x = x.cuda()
        #print(x.requires_grad)
        optimizer.zero_grad()
        recon_x, mu, logvar = vae.forward(x)
        utils.save_image(recon_x.data, 'test' + '.png', normalize=True)
        loss = loss_func(recon_x, x, mu, logvar)
        l_sum += loss
        loss.backward()
        optimizer.step()
    print("loss\n", l_sum)


i = 0
with torch.no_grad():
    for t_img, y in test_iter:
        t_img = Variable(t_img).cuda()
        result, mu, logvar = vae.forward(t_img)
        utils.save_image(result.data, str(i) + '.png', normalize=True)
        i += 1