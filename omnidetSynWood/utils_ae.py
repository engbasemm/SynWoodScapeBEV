# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn


def find_named_module(module, query):
    """Helper function to find a named module. Returns a `nn.Module` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the module name to find

    Returns:
        nn.Module or None
    """

    return next((m for n, m in module.named_modules() if n == query), None)


def find_named_buffer(module, query):
    """Helper function to find a named buffer. Returns a `torch.Tensor` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the buffer name to find

    Returns:
        torch.Tensor or None
    """
    return next((b for n, b in module.named_buffers() if n == query), None)


def _update_registered_buffer(
    module,
    buffer_name,
    state_dict_key,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    new_size = state_dict[state_dict_key].size()
    registered_buf = find_named_buffer(module, buffer_name)

    if policy in ("resize_if_empty", "resize"):
        if registered_buf is None:
            raise RuntimeError(f'buffer "{buffer_name}" was not registered')

        if policy == "resize" or registered_buf.numel() == 0:
            registered_buf.resize_(new_size)

    elif policy == "register":
        if registered_buf is not None:
            raise RuntimeError(f'buffer "{buffer_name}" was already registered')

        module.register_buffer(buffer_name, torch.empty(new_size, dtype=dtype).fill_(0))

    else:
        raise ValueError(f'Invalid policy "{policy}"')


def update_registered_buffers(
    module,
    module_name,
    buffer_names,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    """Update the registered buffers in a module according to the tensors sized
    in a state_dict.

    (There's no way in torch to directly load a buffer with a dynamic size)

    Args:
        module (nn.Module): the module
        module_name (str): module name in the state dict
        buffer_names (list(str)): list of the buffer names to resize in the module
        state_dict (dict): the state dict
        policy (str): Update policy, choose from
            ('resize_if_empty', 'resize', 'register')
        dtype (dtype): Type of buffer to be registered (when policy is 'register')
    """
    valid_buffer_names = [n for n, _ in module.named_buffers()]
    for buffer_name in buffer_names:
        if buffer_name not in valid_buffer_names:
            raise ValueError(f'Invalid buffer name "{buffer_name}"')

    for buffer_name in buffer_names:
        _update_registered_buffer(
            module,
            buffer_name,
            f"{module_name}.{buffer_name}",
            state_dict,
            policy,
            dtype,
        )


def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )


import os
import time
import json
from itertools import product
from collections import namedtuple
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torchvision
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from IPython.display import display, clear_output
# ************************
# * Function definitions *
# ************************

def train_val_split(dataset, val_ratio=0.1):
    """
    Splits `dataset` into a training set and a validation set, by the given ratio `val_ratio`.
    """

    train_size = int((1 - val_ratio) * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    return train_set, val_set


def to_img(x):
    """
    Denormalises Tensor `x` (normalised from -1 to 1) to image format (from 0 to 1).
    """

    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    return x


def image_noiser(images, std=0.1):
    """
    Adds random zero-mean Gaussian noise of given standard deviation to image tensor.
    """
    mu = torch.zeros(images.shape)
    sigma = std * torch.ones(images.shape)
    noise = torch.normal(mu, sigma)
    noisy_images = images + noise
    return noisy_images.clamp(-1, 1)


def imgviz(images, save_fname=None, nrow=10):
    """
    Displays `images` in a grid, and saves (optionally) output as an image file with
    filename `save_fname`. Format of input must be a Tensor of size 4 (B, C, H, W).
    """

    assert len(images.shape) == 4, "Size of input images must be 4."

    grid = make_grid(to_img(images), nrow=nrow)
    plt.figure(figsize=(15, 15))
    plt.imshow(np.transpose(grid, (1, 2, 0)))

    # If save_fname is given and is a string, save image to local directory
    if type(save_fname) == str:
        if not os.path.exists('./saved_imgs'):
            os.mkdir('./saved_imgs')
        plt.imsave('./saved_imgs/' + save_fname, np.transpose(grid, (1, 2, 0)).numpy())

    # Show image
    plt.show()


# *********************
# * Class definitions *
# *********************

class Epoch:
    """
    Epoch class, keeps track of progress for training epochs
    """

    def __init__(self):
        self.count = 0
        self.train_loss = 0
        self.val_loss = 0
        self.train_batches = 0
        self.val_batches = 0
        self.start_time = None
        self.active = False

    # Initialises epoch if not active
    def begin(self):
        assert not self.active, "Cannot initialise epoch, already active"
        self.count += 1
        self.train_loss = 0
        self.val_loss = 0
        self.train_batches = 0
        self.val_batches = 0
        self.start_time = time.time()
        self.active = True

    # Finalises epoch if active and returns epoch results
    def end(self):
        assert self.active, "Cannot finalise epoch, no epoch active"
        duration = self.duration()
        train_loss = self.train_loss / self.train_batches
        val_loss = self.val_loss / self.val_batches
        self.active = False
        return duration, train_loss, val_loss

    # Adds batch loss to total loss
    def add_loss(self, loss, batch_size, mode='train'):
        if mode == 'val':
            self.val_loss += loss.item() * batch_size
            self.val_batches += batch_size
        else:
            self.train_loss += loss.item() * batch_size
            self.train_batches += batch_size

    # Returns duration of epoch processing time
    def duration(self):
        return time.time() - self.start_time


class Run:
    """
    Run class, keeping track of progress for training runs.
    """

    def __init__(self):
        self.hparams = None
        self.count = 0
        self.data = []
        self.start_time = None
        self.active = False

    # Initialises run if not active
    def begin(self, hparams):
        assert not self.active, "Cannot initialise run, already active"
        self.hparams = hparams
        self.count += 1
        self.start_time = time.time()
        self.active = True

    # Finalises run if active
    def end(self, epoch):
        assert self.active, "Cannot finalise run, no run active"
        epoch.count = 0
        self.active = False

    # Returns duration of run processing time
    def duration(self):
        return time.time() - self.start_time

    # Appends results to all data and displays data
    def append_and_display_data(self, results):
        # Append results to data
        self.data.append(results)

        # Create Pandas DataFrame to all data and display data
        df = pd.DataFrame.from_dict(self.data, orient='columns')
        clear_output(wait=True)
        display(df)

    # Saves all data as a json file
    def save(self, filename):
        pd.DataFrame.from_dict(self.data, orient='columns').to_csv(f'{filename}.csv')

        with open(f'{filename}.json', 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=4)


class RunManager:
    """
    RunManager class, keeping track of overall training progress.
    """

    def __init__(self):
        self.epoch = Epoch()
        self.run = Run()
        self.net = None
        self.images = None
        self.noisy_images = None
        self.tb = None
        self.latents = None
        self.min_val_loss = float('inf')

    def begin_run(self, hparams, net, test_images, test_noisy_images):
        # Begin next run with new hyperparameters
        self.run.begin(hparams)

        # Setup network, data and SummaryWriter
        self.net = net
        self.images = test_images
        self.noisy_images = test_noisy_images
        self.tb = SummaryWriter(comment=f'-{hparams}')

        # Add test images and graph to TensorBoard
        grid = make_grid(to_img(self.images), nrow=10)
        noisy_grid = make_grid(to_img(self.noisy_images), nrow=10)
        self.tb.add_image('original images', grid)
        self.tb.add_image('noisy images', noisy_grid)
        #self.tb.add_graph(self.net, to_img(self.images))
        self.save_img(grid.cpu(), 'original_images.png')
        self.save_img(noisy_grid.cpu(), 'noisy_images.png')

    def end_run(self):
        self.tb.flush()
        self.tb.close()
        self.net = None
        self.images = None
        self.noisy_images = None
        self.tb = None
        self.min_val_loss = float('inf')
        self.run.end(self.epoch)

    def begin_epoch(self):
        assert self.run.active, "Run is not active, cannot initialise epoch"
        self.epoch.begin()

    def end_epoch(self):
        run_duration = self.run.duration()
        epoch_duration, train_loss, val_loss = self.epoch.end()

        self.tb.add_scalar('Training loss', train_loss, self.epoch.count)
        self.tb.add_scalar('Validation loss', val_loss, self.epoch.count)

        with torch.no_grad():
            preds = self.net(self.images)
            if type(preds) is dict:
                if 'x_hat' in preds:
                    preds = preds["x_hat"]
            if type(preds) is list:
                    preds = preds[0]
            pred_imgs = to_img(preds)
            grid = make_grid(pred_imgs, nrow=10)
            self.tb.add_image('reconstructed images', grid, self.epoch.count)

        #for name, param in self.net.named_parameters():
        #    self.tb.add_histogram(name, param, self.epoch.count)
        #    self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch.count)

        if val_loss < self.min_val_loss:
            torch.save(self.net, './models/AEs/results_models_resnet_ae/best_' + str(self.run.hparams) + '.pth')
            self.min_val_loss = val_loss
            self.save_img(grid.cpu(), 'epoch{0}.png'.format(self.epoch.count))

        results = OrderedDict()
        results['run'] = self.run.count
        results['epoch'] = self.epoch.count
        results['train loss'] = train_loss
        results['validation loss'] = val_loss
        results['epoch duration'] = epoch_duration
        results['run duration'] = run_duration

        for k, v in self.run.hparams._asdict().items():
            results[k] = v

        self.run.append_and_display_data(results)

    def track_loss(self, loss, batch_size, mode='train'):
        self.epoch.add_loss(loss, batch_size, mode)

    def generate_images(self):
        with torch.no_grad():
            generated_imgs = nn.Sigmoid()(self.net.decoder(self.latents))
            imgviz(generated_imgs)

    def save(self, filename):
        self.run.save(filename)

    # Save image to local directory
    def save_img(self, grid, filename):
        if not os.path.exists('./gif'):
            os.mkdir('./gif')

        #plt.figure(figsize=(15, 15))
        plt.imsave('./gif/' + filename, np.transpose(grid, (1, 2, 0)).numpy())


class RunManagerVae:
    """
    RunManager class, keeping track of overall training progress.
    """

    def __init__(self):
        self.epoch = Epoch()
        self.run = Run()
        self.net = None
        self.tb = None
        self.images = None
        self.latents = None
        self.min_val_loss = float('inf')

    def begin_run(self, hparams, net, test_images, test_latents):
        # Begin next run with new hyperparameters
        self.run.begin(hparams)

        # Setup network, data and SummaryWriter
        self.net = net
        self.tb = SummaryWriter(comment=f'-{hparams}')
        self.latents = test_latents

        # Add images and graph to TensorBoard
        self.images = test_images
        grid = make_grid(self.images, nrow=10)
        self.tb.add_image('original images', grid)
        self.tb.add_graph(self.net, self.images)
        self.save_img(grid, 'original_images.png')

    def end_run(self):
        self.tb.flush()
        self.tb.close()
        self.net = None
        self.tb = None
        self.images = None
        self.latents = None
        self.min_val_loss = float('inf')
        self.run.end(self.epoch)

    def begin_epoch(self):
        assert self.run.active, "Run is not active, cannot initialise epoch"
        self.epoch.begin()

    def end_epoch(self):
        run_duration = self.run.duration()
        epoch_duration, train_loss, val_loss = self.epoch.end()

        self.tb.add_scalar('Training loss', train_loss, self.epoch.count)
        self.tb.add_scalar('Validation loss', val_loss, self.epoch.count)

        with torch.no_grad():
            pred_imgs = nn.Sigmoid()(self.net.decoder(self.latents))
            grid = make_grid(pred_imgs, nrow=10)
            self.tb.add_image('generated images', grid, self.epoch.count)

        for name, param in self.net.named_parameters():
            self.tb.add_histogram(name, param, self.epoch.count)
            self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch.count)

        if val_loss < self.min_val_loss:
            torch.save(self.net, './results_vae/best_' + str(self.run.hparams) + '.pth')
            self.min_val_loss = val_loss
            self.save_img(grid, 'epoch{0}.png'.format(self.epoch.count))

        results = OrderedDict()
        results['run'] = self.run.count
        results['epoch'] = self.epoch.count
        results['train loss'] = train_loss
        results['validation loss'] = val_loss
        results['epoch duration'] = epoch_duration
        results['run duration'] = run_duration

        for k, v in self.run.hparams._asdict().items():
            results[k] = v

        self.run.append_and_display_data(results)
        self.generate_images()

    def track_loss(self, loss, batch_size, mode='train'):
        self.epoch.add_loss(loss, batch_size, mode)

    def generate_images(self):
        with torch.no_grad():
            generated_imgs = nn.Sigmoid()(self.net.decoder(self.latents))
            imgviz(generated_imgs)

    def save(self, filename):
        self.run.save(filename)

    # Save image to local directory
    def save_img(self, grid, filename):
        if not os.path.exists('./gif'):
            os.mkdir('./gif')

        #plt.figure(figsize=(15, 15))
        plt.imsave('./gif/' + filename, np.transpose(grid, (1, 2, 0)).numpy())


class RunBuilder:
    """
    RunBuilder class, builds Run tuples that encapsulate hyperparameters for each training run.
    """

    @staticmethod
    def get_runs_product(params):
        Run = namedtuple("Run", params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs

    @staticmethod
    def get_runs_from_params(names, params):
        assert all([len(names) == len(p) for p in params.values()]), \
            "Length of names must be equal to length of parameters for each run"

        Run = namedtuple("Run", names)

        runs = []
        for v in params.values():
            runs.append(Run(*v))
        return runs