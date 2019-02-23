# imports
from __future__ import print_function
import argparse
import os
import shutil
import sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.distributions as distr
import models._DRAGAN as dcgm
from utils.utils import Logger
from utils.utils import MidpointNormalize
import subprocess
import errno
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.serialization import load_lua
import torchfile

nc = 1
ndf = 64
outf = '{}/{}'.format('output', os.path.splitext(os.path.basename(sys.argv[0]))[0])
# Create Logger instance
logger = Logger(model_name='LRPGAN', data_name='random', dir_name=outf, make_fresh=False)
print('Created Logger')



class Discriminator(nn.Module):
    def __init__(self, d=128):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, 1, 4, 1, 0)

    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = torch.sigmoid(self.conv5(x))

        return x


discriminator = Discriminator()
dict = torch.load('./dcgan_discriminator.pth', map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.__version__ == '0.4.0':
    del dict['conv2_bn.num_batches_tracked']
    del dict['conv3_bn.num_batches_tracked']
    del dict['conv4_bn.num_batches_tracked']
discriminator.load_state_dict(dict)
discriminator.eval()

root_dir = 'dataset/custom'
dataset = datasets.ImageFolder(root=root_dir, transform=transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
))

# out_dir = 'dataset/MNIST'
# dataset = datasets.MNIST(root=out_dir, train=True, download=True,
#                          transform=transforms.Compose(
#                              [
#                                  transforms.Resize(64),
#                                  transforms.ToTensor(),
#                                  transforms.Normalize((0.5,), (0.5,)),
#                              ]
#                          ))

out_dir = 'dataset/cifar10'
dataset = datasets.CIFAR10(root=out_dir, download=True, train=True,
                           transform=transforms.Compose([
                               transforms.Resize(64),
                               transforms.Grayscale(num_output_channels=1),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

root_dir = 'dataset/faces'
dataset = datasets.ImageFolder(root=root_dir, transform=transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
))

nc = 1

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                         shuffle=False, num_workers=2)

for n_batch, (batch_data, _) in enumerate(dataloader, 0):
    # Perturbing input
    # ##############################################################################################
    # batch_data = batch_data + 0.02 * torch.randn(1, nc, opt.imageSize, opt.imageSize, device=gpu)
    # batch_data = torch.randn(opt.batchSize, nc, opt.imageSize, opt.imageSize, device=gpu)
    # batch_data = utils.pink_noise(1, nc, opt.imageSize, opt.imageSize).to(gpu)
    # batch_data = torch.zeros(batch_data.size()).fill_(0)
    # ##############################################################################################


    print('Discriminating image no. {}'.format(n_batch))
    test_result = discriminator(batch_data)
    print('Result: {}'.format(test_result.item()))
    logger.save_image_batch(batch_data, num=n_batch)

    # test_relevance = discriminator.relprop()
    # test_relevance = torch.sum(test_relevance, 1, keepdim=True)
    # test_sensivity = torch.autograd.grad(test_result, batch_data)[0].pw(2)
    #
    # test_relevance = test_relevance[:, :, p:-p, p:-p]
    # batch_data = batch_data[:, :, p:-p, p:-p]
    #
    # logger.save_heatmap_batch(images=batch_data, relevance=test_relevance, probability=test_prob, relu_result=test_result,
    #                           num=n_batch)

    if n_batch > 10:
        break