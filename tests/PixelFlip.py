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
import utils.utils as utils
from utils.utils import Logger
from utils.utils import MidpointNormalize
import subprocess
import errno
import matplotlib.pyplot as plt
import numpy as np
import utils.ciphar10 as ciphar10

gpu = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# add parameters
parser = argparse.ArgumentParser()
parser.add_argument('--loadD', default='', help='path to discriminator')
parser.add_argument('--ngpu', default=1, type=int)
parser.add_argument('--outf', default='output', help='folder to output images and model checkpoints')
parser.add_argument('--alpha', default=1)
parser.add_argument('--dataset', help='mnist | anime | custom', required=True, choices=['mnist', 'anime', 'custom', 'ciphar10'])
opt = parser.parse_args()
ngpu = int(opt.ngpu)
opt.imageSize = 64
outf = '{}/{}'.format(opt.outf, os.path.splitext(os.path.basename(sys.argv[0]))[0])

try:
    os.makedirs(outf, exist_ok=True)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

# load datasets
if opt.dataset == 'mnist':
    out_dir = '../dataset/MNIST'
    dataset = datasets.MNIST(root=out_dir, train=True, download=True,
                             transform=transforms.Compose(
                                 [
                                     transforms.Resize(opt.imageSize),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5,), (0.5,)),
                                 ]
                             ))
    # nc = 1

elif opt.dataset == 'anime':
    root_dir = '../dataset/faces'
    dataset = datasets.ImageFolder(root=root_dir, transform=transforms.Compose(
        [
            transforms.Resize((opt.imageSize, opt.imageSize)),
            # transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    ))
    # nc = 3

elif opt.dataset == 'custom':
    root_dir = '../dataset/custom'
    dataset = datasets.ImageFolder(root=root_dir, transform=transforms.Compose(
        [
            transforms.Resize((opt.imageSize, opt.imageSize)),
            # transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    ))
    # nc = 3
elif opt.dataset == 'ciphar10':
    out_dir = '../dataset/cifar10'
    dataset = ciphar10.CIFAR10(root=out_dir, download=True, train=True,
                               transform=transforms.Compose([
                                   transforms.Resize(opt.imageSize),
                                   # transforms.Grayscale(num_output_channels=1),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

else:
    pass

nc = 1


def get_indices(relevance, k, highest):
    idx = []
    dim = relevance.size(-1)
    unraveled = relevance.reshape(relevance.numel())
    values, indices = torch.topk(unraveled, k=k, largest=highest)
    for index in indices:
        index = index.item()
        dim0 = index // dim
        dim1 = index % dim
        idx.append(torch.Tensor([dim0, dim1]).long())

    return idx


def flip_pixels(batch_data, indices):
    data = batch_data.clone()
    for idx in indices:
        data[:, :, idx[0], idx[1]] *= -1
    return data


# Create Logger instance
logger = Logger(model_name='LRPGAN', data_name=opt.dataset, dir_name=outf, make_fresh=False)
print('Created Logger')

p = 1
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                         shuffle=False, num_workers=2)

discriminator = dcgm.DiscriminatorNetLessCheckerboardToCanonical(nc=nc, alpha=opt.alpha, ndf=64, ngpu=ngpu)
dict = torch.load(opt.loadD, map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
discriminator.load_state_dict(dict, strict=False)
discriminator.to(gpu)

for n_batch, (batch_data, _) in enumerate(dataloader, 0):
    batch_data = batch_data.to(gpu)
    # batch_data = torch.zeros(batch_data.size()).fill_(1)
    # batch_data = utils.drawBoxes(batch_data, True, -1, ([[10, 30], [50, 40]], 1))

    batch_data = F.pad(batch_data, (p, p, p, p), value=-1)
    batch_data.requires_grad = True

    discriminator.passBatchNormParametersToConvolution()
    discriminator.removeBatchNormLayers()
    discriminator.eval()

    flip = True
    test_result, test_prob = discriminator(batch_data, flip=flip)

    test_relevance = discriminator.relprop(flip=flip)
    test_relevance = torch.sum(test_relevance, 1, keepdim=True)

    test_relevance = test_relevance[:, :, p:-p, p:-p]
    batch_data = batch_data[:, :, p:-p, p:-p]

    logger.save_heatmap_batch(images=batch_data, relevance=test_relevance, probability=test_prob, relu_result=test_result,
                              num=n_batch)

    indices = get_indices(test_relevance, k=5, highest=False)

    flipped_image = flip_pixels(batch_data, indices)

    flipped_image = F.pad(flipped_image, (p, p, p, p), value=-1)

    test_result, test_prob = discriminator(flipped_image, flip=flip)

    test_relevance = discriminator.relprop(flip=flip)
    test_relevance = torch.sum(test_relevance, 1, keepdim=True)

    test_relevance = test_relevance[:, :, p:-p, p:-p]
    flipped_image = flipped_image[:, :, p:-p, p:-p]

    logger.save_heatmap_batch(images=flipped_image, relevance=test_relevance, probability=test_prob, relu_result=test_result,
                              num=n_batch + 1)

    break
