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
import torch.utils.data.dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.distributions as distr
import models._DRAGAN as dcgm
from utils import utils
from utils.utils import Logger
from utils.utils import MidpointNormalize
import subprocess
import errno
import matplotlib.pyplot as plt
import numpy as np
import utils.ciphar10 as ciphar10

# add parameters
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='mnist | anime | custom', required=True, choices=['mnist', 'anime', 'custom', 'ciphar10'])
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--outf', default='output', help='folder to output images and model checkpoints')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--imageSize', type=int, default=128)
parser.add_argument('--eps_init', help='Change epsilon for eps rule after loading state dict', type=float, default=None)
parser.add_argument('--num_images', help='Number of images to be generated/discriminated', type=int)
parser.add_argument('--batchSize', help='number of images processed simultaneously', default=5, type=int)
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--loadG', default=None, help='path to generator')
group.add_argument('--loadD', default=None, help='path to discriminator')
parser.add_argument('--alpha', default=1, type=int)
parser.add_argument('--external', help='load external network', action='store_true')

opt = parser.parse_args()
outf = '{}/{}'.format(opt.outf, os.path.splitext(os.path.basename(sys.argv[0]))[0])
ngpu = int(opt.ngpu)
nz = int(opt.nz)
print(opt)
padding = 2
p = padding

try:
    os.makedirs(outf)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

# Create Logger instance
logger = Logger(model_name='LRPGAN', data_name=opt.dataset, dir_name=outf, make_fresh=False)
print('Created Logger')

# CUDA everything
cudnn.benchmark = True
gpu = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')
print(gpu)

# load datasets
if opt.dataset == 'mnist':
    out_dir = 'dataset/MNIST'
    dataset = datasets.MNIST(root=out_dir, train=True, download=True,
                             transform=transforms.Compose(
                                 [
                                     transforms.Resize(opt.imageSize),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5,), (0.5,)),
                                 ]
                             ))
    nc = 1

elif opt.dataset == 'anime':
    root_dir = 'dataset/faces'
    dataset = datasets.ImageFolder(root=root_dir, transform=transforms.Compose(
        [
            transforms.Resize((opt.imageSize, opt.imageSize)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    ))
    nc = 3

elif opt.dataset == 'custom':
    root_dir = 'dataset/custom'
    dataset = datasets.ImageFolder(root=root_dir, transform=transforms.Compose(
        [
            transforms.Resize((opt.imageSize, opt.imageSize)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    ))
    nc = 1
elif opt.dataset == 'ciphar10':
    out_dir = 'dataset/cifar10'
    dataset = ciphar10.CIFAR10(root=out_dir, download=True, train=True,
                            transform=transforms.Compose([
                                transforms.Resize(opt.imageSize),
                                transforms.Grayscale(num_output_channels=1),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    nc = 1



else:
    pass

assert dataset
assert nc


def eps_init(m):
    classname = m.__class__.__name__
    if classname.find('Eps') != -1:
        m.epsilon = float(opt.eps_init)


# if we want to generate stuff
if opt.loadG and not opt.external:
    generator = dcgm.GeneratorNetLessCheckerboard(nc, ngf=128, ngpu=ngpu)
    # generator = dcgm.Generator(nc, ngf=128, ngpu=ngpu)
    dict = torch.load(opt.loadG, map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.__version__ == '0.4.0':
        del dict['net.1.num_batches_tracked']
        del dict['net.4.num_batches_tracked']
        del dict['net.7.num_batches_tracked']
        del dict['net.10.num_batches_tracked']
        del dict['net.13.num_batches_tracked']
    generator.load_state_dict(dict)
    generator.to(gpu)
    generator.eval()

    noise = torch.randn(opt.num_images, nz, 1, 1, device=gpu)

    images = generator(noise)

    logger.save_image_batch(images, num=None)

# if we want to discriminate stuff
if opt.loadD and not opt.external:

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                             shuffle=False, num_workers=2)

    discriminator = dcgm.DiscriminatorNetLessCheckerboardToCanonical(nc=nc, alpha=opt.alpha, ndf=128, ngpu=ngpu)
    dict = torch.load(opt.loadD, map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
    # TODO for standard DCGAN
    if torch.__version__ == '0.4.0':
        del dict['net.1.bn2.num_batches_tracked']
        del dict['net.2.bn3.num_batches_tracked']
        del dict['net.3.bn4.num_batches_tracked']
        del dict['net.4.bn5.num_batches_tracked']

    discriminator.load_state_dict(dict)
    discriminator.to(gpu)


    def batchPrint(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            print('Batch norm mean weights: {}, mean bias: {}'.format(m.weight.mean(), m.bias.mean()))

    # discriminator.apply(batchPrint)

    if opt.eps_init:
        assert discriminator
        discriminator.apply(eps_init)

    for n_batch, (batch_data, _) in enumerate(dataloader, 0):
        batch_data = batch_data.to(gpu)

        # Perturbing input
        # ##############################################################################################
        # batch_data = batch_data + 0.02 * torch.randn(1, nc, opt.imageSize, opt.imageSize, device=gpu)
        # batch_data = torch.randn(opt.batchSize, nc, opt.imageSize, opt.imageSize, device=gpu)
        # batch_data = utils.pink_noise(1, nc, opt.imageSize, opt.imageSize).to(gpu)
        # batch_data = torch.zeros(batch_data.size()).fill_(1)
        # batch_data = utils.drawBoxes(batch_data.size(0), batch_data.size(1), opt.imageSize, -1, ([[10, 30], [50, 40]], 1))
        # ##############################################################################################

        batch_data = F.pad(batch_data, (p, p, p, p), value=-1)
        batch_data.requires_grad = True

        if opt.num_images and n_batch >= opt.num_images:
            break

        discriminator.passBatchNormParametersToConvolution()
        discriminator.removeBatchNormLayers()
        discriminator.eval()

        if (opt.ngpu > 1):
            discriminator.setngpu(1)

        print('Discriminating image no. {}'.format(n_batch))
        test_result, test_prob = discriminator(batch_data)
        test_relevance = discriminator.relprop()
        test_relevance = torch.sum(test_relevance, 1, keepdim=True)
        # test_sensivity = torch.autograd.grad(test_result, batch_data)[0].pw(2)

        test_relevance = test_relevance[:, :, p:-p, p:-p]
        batch_data = batch_data[:, :, p:-p, p:-p]

        logger.save_heatmap_batch(images=batch_data, relevance=test_relevance, probability=test_prob, relu_result=test_result,
                                  num=n_batch)

if opt.loadD and opt.external:

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                             shuffle=False, num_workers=2)

    discriminator = dcgm.Discriminator(nc=1, ndf=64, ngpu=ngpu)
    dict = torch.load(opt.loadD, map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.__version__ == '0.4.0':
        # del dict['main.1.num_batches_tracked']
        del dict['main.3.num_batches_tracked']
        del dict['main.6.num_batches_tracked']
        del dict['main.9.num_batches_tracked']
    discriminator.load_state_dict(dict)
    discriminator.to(gpu)

    for n_batch, (batch_data, _) in enumerate(dataloader, 0):
        batch_data = batch_data.to(gpu)
        # if batch_data.size(1) == 1:
        #     batch_data = batch_data.repeat(1, 3, 1, 1)
        # batch_data = batch_data + 0.02 * torch.randn(1, nc, opt.imageSize, opt.imageSize, device=gpu)
        # batch_data = torch.randn(1, nc, opt.imageSize, opt.imageSize, device=gpu)
        # batch_data = utils.pink_noise(1, nc, opt.imageSize, opt.imageSize).to(gpu)
        # batch_data = utils.drawBoxes(batch_data.size(0), batch_data.size(1), opt.imageSize, -1, ([[20, 20], [50, 50]], 'uniform'))
        # batch_data = torch.zeros(batch_data.size()).fill_(-1)
        # batch_data = batch_data + 0.02 * torch.randn(1, nc, opt.imageSize, opt.imageSize, device=gpu)

        batch_data = F.pad(batch_data, (p, p, p, p), value=-1)
        batch_data.requires_grad = True

        if opt.num_images  and n_batch >= opt.num_images:
            break
        discriminator.eval()
        test_result = discriminator(batch_data)
        print(test_result.mean().item())

        batch_data = batch_data[:, :, p:-p, p:-p]

        if batch_data.size(1) == 1:
            batch_data = batch_data.repeat(1, 3, 1, 1)

        logger.save_heatmap_batch(images=batch_data, relevance=torch.sum(batch_data, 1, keepdim=True).detach(), probability=test_result, relu_result=test_result,
                                  num=n_batch)


if opt.loadG and opt.external:

    torch.manual_seed(1234)
    np.random.seed(1234)

    generator = dcgm.Generator(nc, ngf=64, ngpu=ngpu)
    dict = torch.load(opt.loadG, map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.__version__ == '0.4.0':
        del dict['main.1.num_batches_tracked']
        del dict['main.4.num_batches_tracked']
        del dict['main.7.num_batches_tracked']
        del dict['main.10.num_batches_tracked']
    generator.load_state_dict(dict)
    generator.to(gpu)
    generator.eval()

    noise = torch.randn(opt.num_images, nz, 1, 1, device=gpu)

    images = generator(noise)

    logger.save_image_batch(images, num=None)
