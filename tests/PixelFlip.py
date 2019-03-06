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
torch.set_default_dtype(torch.float32)
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

# add parameters
parser = argparse.ArgumentParser()
parser.add_argument('--loadD', default='', help='path to discriminator')
parser.add_argument('--ngpu', default=1, type=int)
parser.add_argument('--outf', default='output', help='folder to output images and model checkpoints')
parser.add_argument('--alpha', default=2)
parser.add_argument('--num_images', default=1000, type=int)
parser.add_argument('--dataset', help='mnist | anime | custom', required=True, choices=['mnist', 'anime', 'custom', 'ciphar10'])
parser.add_argument('--k', help='Number of pixels to flip', type=int, default=None)
parser.add_argument('--p', help='Percent of pixels to flip', type=int)
parser.add_argument('--highest', action='store_true')
parser.add_argument('--filename')
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


def eps_init(m):
    classname = m.__class__.__name__
    if classname.find('Eps') != -1:
        m.epsilon = 1e-9



# Create Logger instance
logger = Logger(model_name='LRPGAN', data_name=opt.dataset, dir_name=outf, make_fresh=False)
print('Created Logger')

p = 1
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64,
                                         shuffle=False, num_workers=0)

discriminator = dcgm.DiscriminatorNetLessCheckerboardToCanonical(nc=nc, alpha=opt.alpha, ndf=128, ngpu=ngpu)
dict = torch.load(opt.loadD, map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
discriminator.load_state_dict(dict, strict=False)
discriminator = discriminator.to(gpu)
if torch.cuda.is_available():
    discriminator.cuda()

discriminator.apply(eps_init)

# print('Stabilizing batch norm')
# for n_batch, (batch_data, _) in enumerate(dataloader, 0):
#     batch_data = batch_data.to(gpu)
#     batch_data = F.pad(batch_data, (p, p, p, p), value=-1)
#     _ = discriminator(batch_data)
#
print('Stabilizing batch norm')
dataloader_iter = iter(dataloader)
for i in range(0, 30):
    print(i)
    batch_data, _ = next(dataloader_iter)
    batch_data = batch_data.to(gpu)
    batch_data = F.pad(batch_data, (p, p, p, p), mode='replicate')
    _ = discriminator(batch_data)

discriminator.passBatchNormParametersToConvolution()
discriminator.removeBatchNormLayers()
discriminator.eval()

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                         shuffle=False, num_workers=0)

flip = True
all_before_scores = []
all_after_scores = []
highest = opt.highest
if ngpu > 1:
    discriminator.setngpu(1)

for percent in range(0, opt.p):
    before_score = []
    after_score = []
    percentile = percent / 100
    k = int(64 * 64 * percentile)
    for n_batch, (batch_data, _) in enumerate(dataloader, 0):
        print('Image {}'.format(n_batch + 1))
        batch_data = batch_data.to(gpu)
        print_obj = n_batch == opt.num_images

        batch_data = F.pad(batch_data, (p, p, p, p), mode='replicate')
        batch_data.requires_grad = True

        test_result, test_prob = discriminator(batch_data, flip=flip)
        before_score.append(test_prob.detach().item())

        if test_prob.detach().item() > 0.5:
            test_result, test_prob = discriminator(batch_data, flip=False)
            test_relevance = discriminator.relprop(flip=False)
        else:
            test_relevance = discriminator.relprop(flip=flip)

        test_relevance = torch.sum(test_relevance, 1, keepdim=True)

        test_relevance = test_relevance[:, :, p:-p, p:-p]
        batch_data = batch_data[:, :, p:-p, p:-p]

        if print_obj:
            logger.save_heatmap_batch(images=batch_data, relevance=test_relevance, probability=test_prob, relu_result=test_result,
                                      num='{}_{}_{}'.format('before', percentile, highest))

        if k > 0:
            indices = get_indices(test_relevance, k=k, highest=highest)

            flipped_image = flip_pixels(batch_data, indices)
        else:
            flipped_image = batch_data

        flipped_image = F.pad(flipped_image, (p, p, p, p), mode='replicate')

        test_result, test_prob = discriminator(flipped_image, flip=flip)
        after_score.append(test_prob.detach().item())

        if test_prob.detach().item() > 0.5:
            test_result, test_prob = discriminator(flipped_image, flip=False)
            test_relevance = discriminator.relprop(flip=False)
        else:
            test_relevance = discriminator.relprop(flip=flip)

        test_relevance = torch.sum(test_relevance, 1, keepdim=True)

        test_relevance = test_relevance[:, :, p:-p, p:-p]
        flipped_image = flipped_image[:, :, p:-p, p:-p]

        if print_obj:
            logger.save_heatmap_batch(images=flipped_image, relevance=test_relevance, probability=test_prob, relu_result=test_result,
                                      num='{}_{}_{}'.format('after', percentile, highest))

        if n_batch >= opt.num_images:
            break

    before_score_mean = np.mean(before_score)
    after_score_mean = np.mean(after_score)

    if highest:
        print('Before flipping {} highest pixels: {}'.format(percentile, before_score_mean))
        print('After flipping {} highest pixels: {}'.format(percentile, after_score_mean))
        print('Change of {}'.format(100 - (before_score_mean / after_score_mean * 100)))
    else:
        print('Before flipping {} lowest pixels: {}'.format(percentile, before_score_mean))
        print('After flipping {} lowest pixels: {}'.format(percentile, after_score_mean))
        print('Change of {}%'.format(100 - (before_score_mean / after_score_mean * 100)))

    all_before_scores.append(before_score_mean)
    all_after_scores.append(after_score_mean)

    text_file = open("{}/{}_highest_{}.txt".format(outf, opt.filename, opt.highest), "a+")
    text_file.write(f'{k} {before_score_mean} {after_score_mean}\n')
    text_file.close()
