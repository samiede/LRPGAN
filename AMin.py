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

nc = 3
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

discriminator.passBatchNormParametersToConvolution()
discriminator.removeBatchNormLayers()
discriminator.eval()

original_data = torch.randn(1, 3, opt.imageSize, opt.imageSize, device=gpu)
original_data = torch.zeros(original_data.size()).fill_(1)
original_data = F.pad(original_data, (p, p, p, p), value=-1)
original_data.requires_grad = True
score = torch.zeros(1)
criterion = nn.BCELoss()

i = 0
while score.item() < 0.95:
    result, prob = discriminator(original_data)
    score = prob
    test_relevance = discriminator.relprop()
    test_relevance = torch.sum(test_relevance, 1, keepdim=True)
    test_relevance = test_relevance[:, :, p:-p, p:-p]
    print_data = original_data[:, :, p:-p, p:-p]

    logger.save_heatmap_batch(images=print_data, relevance=test_relevance, probability=prob, relu_result=result,
                              num=i)

    loss = criterion(prob, torch.zeros(1))
    gradient = torch.autograd.grad(loss, original_data)[0]
    nu = torch.sign(gradient)
    original_data = original_data + 0.007 * nu
    logger.save_image_batch(gradient, num=i)
    print('Score: {}, under 0.95: {}'.format(score.item(), score.item() < 0.95))

    i += 1


