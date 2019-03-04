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
parser.add_argument('--ngpu', type=int, default=1)
parser.add_argument('--genfolder', required=True)
parser.add_argument('--epochs', required=True)
parser.add_argument('--num_images', default=9)
parser.add_argument('--outf')
opt = parser.parse_args()

random.seed(1234)
torch.manual_seed(1234)

# CUDA everything
cudnn.benchmark = True
gpu = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)
outf = '{}/{}/{}'.format('../output', os.path.splitext(os.path.basename(sys.argv[0]))[0], opt.outf)
nc = 3

try:
    os.makedirs(outf)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise


def loadGenerator(gen, dict):
    if torch.__version__ == '0.4.0':
        del dict['net.1.num_batches_tracked']
        del dict['net.4.num_batches_tracked']
        del dict['net.7.num_batches_tracked']
        del dict['net.10.num_batches_tracked']
        del dict['net.13.num_batches_tracked']
    generator.load_state_dict(dict)
    generator.to(gpu)
    generator.eval()
    return generator


generator = dcgm.GeneratorNetLessCheckerboard(nc, ngf=128, ngpu=opt.ngpu)
# generator = dcgm.GeneratorNetLessCheckerboardUpsample(nc, ngf=128, ngpu=opt.ngpu)

start = 80
for epoch in range(start, int(opt.epochs)):
    print('Evaluating epoch {}'.format(epoch))
    noise = torch.randn(opt.num_images, 100, 1, 1, device=gpu)
    dictpath = os.path.join(opt.genfolder, 'generator_epoch_{}.pth'.format(epoch))
    try:
        dict = torch.load(dictpath, map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
    except RuntimeError:
        print('Epoch {} checkpoint was corrupted, skipping'.format(epoch))
        continue
    generator = loadGenerator(generator, dict)

    images = generator(noise)

    grid = vutils.make_grid(images.detach(), normalize=True, nrow=int(np.sqrt(opt.num_images)))
    fig = plt.figure(figsize=(64, 64), facecolor='white')
    plt.imshow(np.moveaxis(grid.numpy(), 0, -1))
    ax = plt.axes()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    plt.axis('off')
    fig.savefig('{}/fake_samples_epoch_{}.pdf'.format(outf, epoch), dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()
