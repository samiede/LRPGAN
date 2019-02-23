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
parser.add_argument('--loadG', default=None, help='path to generator')
parser.add_argument('--loadD', default=None, help='path to discriminator')
parser.add_argument('--alpha', default=1, type=int)
parser.add_argument('--external', help='load external network', action='store_true')

opt = parser.parse_args()
outf = '{}/{}'.format(opt.outf, os.path.splitext(os.path.basename(sys.argv[0]))[0])
ngpu = int(opt.ngpu)
nz = int(opt.nz)
print(opt)
padding = 2
p = padding

# CUDA everything
cudnn.benchmark = True
gpu = torch.device('cuda:{}'.format(opt.cuda) if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')
print(gpu)

try:
    os.makedirs(outf)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

# Create Logger instance
logger = Logger(model_name='LRPGAN', data_name=opt.dataset, dir_name=outf, make_fresh=False)
print('Created Logger')


torch.manual_seed(1234)
np.random.seed(1234)
nc = 1

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
# logger.save_image_batch(images, num=None)

root_dir = 'dataset/custom'
dataset = datasets.ImageFolder(root=root_dir, transform=transforms.Compose(
    [
        transforms.Resize((opt.imageSize, opt.imageSize)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
))

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
discriminator.eval()

results_generated = discriminator(images)

for n_batch, (batch_data, _) in enumerate(dataloader, 0):
    results_loaded = discriminator(batch_data)

# logger.save_image_batch(batch_data, num=None)

# logger.save_heatmap_batch(images=batch_data, relevance=images, probability=results_loaded, relu_result=results_generated,
#                           num=n_batch)

print(results_generated == results_loaded)

