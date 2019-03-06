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
parser.add_argument('--genfolder_d', required=True)
parser.add_argument('--genfolder_g', required=True)
parser.add_argument('--outf', default='output')
parser.add_argument('--batch_size', default=128)
parser.add_argument('--filename', required=True)
parser.add_argument('--dataset', required=True)
parser.add_argument('--epochs', default=100, type=int)
opt = parser.parse_args()

random.seed(1234)
torch.manual_seed(1234)

# CUDA everything
cudnn.benchmark = True
gpu = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)
outf = '{}/{}/{}'.format('../output', os.path.splitext(os.path.basename(sys.argv[0]))[0], opt.outf)

try:
    os.makedirs(outf)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

text_file = open("{}/{}.txt".format(outf, opt.filename), "w+")
text_file.write('TP, FN, FP, TN \n')
text_file.close()

if opt.dataset == 'mnist':
    out_dir = '../dataset/MNIST'
    dataset = datasets.MNIST(root=out_dir, train=True, download=True,
                             transform=transforms.Compose(
                                 [
                                     transforms.Resize(64),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5,), (0.5,)),
                                 ]
                             ))
    nc = 1

if opt.dataset == 'anime':
    root_dir = '../dataset/faces'
    dataset = datasets.ImageFolder(root=root_dir, transform=transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    ))
    nc = 3
    idx_train = np.arange(0, int(len(dataset) * 0.8) + 1, 1)
    idx_test = np.arange(int(len(dataset) * 0.8) + 1, len(dataset), 1)
    trainingset = torch.utils.data.dataset.Subset(dataset, idx_train)
    test_set = torch.utils.data.dataset.Subset(dataset, idx_test)
    dataset = test_set

ndf = 128
ngf = 128
alpha = 1
ngpu = opt.ngpu
p = 1
batch_size = 64

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=0)


def loadGenerator(gen, dict):
    if torch.__version__ == '0.4.0':
        del dict['net.1.num_batches_tracked']
        del dict['net.4.num_batches_tracked']
        del dict['net.7.num_batches_tracked']
        del dict['net.10.num_batches_tracked']
        del dict['net.13.num_batches_tracked']
    generator.load_state_dict(dict)
    generator.to(gpu)
    return generator


def loadDiscriminator(discr, dict):
    if torch.__version__ == '0.4.0':
        del dict['net.1.bn2.num_batches_tracked']
        del dict['net.2.bn3.num_batches_tracked']
        del dict['net.3.bn4.num_batches_tracked']
        del dict['net.4.bn5.num_batches_tracked']
    discriminator.load_state_dict(dict)
    discriminator.to(gpu)
    return discriminator


# generator = dcgm.GeneratorNetLessCheckerboardUpsample(nc, ngf=128, ngpu=opt.ngpu)
generator = dcgm.GeneratorNetLessCheckerboard(nc=nc, ngf=128, ngpu=opt.ngpu)
discriminator = dcgm.DiscriminatorNetLessCheckerboardToCanonical(nc=nc, alpha=2, ndf=128, ngpu=opt.ngpu)

for epoch in range(int(opt.epochs)):
    print('Evaluating epoch {}'.format(epoch))

    dictpath_g = os.path.join(opt.genfolder_g, 'generator_epoch_{}.pth'.format(epoch))
    dictpath_d = os.path.join(opt.genfolder_d, 'discriminator_epoch_{}.pth'.format(epoch))

    dict_d = torch.load(dictpath_d, map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
    dict_g = torch.load(dictpath_g, map_location='cuda:0' if torch.cuda.is_available() else 'cpu')

    discriminator = loadDiscriminator(discriminator, dict_d)
    generator = loadGenerator(generator, dict_g)

    print('Stabilizing Batch Norm for discriminator')
    discriminator.train()
    for n_batch, (batch_data, _) in enumerate(dataloader, 0):
        batch_data = F.pad(batch_data, (p, p, p, p), mode='replicate').to(gpu)
        _ = discriminator(batch_data)

    print('Stabilizing Batch Norm for generator')
    generator.train()
    for i in range(0, len(dataloader)):
        noise = torch.randn(int(opt.batch_size), 100, 1, 1, device=gpu)
        _ = generator(noise)

    true_positive = 0
    false_negative = 0

    print('Computing on real data')
    discriminator.eval()
    for n_batch, (batch_data, _) in enumerate(dataloader, 0):
        batch_data = F.pad(batch_data, (p, p, p, p), mode='replicate').to(gpu)

        _, probs = discriminator(batch_data)
        true_positive += len(probs[probs > 0.5])
        false_negative += len(probs[probs < 0.5])

    print('TP: {} FN: {}'.format(true_positive, false_negative))

    true_negative = 0
    false_positive = 0

    print('Computing on fake data')
    generator.eval()
    for n_batch, (batch_data, _) in enumerate(dataloader, 0):
        noise = torch.randn(batch_data.size(0), 100, 1, 1, device=gpu)
        images = generator(noise)
        images = F.pad(images, (p, p, p, p), mode='replicate')

        _, probs = discriminator(images)

        true_negative += len(probs[probs < 0.5])
        false_positive += len(probs[probs > 0.5])

    print('TN: {} FP: {}'.format(true_negative, false_positive))

    text_file = open("{}/{}.txt".format(outf, opt.filename), "w+")
    text_file.write(f'{true_positive} {false_negative} {false_positive} {true_negative}\n')
    text_file.close()
