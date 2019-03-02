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
parser.add_argument('--loadD', required=True)
parser.add_argument('--loadG', required=True)
parser.add_argument('--outf', default='output')
parser.add_argument('--batch_size', default=128)
parser.add_argument('--filename', required=True)
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

root_dir = '../dataset/faces'
dataset = datasets.ImageFolder(root=root_dir, transform=transforms.Compose(
    [
        transforms.Resize((64, 64)),
        # transforms.Grayscale(num_output_channels=1),
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

dataloader = torch.utils.data.DataLoader(dataset, batch_size=128,
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
    generator.eval()
    return generator


def loadDiscriminator(discr, dict):
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
    return discriminator


generator = dcgm.GeneratorNetLessCheckerboard(nc, ngf=128, ngpu=opt.ngpu)
discriminator = dcgm.DiscriminatorNetLessCheckerboardToCanonical(nc=nc, alpha=2, ndf=128, ngpu=opt.ngpu)

dict_d = torch.load(opt.loadD, map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
dict_g = torch.load(opt.loadG, map_location='cuda:0' if torch.cuda.is_available() else 'cpu')

discriminator = loadDiscriminator(discriminator, dict_d)
generator = loadGenerator(generator, dict_g)

true_positive = 0
false_negative = 0


p = 1
print('Computing on real data')
for n_batch, (batch_data, _) in enumerate(dataloader, 0):
    batch_data = F.pad(batch_data, (p, p, p, p), mode='replicate')

    relu, probs = discriminator(batch_data)
    true_positive += len(probs[probs > 0.5])
    false_negative += len(probs[probs < 0.5])

print('TP: {} FP: {}'.format(true_positive, false_negative))

iterations = len(dataset) / int(opt.batch_size)
full_it = int(iterations)
remainder = iterations - full_it

true_negative = 0
false_positive = 0

print('Computing on fake data')
for i in range(full_it):
    noise = torch.randn(int(opt.batch_size), 100, 1, 1, device=gpu)
    images = generator(noise)
    images = F.pad(images, (p, p, p, p), mode='replicate')

    relu, probs = discriminator(images)

    true_negative += len(probs[probs < 0.5])
    false_positive += len(probs[probs > 0.5])

print('FP: {} TN: {}'.format(true_negative, false_positive))

noise = torch.randn(int(remainder * int(opt.batch_size)), 100, 1, 1, device=gpu)
images = generator(noise)
images = F.pad(images, (p, p, p, p), mode='replicate')

scores = discriminator(images)

true_negative += len(probs[probs < 0.5])
false_positive += len(probs[probs > 0.5])

text_file = open("{}/{}.txt".format(outf, opt.filename), "w+")
text_file.write('__________|____ positive ____|____ negative ____|\n')
text_file.write('_positive_|______ {} ______|______ {} ________|\n'.format(true_positive, false_negative))
text_file.write('_negative_|______ {} ______|______ {} ________|  '.format(false_positive, true_negative))
text_file.close()
