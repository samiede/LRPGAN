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


# Create Logger instance
outf = '{}/{}'.format('output', os.path.splitext(os.path.basename(sys.argv[0]))[0])
logger = Logger(model_name='LRPGAN', data_name='anime', dir_name=outf, make_fresh=False)
print('Created Logger')


imageSize = 64
batchSize = 20

root_dir = 'dataset/faces'

dataset = datasets.ImageFolder(root=root_dir, transform=transforms.Compose(
    [
        transforms.Resize((imageSize, imageSize)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
))
nc = 3

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize,
                                         shuffle=False, num_workers=2)

mean = torch.zeros(1, nc, imageSize, imageSize)
for n_batch, (batch_data, _) in enumerate(dataloader, 0):
    print('Batch {}/{}'.format(n_batch, len(dataloader)))
    mean += torch.sum(batch_data, 0, keepdim=True)
    mean /= batchSize


logger.save_image_batch(mean, num=None)




