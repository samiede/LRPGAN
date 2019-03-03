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

parser = argparse.ArgumentParser()
parser.add_argument('--ngpu', type=int)
parser.add_argument('--genfolder', required=True)
parser.add_argument('--epochs', required=True)
parser.add_argument('--num_images', default=10)
parser.add_argument('--filename')
parser.add_argument('--batch_size', default=2)
opt = parser.parse_args()


# CUDA everything
cudnn.benchmark = True
gpu = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')
print(gpu)

random.seed(1234)
torch.manual_seed(1234)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.conv3 = nn.Conv2d(50, 100, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 100, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 80)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Load Test network
dirpath = os.path.dirname(__file__)
filepath = os.path.join(dirpath, 'mnist_cnn.pt')
net = Net().to(gpu)
net.load_state_dict(torch.load(filepath, map_location='cuda:0' if torch.cuda.is_available() else 'cpu'))
#
#
# def inception_score(images):
#     scores = net(images)
#
#     p_yx = F.softmax(scores, dim=1)
#     p_y = p_yx.mean(0).unsqueeze(0).expand(p_yx.size(0), -1)
#     KL_d = p_yx * (torch.log(p_yx) - torch.log(p_y))
#     # final_score = KL_d.mean()
#     return KL_d


out_dir = '../dataset/MNIST'
dataset = datasets.MNIST(root=out_dir, train=True, download=True,
                         transform=transforms.Compose(
                             [
                                 transforms.Resize(64),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5,), (0.5,)),
                             ]
                         ))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                         shuffle=False, num_workers=2)

for n_batch, (batch_data, _) in enumerate(dataloader, 0):
    scores = net(batch_data)
    print(scores)


def inception_score(scores):
    # scores = net(images)

    p_yx = F.softmax(scores, dim=1)
    p_y = p_yx.mean(0).unsqueeze(0).expand(p_yx.size(0), -1)
    KL_d = p_yx * (torch.log(p_yx) - torch.log(p_y))
    final_score = KL_d.mean()
    return final_score




nc = 1
ndf = 128
alpha = 2
ngpu = opt.ngpu
generator = dcgm.GeneratorNetLessCheckerboard(nc, ndf, ngpu).to(gpu)

all_scores = []
testsetsize = int(opt.num_images)
batch_size = int(opt.batch_size)
num_images = int(opt.num_images)

for epoch in range(int(opt.epochs)):
    print('Evaluating epoch {}'.format(epoch))
    dictpath = os.path.join(opt.genfolder, 'generator_epoch_{}.pth'.format(epoch))
    dict = torch.load(dictpath,  map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
    generator.load_state_dict(dict, strict=False)
    generator.eval()

    it = num_images // batch_size

    noise = torch.randn(batch_size, 100, 1, 1)
    images = generator(noise)

    internal_scores = net(images)
    print('Generating images and calculating score...')

    for iteration in range(1, it):
        noise = torch.randn(batch_size, 100, 1, 1)
        images = generator(noise)
        scores = net(images)
        internal_scores = torch.cat((internal_scores, scores.detach()), dim=0)

    log_inception = inception_score(internal_scores)
    all_scores.append(torch.exp(log_inception.detach()))
    print('Epoch {} has an inception score of {}'.format(epoch, all_scores[epoch]))

print('Best score of the run was {} at epoch {}'. format(max(all_scores).item(), all_scores.index(max(all_scores))))

text_file = open("{}/{}.txt".format('./', opt.filename), "w+")
text_file.write('Best score of the run was {} at epoch {}\n'. format(max(all_scores).item(), all_scores.index(max(all_scores))))
for score in all_scores:
    text_file.write('{}\n'.format(str(score.item())))

text_file.close()



