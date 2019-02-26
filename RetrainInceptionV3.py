import os
import sys
import shutil
import numpy as np
import errno
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from IPython import display
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import colorednoise as cn

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import inception_v3
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from utils.finetuning_torchvision_models_tutorial import train_model


data_dir = 'dataset/PNGMNIST'
dataset = datasets.ImageFolder(root=data_dir,
                         transform=transforms.Compose(
                             [
                                 transforms.RandomResizedCrop(299),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                             ]
                         ))

transforms = {
    'train' : transforms.Compose(
    [
        transforms.RandomResizedCrop(299),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    }
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                      transforms[x]) for x in ['train', 'val']}

dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=0) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
model = inception_v3(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

num_parameters = model.fc.in_features
model.fc = nn.Linear(num_parameters, 10)

optimizer = optim.SGD(list(filter(lambda p: p.requires_grad, model.parameters())), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

criterion = nn.CrossEntropyLoss()

model_ft, hist = train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=15, is_inception=True)

torch.save(model_ft.state_dict(), './inception_v3_mnist.pth')


