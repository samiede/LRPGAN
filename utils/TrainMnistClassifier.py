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

import subprocess
import errno
import matplotlib.pyplot as plt
import numpy as np


gpu = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

epochs = 12
out_dir = 'dataset/MNIST'
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,)),
    ]
)

traindata = datasets.MNIST(root=out_dir, train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(traindata, batch_size=128, shuffle=True, num_workers=2)

testdata = datasets.MNIST(root=out_dir, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(traindata, batch_size=128, shuffle=True, num_workers=2)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Print(nn.Module):
    def forward(self, input):
        print(input.shape)
        return input


class MNISTNet(nn.Module):

    def __init__(self, ngpu):
        super(MNISTNet, self).__init__()

        self.ngpu = ngpu
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
            Flatten(),
            nn.Linear(in_features=9216, out_features=128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=128, out_features=10),
            nn.Softmax(dim=1),

        )

    def forward(self, x):
        if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.net, x, range(self.ngpu))
        else:
            output = self.net(x)

        return output


net = MNISTNet(1).to(gpu)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adadelta(net.parameters())

for epoch in range(epochs):
    running_loss = 0.0
    for n_batch, (batch_data, labels) in enumerate(trainloader, 0):
        print('Batch {}/{}'.format(n_batch, len(trainloader)))
        optimizer.zero_grad()
        score = net(batch_data.to(gpu))
        loss = criterion(score, labels.to(gpu))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if n_batch % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, n_batch + 1, running_loss / 2000))
            running_loss = 0.0

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy: %d %%' % (
            100 * correct / total))


torch.save(net.state_dict(), './mnist_classifier.pth')
