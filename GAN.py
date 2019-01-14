# imports
from __future__ import print_function
import argparse
import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import modules.ModuleRedefinitions as nnrd
from utils.utils import Logger
import subprocess

# add parameters
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--dataset', help='mnist', default='mnist')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64, help='number of generator filters in first layer')
parser.add_argument('--ndf', type=int, default=64, help='number of discriminator filters in first layer')
parser.add_argument('--epochs', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--outf', default='output', help='folder to output images and model checkpoints')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--imageSize', type=int, default=64)
parser.add_argument('--loadG', default='', help='path to generator (to continue training')
parser.add_argument('--loadD', default='', help='path to discriminator (to continue training')

opt = parser.parse_args()
outf = '{}/{}'.format(opt.outf, os.path.splitext(os.path.basename(sys.argv[0]))[0])
checkpointdir = '{}/{}'.format(outf, 'checkpoints')
ngpu = int(opt.ngpu)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nz = int(opt.nz)
print(opt)

try:
    os.makedirs(outf)
except OSError:
    pass

try:
    os.makedirs(checkpointdir)
except OSError:
    pass

# CUDA everything
cudnn.benchmark = True
gpu = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')
print(gpu)

# load datasets
if opt.dataset == 'mnist':
    out_dir = 'dataset/MNIST'
    dataset = datasets.MNIST(root=out_dir, train=True, download=True,
                             transform=transforms.Compose(
                                 [
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, .5, .5), (0.5, 0.5, 0.5)),
                                 ]
                             ))
else:
    pass

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=2)


def images_to_vectors(images):
    images = images.view(images.size(0), 784)
    images.requires_grad = True
    return images

def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28)


def noise(size):
    """

    Generates a 1-d vector of gaussian sampled random values
    """
    # noinspection PyUnresolvedReferences
    z = torch.randn((size, 100), requires_grad=True)
    return z


def discriminator_target(size):
    """
    Tensor containing ones, with shape = size
    """
    # noinspection PyUnresolvedReferences
    return torch.ones(size, 1)


def generator_target(size):
    """
    Tensor containing zeroes, with shape = size
    :param size: shape of vector
    :return: zeros tensor
    """
    # noinspection PyUnresolvedReferences
    return torch.zeros(size, 1)


class DiscriminatorNet(nn.Module):
    """
    A three hidden-layer discriminative neural network
    """

    def __init__(self, ngpu):
        super(DiscriminatorNet, self).__init__()
        self.ngpu = ngpu

        # All attributes are automatically assigned to the modules parameter list
        n_features = 784
        n_out = 1
        self.net = nnrd.RelevanceNet(
            nnrd.Layer(
                nnrd.FirstLinear(n_features, 1024),
                nnrd.ReLu(),
                nnrd.Dropout(0.3)
            ),
            nnrd.Layer(
                nnrd.NextLinear(1024, 1024),
                nnrd.ReLu(),
                nnrd.Dropout(0.3)
            ),
            nnrd.Layer(
                nnrd.NextLinear(1024, 512),
                nnrd.ReLu(),
                nnrd.Dropout(0.3),
            ),
            nnrd.Layer(
                nnrd.NextLinear(512, 512),
                nnrd.ReLu(),
                nnrd.Dropout(0.3)
            ),
            nnrd.Layer(
                nnrd.NextLinear(512, 256),
                nnrd.ReLu(),
                nnrd.Dropout(0.3),
            ),
            nnrd.Layer(
                nnrd.NextLinear(256, n_out),
                nn.Sigmoid()
            )

        )

    def forward(self, x):
        if x.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.net, x, range(self.ngpu))
        else:
            output = self.net(x)
        return output

    def relprop(self):
        return self.net.relprop()

    def setngpu(self, ngpu):
        self.ngpu = ngpu


class GeneratorNet(nn.Module):
    """
    A three hidden-layer generative neural network
    """

    def __init__(self, ngpu):
        super(GeneratorNet, self).__init__()
        n_features = 100
        n_out = 784
        self.ngpu = ngpu

        self.net = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),

            nn.Linear(1024, n_out),
            nn.Tanh()
        )


    def forward(self, x):
        if x.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.net, x, range(self.ngpu))
        else:
            output = self.net(x)
        return output


# Network Functions

def train_discriminator(optimizer, real_data_, fake_data_):
    N = real_data_.size(0)
    # Reset gradients
    optimizer.zero_grad()

    # 1.1 Train on Real Data
    predictions_real = discriminator(real_data_)
    # Calculate error and backpropagate
    error_real = loss(predictions_real, discriminator_target(N))
    error_real.backward()

    # 1.2 Train on Fake Data
    predictions_fake = discriminator(fake_data_)
    # Calculate Error and backpropagate
    error_fake = loss(predictions_fake, generator_target(N))
    error_fake.backward()

    # 1.3 update weights with gradients
    optimizer.step()

    # Return error and predictions
    return error_fake + error_real, predictions_real, predictions_fake


def train_generator(optimizer, fake_data_):
    N = fake_data_.size(0)

    # Reset gradients
    optimizer.zero_grad()

    # Sample noise and generate fake data
    prediction = discriminator(fake_data_)

    # Calculate error and backpropagate
    error = loss(prediction, discriminator_target(N))
    error.backward()

    # Update weights with gradient
    optimizer.step()

    return error


# Load Data


# Create Loader so we can iterate over the data
# Num Batches
num_batches = len(dataloader)

discriminator = DiscriminatorNet(opt.ngpu).to(gpu)
generator = GeneratorNet(opt.ngpu).to(gpu)

g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)

loss = nn.BCELoss().to(gpu)

num_test_samples = 1
test_noise = noise(num_test_samples)

# Create Logger instance
logger = Logger(model_name='LRPGAN', data_name=opt.dataset, dir_name=outf)

# Training Epochs

d_steps = 1

num_epochs = 200

for epoch in range(num_epochs):
    for n_batch, (real_batch, _) in enumerate(dataloader):
        print('Batch', n_batch, end='\r')

        N = real_batch.size(0)

        # 1. Train Discriminator
        real_data = images_to_vectors(real_batch).to(gpu)

        # Generate fake data and detach
        # (so gradients are not calculated for generator)
        fake_data = generator(noise(N)).to(gpu).detach()

        # Train Discriminator
        d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer, real_data, fake_data)

        # 2. Train Generator

        # Generate fake data
        fake_data = generator(noise(N)).to(gpu)

        # Train Generator
        g_error = train_generator(g_optimizer, fake_data)

        # Log Batch error
        logger.log(d_error, g_error, epoch, n_batch, num_batches)

        # Display Progress every few batches
        if n_batch % 100 == 0:

            # generate fake with fixed noise
            test_fake = generator(test_noise)

            # set ngpu to one, so relevance propagation works
            if (opt.ngpu > 1):
                discriminator.setngpu(1)

            # eval needs to be set so batch norm works with batch size of 1
            # discriminator.eval()
            test_result = discriminator(test_fake)
            # test_relevance = discriminator.relprop()
            #
            # test_relevance = vectors_to_images(test_relevance)
            test_fake = vectors_to_images(test_fake)

            # set ngpu back to opt.ngpu
            if (opt.ngpu > 1):
                discriminator.setngpu(opt.ngpu)
            # discriminator.train()

            # Add up relevance of all color channels
            # test_relevance = torch.sum(test_relevance, 1, keepdim=True)


            logger.log_images(
                test_fake.detach(), test_fake.detach(), 1,
                epoch, n_batch, len(dataloader)
            )

            # show images inline
            subprocess.call([os.path.expanduser('~/.iterm2/imgcat'),
                             outf + '/mnist/hori_epoch_' + str(epoch) + '_batch_' + str(n_batch) + '.png'])

            # Display status Logs
            logger.display_status(
                epoch, num_epochs, n_batch, num_batches,
                d_error, g_error, d_pred_real, d_pred_fake
            )

    # do checkpointing
    torch.save(discriminator.state_dict(), '%s/generator.pth' % checkpointdir)
    torch.save(generator.state_dict(), '%s/discriminator.pth' % checkpointdir)