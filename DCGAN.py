# imports
from __future__ import print_function
import argparse
import os
import sys
sys.path.append('..')
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

# add parameters
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--dataset', help='mnist', default='mnist')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64, help='number of generator filters in first layer')
parser.add_argument('--ndf', type=int, default=64, help='number of discriminator filters in first layer')
parser.add_argument('--epochs', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--outf', default='../output', help='folder to output images and model checkpoints')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--imageSize', type=int, default=64)
parser.add_argument('--loadG', default='', help='path to generator (to continue training')
parser.add_argument('--loadD', default='', help='path to discriminator (to continue training')

opt = parser.parse_args()
outf = '{}/{}'.format(opt.outf, os.path.splitext(os.path.basename(sys.argv[0]))[0])
ngpu = int(opt.ngpu)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nz = int(opt.nz)
print(opt)

try:
    os.makedirs(outf)
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
    out_dir = '../dataset/MNIST'
    dataset = datasets.MNIST(root=out_dir, train=True, download=True,
                             transform=transforms.Compose(
                                 [
                                     transforms.Resize(opt.imageSize),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5,), (0.5,)),
                                 ]
                             ))
    nc = 1
else:
    pass

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=2)


# misc. helper functions

def discriminator_target(size):
    """
    Tensor containing soft labels, with shape = size
    """
    # noinspection PyUnresolvedReferences
    return torch.Tensor(size).uniform_(0.8, 1.0)


def generator_target(size):
    """
    Tensor containing zeroes, with shape = size
    :param size: shape of vector
    :return: zeros tensor
    """
    # noinspection PyUnresolvedReferences
    return torch.Tensor(size).zero_()
    # return torch.Tensor(size).uniform_(0, 0.3)


# init networks

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class GeneratorNet(nn.Module):
    def __init__(self, ngpu):
        super(GeneratorNet, self).__init__()
        self.ngpu = ngpu
        self.net = nn.Sequential(

            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, x):
        if x.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.net, x, range(self.ngpu))
        else:
            output = self.net(x)
        return output


class DiscriminatorNet(nn.Module):

    def __init__(self, ngpu=1):
        super(DiscriminatorNet, self).__init__()

        self.ngpu = ngpu
        self.net = nnrd.RelevanceNet(
            nnrd.FirstConvolution(nc, ndf, 4, 2, 1),
            nnrd.ReLu(),
            # state size. (ndf) x 32 x 32
            nnrd.NextConvolution(ndf, ndf * 2, 4, 2, 1),
            nnrd.BatchNorm2d(ndf * 2),
            nnrd.ReLu(),
            # state size. (ndf*2) x 16 x 16
            nnrd.NextConvolution(ndf * 2, ndf * 4, 4, 2, 1),
            nnrd.BatchNorm2d(ndf * 4),
            nnrd.ReLu(),
            # state size. (ndf*4) x 8 x 8
            nnrd.NextConvolution(ndf * 4, ndf * 8, 4, 2, 1),
            nnrd.BatchNorm2d(ndf * 8),
            nnrd.ReLu(),
            # state size. (ndf*8) x 4 x 4
            nnrd.NextConvolution(ndf * 8, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):

        if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.net, x, range(self.ngpu))
        else:
            output = self.net(x)
        return output.view(-1, 1).squeeze(1)

    def relprop(self):
        return self.net.relprop()

    def setngpu(self, ngpu):
        self.ngpu = ngpu


generator = GeneratorNet(ngpu).to(gpu)
generator.apply(weights_init)
if opt.loadG != '':
    generator.load_state_dict(torch.load(opt.loadG))

discriminator = DiscriminatorNet(ngpu).to(gpu)
discriminator.apply(weights_init)
if opt.loadD != '':
    discriminator.load_state_dict(torch.load(opt.loadG))

# init optimizer + loss

d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

loss = nn.BCELoss()

# init fixed noise

fixed_noise = torch.randn(1, nz, 1, 1, device=gpu)

# Create Logger instance
logger = Logger(model_name='LRPGAN', data_name=opt.dataset, dir_name=outf)
print('Created Logger')

# training

for epoch in range(opt.epochs):
    for n_batch, (batch_data, _) in enumerate(dataloader, 0):
        print('Batch', n_batch, end='\r')
        batch_size = batch_data.size(0)

        ############################
        # (1) Update Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        discriminator.zero_grad()
        real_data = batch_data.to(gpu)
        label_real = discriminator_target(batch_size).to(gpu)

        prediction_real = discriminator(real_data)
        d_err_real = loss(prediction_real, label_real)
        d_err_real.backward()
        d_real = prediction_real.mean().item()

        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=gpu)
        fake = generator(noise)
        label_fake = generator_target(batch_size).to(gpu)
        prediction_fake = discriminator(fake.detach())
        d_err_fake = loss(prediction_fake, label_fake)
        d_err_fake.backward()
        d_fake_1 = prediction_fake.mean().item()
        d_error_total = d_err_real + d_err_fake
        d_optimizer.step()

        ############################
        # (2) Update Generator: maximize log(D(G(z)))
        ###########################
        generator.zero_grad()
        prediction_fake_g = discriminator(fake)
        g_err = loss(prediction_fake_g, label_real)
        g_err.backward()
        d_fake_2 = prediction_fake_g.mean().item()
        g_optimizer.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.epochs, n_batch, len(dataloader),
                 d_error_total.item(), g_err.item(), d_real, d_fake_1, d_fake_2))

        if n_batch % 100 == 0:
            # generate fake with fixed noise
            test_fake = generator(fixed_noise)

            # set ngpu to one, so relevance propagation works
            if (opt.ngpu > 1):
                discriminator.setngpu(1)

            # eval needs to be set so batch norm works with batch size of 1
            discriminator.eval()
            test_result = discriminator(test_fake)
            test_relevance = discriminator.relprop()

            # set ngpu back to opt.ngpu
            if (opt.ngpu > 1):
                discriminator.setngpu(opt.ngpu)

            # Add up relevance of all color channels
            test_relevance = torch.sum(test_relevance, 1, keepdim=True)

            logger.log_images(
                test_fake.detach(), test_relevance, 1,
                epoch, n_batch, len(dataloader)
            )

            status = logger.display_status(epoch, opt.epochs, n_batch, len(dataloader), d_error_total, g_err,
                                           prediction_real, prediction_fake)

    # do checkpointing
    torch.save(discriminator.state_dict(), '%s/generator_epoch_%d.pth' % (outf, epoch))
    torch.save(generator.state_dict(), '%s/discriminator_epoch_%d.pth' % (outf, epoch))