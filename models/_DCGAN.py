import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data

import modules.ModuleRedefinitions as nnrd


# ######################################## Less Checkerboard pattern VBN ########################################


class GeneratorNetVBN(nn.Module):
    def __init__(self, nc, ngf, ngpu):
        super(GeneratorNetVBN, self).__init__()
        self.ngpu = ngpu
        nz = 100
        self.net = nn.Sequential(

            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0),
            nn.VBN(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ngf * 8, ngf * 8, 3, 1, 1),
            nnrd.VBN(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
            nnrd.VBN(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
            nnrd.VBN(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
            nnrd.VBN(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, x):
        if x.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.net, x, range(self.ngpu))
        else:
            output = self.net(x)
        return output


class DiscriminatorVBN(nn.Module):

    def __init__(self, nc, ndf, alpha, beta, ngpu=1):
        super(DiscriminatorVBN, self).__init__()

        self.ngpu = ngpu
        self.net = nnrd.RelevanceNet(
            nnrd.Layer(
                nnrd.FirstConvolution(nc, ndf, 5, 1, 2),
                nnrd.ReLu(),
            ),
            nnrd.Layer(
                nnrd.NextConvolution(ndf, ndf, 4, '0', 2, 1, alpha=alpha, beta=beta),
                nnrd.BatchNorm2d(ndf),
                nnrd.ReLu(),
            ),
            # state size. (ndf) x 32 x 32
            nnrd.Layer(
                nnrd.NextConvolution(ndf, ndf * 2, 4, '1', 2, 1, alpha=alpha, beta=beta),
                nnrd.BatchNorm2d(ndf * 2),
                nnrd.ReLu(),
            ),
            # state size. (ndf*2) x 16 x 16
            nnrd.Layer(
                nnrd.NextConvolution(ndf * 2, ndf * 4, 4, '2', 2, 1, alpha=alpha, beta=beta),
                nnrd.BatchNorm2d(ndf * 4),
                nnrd.ReLu(),
            ),
            # state size. (ndf*4) x 8 x 8
            nnrd.Layer(
                nnrd.NextConvolution(ndf * 4, ndf * 8, 4, '3', 2, 1, alpha=alpha, beta=beta),
                nnrd.BatchNorm2d(ndf * 8),
                nnrd.ReLu(),
            ),
            # state size. (ndf*8) x 4 x 4
            nnrd.Layer(
                nnrd.NextConvolution(ndf * 8, 1, 4, '4', 1, 0),
                nn.Sigmoid()
            )
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






# ######################################## Less Checkerboard pattern ########################################


class GeneratorNetLessCheckerboard(nn.Module):
    def __init__(self, nc, ngf, ngpu):
        super(GeneratorNetLessCheckerboard, self).__init__()
        self.ngpu = ngpu
        nz = 100
        self.net = nn.Sequential(

            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ngf * 8, ngf * 8, 3, 1, 1),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, x):
        if x.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.net, x, range(self.ngpu))
        else:
            output = self.net(x)
        return output


class DiscriminatorNetLessCheckerboard(nn.Module):

    def __init__(self, nc, ndf, alpha, beta, ngpu=1):
        super(DiscriminatorNetLessCheckerboard, self).__init__()

        self.ngpu = ngpu
        self.net = nnrd.RelevanceNet(
            nnrd.Layer(
                nnrd.FirstConvolution(nc, ndf, 5, 1, 2),
                nnrd.ReLu(),
            ),
            nnrd.Layer(
                nnrd.NextConvolution(ndf, ndf, 4, '0', 2, 1, alpha=alpha, beta=beta),
                nnrd.BatchNorm2d(ndf),
                nnrd.ReLu(),
                nnrd.Dropout(0.3),
            ),
            # state size. (ndf) x 32 x 32
            nnrd.Layer(
                nnrd.NextConvolution(ndf, ndf * 2, 4, '1', 2, 1, alpha=alpha, beta=beta),
                nnrd.BatchNorm2d(ndf * 2),
                nnrd.ReLu(),
                nnrd.Dropout(0.3),

            ),
            # state size. (ndf*2) x 16 x 16
            nnrd.Layer(
                nnrd.NextConvolution(ndf * 2, ndf * 4, 4, '2', 2, 1, alpha=alpha, beta=beta),
                nnrd.BatchNorm2d(ndf * 4),
                nnrd.ReLu(),
                nnrd.Dropout(0.3),
            ),
            # state size. (ndf*4) x 8 x 8
            nnrd.Layer(
                nnrd.NextConvolution(ndf * 4, ndf * 8, 4, '3', 2, 1, alpha=alpha, beta=beta),
                nnrd.BatchNorm2d(ndf * 8),
                nnrd.ReLu(),
                nnrd.Dropout(0.3),
            ),
            # state size. (ndf*8) x 4 x 4
            nnrd.Layer(
                nnrd.NextConvolution(ndf * 8, 1, 4, '4', 1, 0),
                nn.Sigmoid()
            )
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



 # ######################################## Unmodified ########################################


class GeneratorNet(nn.Module):
    def __init__(self, nc, ngf, ngpu):
        super(GeneratorNet, self).__init__()
        nz = 100
        self.ngpu = ngpu
        self.net = nn.Sequential(

            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1),
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

    def __init__(self, nc, ndf, alpha, beta, ngpu=1):
        super(DiscriminatorNet, self).__init__()

        self.ngpu = ngpu
        self.net = nnrd.RelevanceNet(
            nnrd.Layer(
                nnrd.FirstConvolution(nc, ndf, 4, 2, 1),
                nnrd.ReLu(),
            ),
            # state size. (ndf) x 32 x 32
            nnrd.Layer(
                nnrd.NextConvolution(ndf, ndf * 2, 4, '1', 2, 1, alpha=alpha, beta=beta),
                nnrd.BatchNorm2d(ndf * 2),
                nnrd.ReLu(),
            ),
            # state size. (ndf*2) x 16 x 16
            nnrd.Layer(
                nnrd.NextConvolution(ndf * 2, ndf * 4, 4, '2', 2, 1, alpha=alpha, beta=beta),
                nnrd.BatchNorm2d(ndf * 4),
                nnrd.ReLu(),
            ),
            # state size. (ndf*4) x 8 x 8
            nnrd.Layer(
                nnrd.NextConvolution(ndf * 4, ndf * 8, 4, '3', 2, 1, alpha=alpha,  beta=beta),
                nnrd.BatchNorm2d(ndf * 8),
                nnrd.ReLu(),
            ),
            # state size. (ndf*8) x 4 x 4
            nnrd.Layer(
                nnrd.NextConvolution(ndf * 8, 1, 4, '4', 1, 0),
                nn.Sigmoid()
            )
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