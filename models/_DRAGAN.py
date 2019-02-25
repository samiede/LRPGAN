from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F

import modules.ModuleRedefinitions as nnrd


class EDiscriminator(nn.Module):
    def __init__(self, d=128):
        super(EDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, 1, 4, 1, 0)

    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = torch.sigmoid(self.conv5(x))

        return x


# ########################################        Standard LRP DCGAN      ########################################


class LRPGeneratorNet(nn.Module):
    def __init__(self, nc, ngf, ngpu=1):
        super(LRPGeneratorNet, self).__init__()
        self.ngpu = ngpu
        nz = 100
        self.net = nn.Sequential(

            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
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


class LRPDiscriminatorNet(nn.Module):

    def __init__(self, nc, ndf, alpha, ngpu=1):
        super(LRPDiscriminatorNet, self).__init__()

        self.ngpu = ngpu

        self.net = nnrd.RelevanceNetAlternate(
            nnrd.Layer(OrderedDict([
                ('conv2',
                 nnrd.FirstConvolution(in_channels=nc, out_channels=ndf, kernel_size=4, stride=2, padding=0)),
                ('relu2', nnrd.ReLu()),
                ('dropou2', nnrd.Dropout(0.3)),
            ])
            ),
            # state size. (ndf) x 32 x 32
            nnrd.Layer(OrderedDict([
                ('conv3', nnrd.NextConvolution(in_channels=ndf, out_channels=ndf * 2, kernel_size=4, name='1', stride=2,
                                               padding=1, alpha=alpha)),
                ('bn3', nnrd.BatchNorm2d(ndf * 2)),
                ('relu3', nnrd.ReLu()),
                ('dropout3', nnrd.Dropout(0.3)),
            ])

            ),
            # state size. (ndf*2) x 16 x 16
            nnrd.Layer(OrderedDict([
                ('conv4',
                 nnrd.NextConvolution(in_channels=ndf * 2, out_channels=ndf * 4, kernel_size=4, name='2', stride=2,
                                      padding=1, alpha=alpha)),
                ('bn4', nnrd.BatchNorm2d(ndf * 4)),
                ('relu4', nnrd.ReLu()),
                ('dropout4', nnrd.Dropout(0.3)),
            ])
            ),  # state size. (ndf*2) x 16 x 16
            # state size. (ndf*4) x 8 x 8
            nnrd.Layer(OrderedDict([
                ('conv5',
                 nnrd.NextConvolution(in_channels=ndf * 4, out_channels=ndf * 8, kernel_size=4, name='3', stride=2,
                                      padding=1, alpha=alpha)),
                ('bn5', nnrd.BatchNorm2d(ndf * 8)),
                ('relu5', nnrd.ReLu()),
                ('dropout5', nnrd.Dropout(0.3)),
            ])
            ),
        )

        self.lastConvolution = nnrd.LastConvolution(in_channels=ndf * 8, out_channels=1, kernel_size=4, name='4',
                                                    stride=1, padding=0, alpha=alpha)

        self.sigmoid = nn.Sigmoid()
        self.lastReLU = nnrd.ReLu()

    def forward(self, x, flip=True):

        if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.net, x, range(self.ngpu))
        else:
            output = self.net(x)

        if self.training:
            output = self.lastConvolution(output)
            output = self.sigmoid(output)
            return output.view(-1, 1).squeeze(1)

        # relevance propagation
        else:
            probability = self.lastConvolution(output)
            probability = self.sigmoid(probability)

            output = self.lastConvolution(output, flip=flip)
            output = self.lastReLU(output)
            self.relevance = output
            return output.view(-1, 1).squeeze(1), probability.view(-1, 1).squeeze(1)

    def relprop(self, flip=True):
        relevance = self.lastConvolution.relprop(self.relevance, flip)
        return self.net.relprop(relevance)

    def setngpu(self, ngpu):
        self.ngpu = ngpu

    def passBatchNormParametersToConvolution(self):

        i = 1
        for layer in self.net.children():
            names = []
            for name, module in layer.named_children():
                names.append(name)
            if 'conv' + str(i) in names and 'bn' + str(i) in names:
                layer[0].incorporateBatchNorm(layer[1])

            i += 1

    def removeBatchNormLayers(self):
        layers = []

        i = 1
        for layer in self.net.children():
            names = []
            for name, module in layer.named_children():
                names.append(name)
            if 'conv' + str(i) in names and 'bn' + str(i) in names:
                layer = nnrd.Layer(
                    layer[0],
                    layer[2],
                    layer[3]
                )
                layers.append(layer)
            else:
                layers.append(layer)
            i += 1

        self.net = nnrd.RelevanceNetAlternate(
            *layers
        )


# ########################################        Standard DCGAN      ########################################

class Generator(nn.Module):
    def __init__(self, nc, ngf, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        nz = 100
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class Discriminator(nn.Module):
    def __init__(self, nc, ndf, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


# ######################################## Less Checkerboard pattern ########################################

class GeneratorNetLessCheckerboard(nn.Module):
    def __init__(self, nc, ngf, ngpu):
        super(GeneratorNetLessCheckerboard, self).__init__()
        self.ngpu = ngpu
        nz = 100
        self.net = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=True),
            nn.BatchNorm2d(ngf * 16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(ngf * 16, ngf * 8, 3, 1, 1, bias=True),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=True),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, x):
        if x.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.net, x, range(self.ngpu))
        else:
            output = self.net(x)
        return output


class DiscriminatorNetLessCheckerboardToCanonical(nn.Module):

    def __init__(self, nc, ndf, alpha, ngpu=1):
        super(DiscriminatorNetLessCheckerboardToCanonical, self).__init__()

        self.relevance = None
        self.ngpu = ngpu

        self.net = nnrd.RelevanceNetAlternate(
            nnrd.Layer(OrderedDict([
                ('conv1', nnrd.FirstConvolution(in_channels=nc, out_channels=ndf, kernel_size=3, stride=1, padding=0)),
                ('relu1', nnrd.ReLu()),
            ])
            ),
            nnrd.Layer(OrderedDict([
                ('conv2',
                 nnrd.NextConvolution(in_channels=ndf, out_channels=ndf, kernel_size=4, name='0', stride=2, padding=1,
                                      alpha=alpha)),
                ('bn2', nnrd.BatchNorm2d(ndf)),
                ('relu2', nnrd.ReLu()),
                ('dropou2', nnrd.Dropout(0.3)),
            ])
            ),
            # state size. (ndf) x 32 x 32
            nnrd.Layer(OrderedDict([
                ('conv3', nnrd.NextConvolution(in_channels=ndf, out_channels=ndf * 2, kernel_size=4, name='1', stride=2,
                                               padding=1, alpha=alpha)),
                ('bn3', nnrd.BatchNorm2d(ndf * 2)),
                ('relu3', nnrd.ReLu()),
                ('dropout3', nnrd.Dropout(0.3)),
            ])

            ),
            # state size. (ndf*2) x 16 x 16
            nnrd.Layer(OrderedDict([
                ('conv4',
                 nnrd.NextConvolution(in_channels=ndf * 2, out_channels=ndf * 4, kernel_size=4, name='2', stride=2,
                                      padding=1, alpha=alpha)),
                ('bn4', nnrd.BatchNorm2d(ndf * 4)),
                ('relu4', nnrd.ReLu()),
                ('dropout4', nnrd.Dropout(0.3)),
            ])
            ),  # state size. (ndf*2) x 16 x 16
            # state size. (ndf*4) x 8 x 8
            nnrd.Layer(OrderedDict([
                ('conv5',
                 nnrd.NextConvolution(in_channels=ndf * 4, out_channels=ndf * 8, kernel_size=4, name='3', stride=2,
                                         padding=1, alpha=alpha)),
                ('bn5', nnrd.BatchNorm2d(ndf * 8)),
                ('relu5', nnrd.ReLu()),
                ('dropout5', nnrd.Dropout(0.3)),
            ])
            ),
        )

        self.lastConvolution = nnrd.LastConvolution(in_channels=ndf * 8, out_channels=1, kernel_size=4, name='4',
                                                       stride=1, padding=0, alpha=alpha)

        self.sigmoid = nn.Sigmoid()
        self.lastReLU = nnrd.ReLu()

    def forward(self, x, flip=True):

        if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.net, x, range(self.ngpu))
        else:
            output = self.net(x)

        if self.training:
            output = self.lastConvolution(output)
            output = self.sigmoid(output)
            return output.view(-1, 1).squeeze(1)

        # relevance propagation
        else:
            probability = self.lastConvolution(output)
            # print('Before sigmoid: {}'.format(probability.item()))
            probability = self.sigmoid(probability)

            output = self.lastConvolution(output, flip=flip)
            output = self.lastReLU(output)
            self.relevance = output
            return output.view(-1, 1).squeeze(1), probability.view(-1, 1).squeeze(1)

    def relprop(self, flip=True):
        relevance = self.lastConvolution.relprop(self.relevance, flip)
        return self.net.relprop(relevance)

    def setngpu(self, ngpu):
        self.ngpu = ngpu

    def passBatchNormParametersToConvolution(self):

        i = 1
        for layer in self.net.children():
            names = []
            for name, module in layer.named_children():
                names.append(name)
            if 'conv' + str(i) in names and 'bn' + str(i) in names:
                layer[0].incorporateBatchNorm(layer[1])

            i += 1

    def removeBatchNormLayers(self):
        layers = []

        i = 1
        for layer in self.net.children():
            names = []
            for name, module in layer.named_children():
                names.append(name)
            if 'conv' + str(i) in names and 'bn' + str(i) in names:
                layer = nnrd.Layer(
                    layer[0],
                    layer[2],
                    layer[3]
                )
                layers.append(layer)
            else:
                layers.append(layer)
            i += 1

        self.net = nnrd.RelevanceNetAlternate(
            *layers
        )


class DiscriminatorNetLessCheckerboardToCanonicalAB(nn.Module):

    def __init__(self, nc, ndf, alpha, ngpu=1):
        super(DiscriminatorNetLessCheckerboardToCanonicalAB, self).__init__()

        self.relevance = None
        self.ngpu = ngpu

        self.net = nnrd.RelevanceNetAlternate(
            nnrd.Layer(OrderedDict([
                ('conv1', nnrd.FirstConvolution(in_channels=nc, out_channels=ndf, kernel_size=3, stride=1, padding=0)),
                ('relu1', nn.LeakyReLU()),
            ])
            ),
            nnrd.Layer(OrderedDict([
                ('conv2',
                 nnrd.NextConvolution(in_channels=ndf, out_channels=ndf, kernel_size=4, name='0', stride=2, padding=1,
                                      alpha=alpha)),
                ('bn2', nnrd.BatchNorm2d(ndf)),
                ('relu2', nn.LeakyReLU()),
                # ('dropou2', nnrd.Dropout(0.3)),
            ])
            ),
            # state size. (ndf) x 32 x 32
            nnrd.Layer(OrderedDict([
                ('conv3', nnrd.NextConvolution(in_channels=ndf, out_channels=ndf * 2, kernel_size=4, name='1', stride=2,
                                               padding=1, alpha=alpha)),
                ('bn3', nnrd.BatchNorm2d(ndf * 2)),
                ('relu3', nn.LeakyReLU()),
                # ('dropout3', nnrd.Dropout(0.3)),
            ])

            ),
            # state size. (ndf*2) x 16 x 16
            nnrd.Layer(OrderedDict([
                ('conv4',
                 nnrd.NextConvolution(in_channels=ndf * 2, out_channels=ndf * 4, kernel_size=4, name='2', stride=2,
                                      padding=1, alpha=alpha)),
                ('bn4', nnrd.BatchNorm2d(ndf * 4)),
                ('relu4', nn.LeakyReLU()),
                # ('dropout4', nnrd.Dropout(0.3)),
            ])
            ),  # state size. (ndf*2) x 16 x 16
            # state size. (ndf*4) x 8 x 8
            nnrd.Layer(OrderedDict([
                ('conv5',
                 nnrd.NextConvolution(in_channels=ndf * 4, out_channels=ndf * 8, kernel_size=4, name='3', stride=2,
                                      padding=1, alpha=alpha)),
                ('bn5', nnrd.BatchNorm2d(ndf * 8)),
                ('relu5', nn.LeakyReLU()),
                # ('dropout5', nnrd.Dropout(0.3)),
            ])
            ),
        )

        self.lastConvolution = nnrd.LastConvolution(in_channels=ndf * 8, out_channels=1, kernel_size=4, name='4',
                                                    stride=1, padding=0, alpha=alpha)

        self.sigmoid = nn.Sigmoid()
        self.lastReLU = nnrd.ReLu()

    def forward(self, x, flip=True):

        if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.net, x, range(self.ngpu))
        else:
            output = self.net(x)

        if self.training:
            output = self.lastConvolution(output)
            output = self.sigmoid(output)
            return output.view(-1, 1).squeeze(1)

        # relevance propagation
        else:
            probability = self.lastConvolution(output)
            probability = self.sigmoid(probability)

            output = self.lastConvolution(output, flip=flip)
            output = self.lastReLU(output)
            self.relevance = output
            return output.view(-1, 1).squeeze(1), probability.view(-1, 1).squeeze(1)

    def relprop(self, flip=True):
        relevance = self.lastConvolution.relprop(self.relevance, flip)
        return self.net.relprop(relevance)

    def setngpu(self, ngpu):
        self.ngpu = ngpu

    def passBatchNormParametersToConvolution(self):

        i = 1
        for layer in self.net.children():
            names = []
            for name, module in layer.named_children():
                names.append(name)
            if 'conv' + str(i) in names and 'bn' + str(i) in names:
                layer[0].incorporateBatchNorm(layer[1])

            i += 1

    def removeBatchNormLayers(self):
        layers = []

        i = 1
        for layer in self.net.children():
            names = []
            for name, module in layer.named_children():
                names.append(name)
            if 'conv' + str(i) in names and 'bn' + str(i) in names:
                layer = nnrd.Layer(
                    layer[0],
                    layer[2],
                    layer[3]
                )
                layers.append(layer)
            else:
                layers.append(layer)
            i += 1

        self.net = nnrd.RelevanceNetAlternate(
            *layers
        )


# ######################################## Smoothing layer ########################################


class SmoothingLayerDiscriminator(nn.Module):

    def __init__(self, nc, ndf, alpha, ngpu=1):
        super(SmoothingLayerDiscriminator, self).__init__()

        self.relevance = None
        self.ngpu = ngpu

        self.net = nnrd.RelevanceNetAlternate(
            nnrd.Layer(OrderedDict([
                ('conv1', nnrd.FirstConvolution(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=0)),
                ('relu1', nnrd.ReLu()),
            ])
            ),
            nnrd.Layer(OrderedDict([
                ('conv2',
                 nnrd.NextConvolution(in_channels=nc, out_channels=ndf, kernel_size=4, name='0', stride=2, padding=1,
                                      alpha=alpha)),
                ('bn2', nnrd.BatchNorm2d(ndf)),
                ('relu2', nnrd.ReLu()),
                ('dropou2', nnrd.Dropout(0.3)),
            ])
            ),
            # state size. (ndf) x 32 x 32
            nnrd.Layer(OrderedDict([
                ('conv3', nnrd.NextConvolution(in_channels=ndf, out_channels=ndf * 2, kernel_size=4, name='1', stride=2,
                                               padding=1, alpha=alpha)),
                ('bn3', nnrd.BatchNorm2d(ndf * 2)),
                ('relu3', nnrd.ReLu()),
                ('dropout3', nnrd.Dropout(0.3)),
            ])

            ),
            # state size. (ndf*2) x 16 x 16
            nnrd.Layer(OrderedDict([
                ('conv4',
                 nnrd.NextConvolution(in_channels=ndf * 2, out_channels=ndf * 4, kernel_size=4, name='2', stride=2,
                                      padding=1, alpha=alpha)),
                ('bn4', nnrd.BatchNorm2d(ndf * 4)),
                ('relu4', nnrd.ReLu()),
                ('dropout4', nnrd.Dropout(0.3)),
            ])
            ),  # state size. (ndf*2) x 16 x 16
            # state size. (ndf*4) x 8 x 8
            nnrd.Layer(OrderedDict([
                ('conv5',
                 nnrd.NextConvolutionEps(in_channels=ndf * 4, out_channels=ndf * 8, kernel_size=4, name='3', stride=2,
                                         padding=1, epsilon=0.01)),
                ('bn5', nnrd.BatchNorm2d(ndf * 8)),
                ('relu5', nnrd.ReLu()),
                ('dropout5', nnrd.Dropout(0.3)),
            ])
            ),
        )

        self.lastConvolution = nnrd.LastConvolutionEps(in_channels=ndf * 8, out_channels=1, kernel_size=4, name='4',
                                                       stride=1, padding=0, epsilon=0.01)

        self.sigmoid = nn.Sigmoid()
        self.lastReLU = nnrd.ReLu()

        # Do not update weights in smoothing layer
        for parameter in self.net[0][0].parameters():
            parameter.requires_grad = False

    def forward(self, x, flip=True):

        if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.net, x, range(self.ngpu))
        else:
            output = self.net(x)

        if self.training:
            output = self.lastConvolution(output)
            output = self.sigmoid(output)
            return output.view(-1, 1).squeeze(1)

        # relevance propagation
        else:
            probability = self.lastConvolution(output)
            probability = self.sigmoid(probability)

            output = self.lastConvolution(output, flip=flip)
            output = self.lastReLU(output)
            self.relevance = output
            return output.view(-1, 1).squeeze(1), probability.view(-1, 1).squeeze(1)

    def relprop(self, flip=True):
        relevance = self.lastConvolution.relprop(self.relevance, flip)
        return self.net.relprop(relevance)

    def setngpu(self, ngpu):
        self.ngpu = ngpu

    def passBatchNormParametersToConvolution(self):

        i = 1
        for layer in self.net.children():
            names = []
            for name, module in layer.named_children():
                names.append(name)
            if 'conv' + str(i) in names and 'bn' + str(i) in names:
                layer[0].incorporateBatchNorm(layer[1])

            i += 1

    def removeBatchNormLayers(self):
        layers = []

        i = 1
        for layer in self.net.children():
            names = []
            for name, module in layer.named_children():
                names.append(name)
            if 'conv' + str(i) in names and 'bn' + str(i) in names:
                layer = nnrd.Layer(
                    layer[0],
                    layer[2],
                    layer[3]
                )
                layers.append(layer)
            else:
                layers.append(layer)
            i += 1

        self.net = nnrd.RelevanceNetAlternate(
            *layers
        )
