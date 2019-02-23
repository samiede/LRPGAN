from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data

import modules.ModuleRedefinitions as nnrd


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



# ########################################      Resnet Generator     ########################################

class ResnetGenerator(nn.Module):
    def __init__(self, nc, nz, ngpu):
        super(ResnetGenerator, self).__init__()
        self.dense_1 = nn.Linear(nz, 64 * 16 * 16)
        self.bn_1 = nn.BatchNorm2d(64)
        self.relu_1 = nn.PReLU()
        self.residual_layer = self.make_residual_layers(block_size=16, kernel_size=3)  # outn=64
        self.conv_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn_2 = nn.BatchNorm2d(64)
        self.relu_2 = nn.PReLU()
        self.pixelshuffle_layer = self.make_pixelshuffle_layers(block_size=3, kernel_size=3)  # outn=64
        self.conv_2 = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4, bias=True)
        self.tanh_1 = nn.Tanh()

    def forward(self, tensor):
        tensor = tensor.squeeze()
        output = self.dense_1(tensor)
        output = output.view(-1, 64, 16, 16)
        output = self.bn_1(output)
        output = self.relu_1(output)
        r_output = output
        output = self.residual_layer(output)
        output = self.conv_1(output)
        output = self.bn_2(output)
        output = self.relu_2(output)
        output += r_output
        output = self.pixelshuffle_layer(output)
        output = self.conv_2(output)
        output = self.tanh_1(output)
        return output

    def make_residual_layers(self, block_size=16, kernel_size=3):
        layers = []
        for _ in range(block_size):
            layers.append(ResidualBlock(64, 64, kernel_size, 1))
        return nn.Sequential(*layers)

    def make_pixelshuffle_layers(self, block_size=3, kernel_size=3):
        layers = []
        for _ in range(block_size):
            layers.append(PixelshuffleBlock(64, 256, kernel_size, 1))
        return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=1, bias=False):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        res_input = input
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.conv2(output)
        output = self.bn2(output)
        output += res_input
        return output


class PixelshuffleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=1, bias=False, upscale_factor=2):
        super(PixelshuffleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.PReLU()

    def forward(self, tensor):
        output = self.conv(tensor)
        output = self.pixel_shuffle(output)
        output = self.bn(output)
        output = self.relu(output)
        return output


class NonResnetDiscriminator(nn.Module):

    def __init__(self, nc, alpha, eps, ngpu=1):
        super(NonResnetDiscriminator, self).__init__()

        self.relevance = None
        self.ngpu = ngpu

        self.input = nnrd.Layer(
            OrderedDict(
                [
                    ('conv_input',
                     nnrd.FirstConvolution(in_channels=nc, out_channels=32, kernel_size=3, stride=1, padding=0)),
                    ('relu_input', nnrd.ReLu())
                ]
            )
        )
        # Block 1
        self.fill_1 = self.make_fill_block(in_channels=32, out_channels=32, num_fill_block=1, kernel_size=4, stride=2, padding=1, alpha=alpha)
        self.block_1 = self.make_block_layer(in_channels=32, out_channels=32, num_block=1, kernel_size=3, stride=1, padding=1, alpha=alpha)
        self.block_2 = self.make_block_layer(in_channels=32, out_channels=32, num_block=2, kernel_size=3, stride=1, padding=1, alpha=alpha)

        # Block 2
        self.fill_2 = self.make_fill_block(in_channels=32, out_channels=64, num_fill_block=2, kernel_size=4, stride=2, padding=1, alpha=alpha)
        self.block_3 = self.make_block_layer(in_channels=64, out_channels=64, num_block=3, kernel_size=3, stride=1, padding=1, alpha=alpha)
        self.block_4 = self.make_block_layer(in_channels=64, out_channels=64, num_block=4, kernel_size=3, stride=1, padding=1, alpha=alpha)
        self.block_5 = self.make_block_layer(in_channels=64, out_channels=64, num_block=5, kernel_size=3, stride=1, padding=1, alpha=alpha)
        self.block_6 = self.make_block_layer(in_channels=64, out_channels=64, num_block=6, kernel_size=3, stride=1, padding=1, alpha=alpha)

        # Block3
        self.fill_3 = self.make_fill_block(in_channels=64, out_channels=128, num_fill_block=2, kernel_size=4, stride=2, padding=1, alpha=alpha)
        self.block_7 = self.make_block_layer(in_channels=128, out_channels=128, num_block=7, kernel_size=3, stride=1, padding=1, alpha=alpha)
        self.block_8 = self.make_block_layer(in_channels=128, out_channels=128, num_block=8, kernel_size=3, stride=1, padding=1, alpha=alpha)
        self.block_9 = self.make_block_layer(in_channels=128, out_channels=128, num_block=9, kernel_size=3, stride=1, padding=1, alpha=alpha)
        self.block_10 = self.make_block_layer(in_channels=128, out_channels=128, num_block=10, kernel_size=3, stride=1, padding=1, alpha=alpha)

        # Block 4
        self.fill_4 = self.make_fill_block(in_channels=128, out_channels=256, num_fill_block=4, kernel_size=3, stride=2, padding=1, alpha=alpha)
        self.block_11 = self.make_block_layer(in_channels=256, out_channels=256, num_block=11, kernel_size=3, stride=1, padding=1, alpha=alpha)
        self.block_12 = self.make_block_layer(in_channels=256, out_channels=256, num_block=12, kernel_size=3, stride=1, padding=1, alpha=alpha)
        self.block_13 = self.make_block_layer(in_channels=256, out_channels=256, num_block=13, kernel_size=3, stride=1, padding=1, alpha=alpha)
        self.block_14 = self.make_block_layer(in_channels=256, out_channels=256, num_block=14, kernel_size=3, stride=1, padding=1, alpha=alpha)

        # Block 5
        self.fill_5 = self.make_fill_block(in_channels=256, out_channels=512, num_fill_block=5, kernel_size=3, stride=2, padding=1, alpha=alpha)
        self.block_15 = self.make_block_layer(in_channels=512, out_channels=512, num_block=15, kernel_size=3, stride=1, padding=1, alpha=alpha)
        self.block_16 = self.make_block_layer(in_channels=512, out_channels=512, num_block=16, kernel_size=3, stride=1, padding=1, alpha=alpha)
        self.block_17 = self.make_block_layer(in_channels=512, out_channels=512, num_block=17, kernel_size=3, stride=1, padding=1, alpha=alpha)
        self.block_18 = self.make_block_layer(in_channels=512, out_channels=512, num_block=18, kernel_size=3, stride=1, padding=1, alpha=alpha)

        self.end_6 = self.make_fill_block(in_channels=512, out_channels=1024, num_fill_block=6, kernel_size=3, stride=2, padding=1, eps=eps)

        self.net = nnrd.RelevanceNetAlternate(
            self.input,
            # 128 x 128
            self.fill_1,
            self.block_1,
            self.block_2,

            # 64 x 64
            self.fill_2,
            self.block_3,
            self.block_4,
            self.block_5,
            self.block_6,

            # 32 x 32
            self.fill_3,
            self.block_7,
            self.block_8,
            self.block_9,
            self.block_10,

            # 16 x 16
            self.fill_4,
            self.block_11,
            self.block_12,
            self.block_13,
            self.block_14,

            # 8 x 8
            self.fill_5,
            self.block_15,
            self.block_16,
            self.block_17,
            self.block_18,

            self.end_6,
        )

        self.lastConvolution = nnrd.LastConvolutionEps(in_channels=1024, out_channels=1, kernel_size=2, name='last',
                                                       stride=1, padding=0, epsilon=eps)

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
            probability = self.lastConvolution(output.detach())
            probability = self.sigmoid(probability.detach())

            output = self.lastConvolution(output, flip=flip)
            output = self.lastReLU(output)
            self.relevance = output
            return output.view(-1, 1).squeeze(1), probability.view(-1, 1).squeeze(1)

    def make_block_layer(self, in_channels, out_channels, num_block, alpha=1, kernel_size=3, stride=1, padding=1, bias=True):
        return nnrd.Layer(OrderedDict(
            [
                ('conv{}'.format(str(num_block)),
                 nnrd.NextConvolution(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      name=str(num_block), stride=stride, padding=padding, alpha=alpha)),
                ('relu{}'.format(str(num_block)), nnrd.ReLu()),
                ('conv{}'.format(str(num_block + 1)),
                 nnrd.NextConvolution(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      name=str(num_block + 1), stride=stride, padding=padding, alpha=alpha)),
                ('relu{}'.format(str(num_block + 1)), nnrd.ReLu()),
            ]
        ))

    def make_fill_block(self, in_channels, out_channels, num_fill_block, kernel_size, alpha=1, stride=2, padding=1, bias=True, eps=None):
        if eps:
            return nnrd.Layer(OrderedDict(
                [
                    ('fill_conv{}'.format(str(num_fill_block)),
                     nnrd.NextConvolutionEps(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                             name=str(num_fill_block), stride=stride, padding=padding, epsilon=eps)),
                    ('fill_relu{}'.format(str(num_fill_block)), nnrd.ReLu()),
                ]
            ))

        else:
            return nnrd.Layer(OrderedDict(
                [
                    ('fill_conv{}'.format(str(num_fill_block)),
                     nnrd.NextConvolution(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                          name=str(num_fill_block), stride=stride, padding=padding, alpha=alpha)),
                    ('fill_relu{}'.format(str(num_fill_block)), nnrd.ReLu()),
                ]
            ))

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


# ######################################## Less Checkerboard pattern ########################################

class GeneratorNetLessCheckerboard(nn.Module):
    def __init__(self, nc, ngf, ngpu):
        super(GeneratorNetLessCheckerboard, self).__init__()
        self.ngpu = ngpu
        nz = 100
        self.net = nn.Sequential(

            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(ngf * 16, ngf * 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
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


class DiscriminatorNetLessCheckerboardAlternate(nn.Module):

    def __init__(self, nc, ndf, alpha, beta, ngpu=1):
        super(DiscriminatorNetLessCheckerboardAlternate, self).__init__()

        self.relevance = None
        self.ngpu = ngpu

        self.net = nnrd.RelevanceNetAlternate(
            nnrd.Layer(
                nnrd.FirstConvolution(in_channels=nc, out_channels=ndf, kernel_size=5, stride=1, padding=0),
                nnrd.ReLu(),
            ),
            nnrd.Layer(
                nnrd.NextConvolution(in_channels=ndf, out_channels=ndf, kernel_size=4, name='0', stride=2, padding=1,
                                     alpha=alpha),
                nnrd.BatchNorm2d(ndf),
                nnrd.ReLu(),
                nnrd.Dropout(0.3),
            ),
            # state size. (ndf) x 32 x 32
            nnrd.Layer(
                nnrd.NextConvolution(in_channels=ndf, out_channels=ndf * 2, kernel_size=4, name='1', stride=2,
                                     padding=1, alpha=alpha),
                nnrd.BatchNorm2d(ndf * 2),
                nnrd.ReLu(),
                nnrd.Dropout(0.3),

            ),
            # state size. (ndf*2) x 16 x 16
            nnrd.Layer(
                nnrd.NextConvolution(in_channels=ndf * 2, out_channels=ndf * 4, kernel_size=4, name='2', stride=2,
                                     padding=1, alpha=alpha),
                nnrd.BatchNorm2d(ndf * 4),
                nnrd.ReLu(),
                nnrd.Dropout(0.3),
            ),  # state size. (ndf*2) x 16 x 16
            # state size. (ndf*4) x 8 x 8
            nnrd.Layer(
                nnrd.NextConvolutionEps(in_channels=ndf * 4, out_channels=ndf * 8, kernel_size=4, name='3', stride=2,
                                        padding=1),
                nnrd.BatchNorm2d(ndf * 8),
                nnrd.ReLu(),
                nnrd.Dropout(0.3),
            ),
            # state size. (ndf*8) x 4 x 4
            nnrd.Layer(
                nnrd.LastConvolutionEps(in_channels=ndf * 8, out_channels=1, kernel_size=4, name='4', stride=1,
                                        padding=0),
            )
        )
        self.sigmoid = nn.Sigmoid()
        self.lastReLU = nnrd.ReLu()

    def forward(self, x):

        if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.net, x, range(self.ngpu))
        else:
            output = self.net(x)

        if self.training:
            output = self.sigmoid(output)
        else:
            output = self.lastReLU(output)
            self.relevance = output

        return output.view(-1, 1).squeeze(1)

    def relprop(self):
        return self.net.relprop(self.relevance)

    def setngpu(self, ngpu):
        self.ngpu = ngpu


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
                # ('dropou2', nnrd.Dropout(0.3)),
            ])
            ),
            # state size. (ndf) x 32 x 32
            nnrd.Layer(OrderedDict([
                ('conv3', nnrd.NextConvolution(in_channels=ndf, out_channels=ndf * 2, kernel_size=4, name='1', stride=2,
                                               padding=1, alpha=alpha)),
                ('bn3', nnrd.BatchNorm2d(ndf * 2)),
                ('relu3', nnrd.ReLu()),
                # ('dropout3', nnrd.Dropout(0.3)),
            ])

            ),
            # state size. (ndf*2) x 16 x 16
            nnrd.Layer(OrderedDict([
                ('conv4',
                 nnrd.NextConvolution(in_channels=ndf * 2, out_channels=ndf * 4, kernel_size=4, name='2', stride=2,
                                      padding=1, alpha=alpha)),
                ('bn4', nnrd.BatchNorm2d(ndf * 4)),
                ('relu4', nnrd.ReLu()),
                # ('dropout4', nnrd.Dropout(0.3)),
            ])
            ),  # state size. (ndf*2) x 16 x 16
            # state size. (ndf*4) x 8 x 8
            nnrd.Layer(OrderedDict([
                ('conv5',
                 nnrd.NextConvolutionEps(in_channels=ndf * 4, out_channels=ndf * 8, kernel_size=4, name='3', stride=2,
                                         padding=1, epsilon=0.01)),
                ('bn5', nnrd.BatchNorm2d(ndf * 8)),
                ('relu5', nnrd.ReLu()),
                # ('dropout5', nnrd.Dropout(0.3)),
            ])
            ),
        )

        self.lastConvolution = nnrd.LastConvolutionEps(in_channels=ndf * 8, out_channels=1, kernel_size=4, name='4',
                                                       stride=1, padding=0, epsilon=0.01)

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
