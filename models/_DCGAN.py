import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data

import modules.ModuleRedefinitions as nnrd


# ######################################## Less Checkerboard Binary ########################################


class GeneratorNetBi(nn.Module):
    def __init__(self, nc, ngf, ngpu):
        super(GeneratorNetBi, self).__init__()
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


class DiscriminatorNetBi(nn.Module):

    def __init__(self, nc, ndf, alpha, beta, ngpu=1):
        super(DiscriminatorNetBi, self).__init__()

        self.ngpu = ngpu
        self.net = nnrd.RelevanceNet(
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
                nnrd.NextConvolutionEps(in_channels=ndf * 8, out_channels=2, kernel_size=4, name='4', stride=1,
                                        padding=0),
            )

        )

        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):

        if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.net, x, range(self.ngpu))
        else:
            output = self.net(x)
        output = output.squeeze()

        return self.softmax(output)

    def relprop(self):
        return self.net.relprop()

    def setngpu(self, ngpu):
        self.ngpu = ngpu


# ######################################## Less Checkerboard + No Padding ########################################


class GeneratorNetLessCheckerboardNoPad(nn.Module):
    def __init__(self, nc, ngf, ngpu):
        super(GeneratorNetLessCheckerboardNoPad, self).__init__()
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


class DiscriminatorNetLessCheckerboardNoPad(nn.Module):

    def __init__(self, nc, ndf, alpha, beta, ngpu=1):
        super(DiscriminatorNetLessCheckerboardNoPad, self).__init__()

        self.ngpu = ngpu
        self.net = nnrd.RelevanceNet(
            nnrd.Layer(
                nnrd.FirstConvolution(in_channels=nc, out_channels=ndf, kernel_size=5, stride=1, padding=0),
                nnrd.ReLu(),
            ),
            nnrd.Layer(
                nnrd.NextConvolution(in_channels=ndf, out_channels=ndf, kernel_size=4, name='0', stride=2, padding=0,
                                     alpha=alpha),
                nnrd.BatchNorm2d(ndf),
                nnrd.ReLu(),
                nnrd.Dropout(0.3),
            ),
            # state size. (ndf) x 32 x 32
            nnrd.Layer(
                nnrd.NextConvolution(in_channels=ndf, out_channels=ndf * 2, kernel_size=4, name='1', stride=2,
                                     padding=0, alpha=alpha),
                nnrd.BatchNorm2d(ndf * 2),
                nnrd.ReLu(),
                nnrd.Dropout(0.3),

            ),
            # state size. (ndf*2) x 16 x 16
            nnrd.Layer(
                nnrd.NextConvolution(in_channels=ndf * 2, out_channels=ndf * 4, kernel_size=4, name='2', stride=2,
                                     padding=0, alpha=alpha),
                nnrd.BatchNorm2d(ndf * 4),
                nnrd.ReLu(),
                nnrd.Dropout(0.3),
            ),
            # state size. (ndf*4) x 8 x 8
            nnrd.Layer(
                nnrd.NextConvolution(in_channels=ndf * 4, out_channels=ndf * 8, kernel_size=4, name='3', stride=2,
                                     padding=0, alpha=alpha),
                nnrd.BatchNorm2d(ndf * 8),
                nnrd.ReLu(),
                nnrd.Dropout(0.3),
            ),
            # state size. (ndf*8) x 4 x 4
            nnrd.Layer(
                nnrd.NextConvolution(in_channels=ndf * 8, out_channels=1, kernel_size=2, name='4', stride=1, padding=0),
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


# ################################### Less Checkerboard pattern + Training Tips ###################################


class GeneratorNetLessCheckerboardTips(nn.Module):
    def __init__(self, nc, ngf, ngpu):
        super(GeneratorNetLessCheckerboardTips, self).__init__()
        self.ngpu = ngpu
        nz = 100
        self.net = nn.Sequential(

            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Conv2d(ngf * 8, ngf * 8, 3, 1, 1),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

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


class DiscriminatorNetLessCheckerboardTips(nn.Module):

    def __init__(self, nc, ndf, alpha, beta, ngpu=1):
        super(DiscriminatorNetLessCheckerboardTips, self).__init__()

        self.ngpu = ngpu
        self.net = nnrd.RelevanceNet(
            nnrd.Layer(
                nnrd.FirstConvolution(nc, ndf, 3, 1, 1),
                nnrd.ReLu(),
            ),
            nnrd.Layer(
                nnrd.NextConvolution(ndf, ndf, 4, '0', 2, 1, alpha=alpha),
                nnrd.BatchNorm2d(ndf),
                nnrd.ReLu(),
            ),
            # state size. (ndf) x 32 x 32
            nnrd.Layer(
                nnrd.NextConvolution(ndf, ndf * 2, 4, '1', 2, 1, alpha=alpha),
                nnrd.BatchNorm2d(ndf * 2),
                nnrd.ReLu(),

            ),
            # state size. (ndf*2) x 16 x 16
            nnrd.Layer(
                nnrd.NextConvolution(ndf * 2, ndf * 4, 4, '2', 2, 1, alpha=alpha),
                nnrd.BatchNorm2d(ndf * 4),
                nnrd.ReLu(),
            ),
            # state size. (ndf*4) x 8 x 8
            nnrd.Layer(
                nnrd.NextConvolution(ndf * 4, ndf * 8, 4, '3', 2, 1, alpha=alpha),
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


# ######################################## Less Checkerboard pattern VBN ########################################


class GeneratorNetVBN(nn.Module):
    def __init__(self, nc, ngf, ngpu, ref_batch):
        super(GeneratorNetVBN, self).__init__()
        self.ngpu = ngpu
        self.ref_batch = ref_batch
        nz = 100

        self.net = nn.Sequential(

            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0),
            nnrd.VBN2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ngf * 8, ngf * 8, 3, 1, 1),
            nnrd.VBN2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
            nnrd.VBN2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*4) x 8 x 8

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
            nnrd.VBN2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*2) x 16 x 16

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
            nnrd.VBN2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, x):
        ref_x = self.ref_batch

        # Reference Pass for virtual batch norm

        ref_x = self.net[0](ref_x)
        ref_x, ref_mean1, ref_meansq1 = self.net[1](ref_x, None, None)
        ref_x = self.net[2](ref_x)

        ref_x = self.net[3](ref_x)
        ref_x, ref_mean2, ref_meansq2 = self.net[4](ref_x, None, None)
        ref_x = self.net[5](ref_x)

        ref_x = self.net[6](ref_x)
        ref_x, ref_mean3, ref_meansq3 = self.net[7](ref_x, None, None)
        ref_x = self.net[8](ref_x)

        ref_x = self.net[9](ref_x)
        ref_x, ref_mean4, ref_meansq4 = self.net[10](ref_x, None, None)
        ref_x = self.net[11](ref_x)

        ref_x = self.net[12](ref_x)
        _, ref_mean5, ref_meansq5 = self.net[13](ref_x, None, None)

        # We don't have to go forward any longer because we don't need the result, only the ref_means

        # Actual training pass

        x = self.net[0](x)
        x, _, _ = self.net[1](x, ref_mean1, ref_meansq1)
        x = self.net[2](x)

        x = self.net[3](x)
        x, _, _ = self.net[4](x, ref_mean2, ref_meansq2)
        x = self.net[5](x)

        x = self.net[6](x)
        x, _, _ = self.net[7](x, ref_mean3, ref_meansq3)
        x = self.net[8](x)

        x = self.net[9](x)
        x, _, _ = self.net[10](x, ref_mean4, ref_meansq4)
        x = self.net[11](x)

        x = self.net[12](x)
        x, _, _ = self.net[13](x, ref_mean5, ref_meansq5)
        x = self.net[14](x)

        x = self.net[15](x)
        x = self.net[16](x)

        return x

        # if x.is_cuda and self.ngpu > 1:
        #     output = nn.parallel.data_parallel(self.net, x, range(self.ngpu))
        # else:
        #     output = self.net(x)
        # return output


class DiscriminatorNetVBN(nn.Module):

    def __init__(self, nc, ndf, alpha, beta, ngpu=1):
        super(DiscriminatorNetVBN, self).__init__()

        self.ngpu = ngpu
        self.net = nnrd.RelevanceNet(
            nnrd.Layer(
                nnrd.FirstConvolution(nc, ndf, 5, 1, 2),
                nnrd.ReLu(),
            ),
            nnrd.Layer(
                nnrd.NextConvolution(ndf, ndf, 4, '0', 2, 1, alpha=alpha),
                nnrd.BatchNorm2d(ndf),
                nnrd.ReLu(),
            ),
            # state size. (ndf) x 32 x 32
            nnrd.Layer(
                nnrd.NextConvolution(ndf, ndf * 2, 4, '1', 2, 1, alpha=alpha),
                nnrd.BatchNorm2d(ndf * 2),
                nnrd.ReLu(),
            ),
            # state size. (ndf*2) x 16 x 16
            nnrd.Layer(
                nnrd.NextConvolution(ndf * 2, ndf * 4, 4, '2', 2, 1, alpha=alpha),
                nnrd.BatchNorm2d(ndf * 4),
                nnrd.ReLu(),
            ),
            # state size. (ndf*4) x 8 x 8
            nnrd.Layer(
                nnrd.NextConvolution(ndf * 4, ndf * 8, 4, '3', 2, 1, alpha=alpha),
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
                nnrd.NextConvolutionEps(in_channels=ndf * 8, out_channels=1, kernel_size=4, name='4', stride=1,
                                        padding=0),
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

    def __init__(self, nc, ndf, alpha, ngpu=1):
        super(DiscriminatorNet, self).__init__()

        self.ngpu = ngpu
        self.net = nnrd.RelevanceNet(
            nnrd.Layer(
                nnrd.FirstConvolution(nc, ndf, 4, 2, 1),
                nnrd.ReLu(),
            ),
            # state size. (ndf) x 32 x 32
            nnrd.Layer(
                nnrd.NextConvolution(ndf, ndf * 2, 4, '1', 2, 1, alpha=alpha),
                nnrd.BatchNorm2d(ndf * 2),
                nnrd.ReLu(),
            ),
            # state size. (ndf*2) x 16 x 16
            nnrd.Layer(
                nnrd.NextConvolution(ndf * 2, ndf * 4, 4, '2', 2, 1, alpha=alpha),
                nnrd.BatchNorm2d(ndf * 4),
                nnrd.ReLu(),
            ),
            # state size. (ndf*4) x 8 x 8
            nnrd.Layer(
                nnrd.NextConvolution(ndf * 4, ndf * 8, 4, '3', 2, 1, alpha=alpha),
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
