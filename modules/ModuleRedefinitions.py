import torch
from torch import nn
from utils import utils
import copy
import numpy as np


class FirstConvolution(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=2, dilation=1, groups=1,
                 bias=False):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        # Variables for Relevance Propagation
        self.X = None

    def forward(self, input):
        # Input shape: minibatch x in_channels, iH x iW
        self.X = input.clone()
        return super().forward(input)

    def relprop(self, R):

        if type(R) is tuple:

            R, params = R

            gamma, var, eps, beta, mean = params['gamma'], params['var'], params['eps'], params['beta'], \
                                          params['mean']
            var = torch.sqrt(var + eps)

            iself = type(self)(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
            iself.load_state_dict(self.state_dict())
            iself.X = self.X.clone()

            iself.weight.data = iself.weight * (gamma / var).reshape(iself.out_channels, 1, 1, 1)

            nself = type(self)(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
            nself.load_state_dict(self.state_dict())
            nself.X = self.X.clone()

            nself.weight.data = nself.weight * (gamma / var).reshape(nself.out_channels, 1, 1, 1)
            nself.weight.data = torch.min(torch.Tensor(1).zero_(), nself.weight)

            pself = type(self)(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
            pself.load_state_dict(self.state_dict())
            pself.X = self.X.clone()

            pself.weight.data = pself.weight * (gamma / var).reshape(pself.out_channels, 1, 1, 1)

            pself.weight.data = torch.max(torch.Tensor(1).zero_(), pself.weight)

            X = iself.X
            L = nself.X.fill_(utils.lowest)
            H = pself.X.fill_(utils.highest)

            iself_f = iself.forward(X)
            pself_f = pself.forward(L)
            nself_f = nself.forward(H)

            Z = iself_f - pself_f - nself_f + 1e-9
            S = R / Z

            iself_b = torch.autograd.grad(iself_f, X, S, retain_graph=True)[0]
            pself_b = torch.autograd.grad(pself_f, L, S, retain_graph=True)[0]
            nself_b = torch.autograd.grad(nself_f, H, S)[0]

            R = X * iself_b - L * pself_b - H * nself_b

        else:

            iself = type(self)(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
            iself.load_state_dict(self.state_dict())
            iself.X = self.X.clone()
            # iself.bias.data *= 0

            nself = type(self)(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
            nself.load_state_dict(self.state_dict())
            nself.X = self.X.clone()
            # nself.bias.data *= 0
            nself.weight.data = torch.min(torch.Tensor(1).zero_(), nself.weight)

            pself = type(self)(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
            pself.load_state_dict(self.state_dict())
            pself.X = self.X.clone()
            # pself.bias.data *= 0
            pself.weight.data = torch.max(torch.Tensor(1).zero_(), pself.weight)

            X = iself.X
            L = nself.X.fill_(utils.lowest)
            H = pself.X.fill_(utils.highest)

            iself_f = iself.forward(X)
            pself_f = pself.forward(L)
            nself_f = nself.forward(H)

            Z = iself_f - pself_f - nself_f + 1e-9
            S = R / Z

            iself_b = torch.autograd.grad(iself_f, X, S, retain_graph=True)[0]
            pself_b = torch.autograd.grad(pself_f, L, S, retain_graph=True)[0]
            nself_b = torch.autograd.grad(nself_f, H, S)[0]

            R = X * iself_b - L * pself_b - H * nself_b

        # print('Input layer weight max: {:.6f}, min: {:.6f}, mean: {:.6f}'.format(self.weight.max(), self.weight.min(), self.weight.mean()))
        return R.detach()


class NextConvolution(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, name, stride=1, padding=2, dilation=1, groups=1,
                 bias=True, alpha=1):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        self.name = name
        # Variables for Relevance Propagation
        self.X = None
        self.alpha = alpha
        self.beta = alpha - 1

    def forward(self, input):
        # Input shape: minibatch x in_channels, iH x iW
        self.X = input.clone()
        return super().forward(input)

    def relprop(self, R):

        # Is the layer before Batch Norm?
        if type(R) is tuple:
            R, params = R

            gamma, var, eps, beta, mean = params['gamma'], params['var'], params['eps'], params['beta'], \
                                          params['mean']
            var_sqrt = torch.sqrt(var + eps)

            # Positive weights
            pself = type(self)(self.in_channels, self.out_channels, self.kernel_size, self.name, self.stride,
                               self.padding)
            pself.load_state_dict(self.state_dict())
            pself.X = self.X.clone()
            pself.alpha = self.alpha
            pself.bias.data = torch.max(torch.Tensor(1).zero_(), pself.bias)
            pself.weight.data = pself.weight * (gamma / var).reshape(pself.out_channels, 1, 1, 1)
            pself.weight.data = torch.max(torch.Tensor([1e-9]), pself.weight)

            # Negative weights
            nself = type(self)(self.in_channels, self.out_channels, self.kernel_size, self.name, self.stride,
                               self.padding)
            nself.load_state_dict(self.state_dict())
            nself.X = self.X.clone()
            nself.beta = self.beta
            nself.bias.data = torch.min(torch.Tensor(1).zero_(), nself.bias)
            nself.weight.data = nself.weight * (gamma / var).reshape(nself.out_channels, 1, 1, 1)
            nself.weight.data = torch.min(torch.Tensor([-1e-9]), nself.weight)

            pX = pself.X + 1e-9
            nX = nself.X + 1e-9

            ZA = pself(pX)
            SA = pself.alpha * torch.div(R, ZA)

            ZB = nself(nX)
            SB = - nself.beta * torch.div(R, ZB)

            C = torch.autograd.grad(ZA, pX, SA)[0] + torch.autograd.grad(ZB, nX, SB)[0]
            R = pself.X * C

        # If not, continue as usual
        else:

            # positive
            pself = type(self)(self.in_channels, self.out_channels, self.kernel_size, self.name, self.stride,
                               self.padding)
            pself.load_state_dict(self.state_dict())
            pself.X = self.X.clone()
            pself.alpha = self.alpha
            pself.bias.data = torch.max(torch.Tensor(1).zero_(), pself.bias)
            pself.weight.data = torch.max(torch.Tensor([1e-9]), pself.weight)

            # negative
            nself = type(self)(self.in_channels, self.out_channels, self.kernel_size, self.name, self.stride,
                               self.padding)
            nself.load_state_dict(self.state_dict())
            nself.X = self.X.clone()
            nself.beta = self.beta
            nself.bias.data = torch.min(torch.Tensor(1).zero_(), nself.bias)
            nself.weight.data = torch.min(torch.Tensor([-1e-9]), nself.weight)

            pX = pself.X + 1e-9
            nX = nself.X + 1e-9

            ZA = pself(pX)
            SA = pself.alpha * torch.div(R, ZA)

            ZB = nself(nX)
            SB = - nself.beta * torch.div(R, ZB)
            ones = torch.Tensor([[1, 1], [1, 1]]).unsqueeze(0).unsqueeze(0)

            C = torch.autograd.grad(ZA, pX, SA)[0] + torch.autograd.grad(ZB, nX, SB)[0]
            R = pX * C

        # utils.Logger.save_intermediate_heatmap(torch.sum(R, 1, keepdim=True).detach(), self.name)
        # print('Layer {}: {}'.format(self.name, R.abs().sum().item()))
        return R

    def incorporateBatchNorm(self, bn):

        gamma = bn.weight
        beta = bn.bias
        mean = bn.running_mean
        var = bn.running_var
        eps = bn.eps

        var_sqrt = torch.sqrt(var + eps)

        w = (self.weight * gamma.reshape(self.out_channels, 1, 1, 1)) / var_sqrt.reshape(self.out_channels, 1,
                                                                                         1, 1)
        b = ((self.bias - mean) * gamma) / var_sqrt + beta

        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(b)


class NextConvolutionEps(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, name, stride=1, padding=2, dilation=1, groups=1,
                 bias=True, epsilon=1e-9):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        self.name = name
        # Variables for Relevance Propagation
        self.X = None
        self.epsilon = epsilon

    def forward(self, input):
        # Input shape: minibatch x in_channels, iH x iW
        self.X = input.clone()
        return super().forward(input)

    def relprop(self, R):

        # Is the layer before Batch Norm?
        if type(R) is tuple:
            R, params = R

            gamma, var, eps, beta, mean = params['gamma'], params['var'], params['eps'], params['beta'], \
                                          params['mean']
            var = torch.div(torch.ones(1), (torch.sqrt(var + eps)))

            # Clone so we don't influence actual layer
            iself = type(self)(self.in_channels, self.out_channels, self.kernel_size, self.name, self.stride,
                               self.padding)
            iself.load_state_dict(self.state_dict())
            iself.X = self.X.clone()
            iself.weight.data = iself.weight.data * gamma.view(-1, 1, 1, 1).expand_as(iself.weight) \
                                * var.unsqueeze(1).view(-1, 1, 1, 1).expand_as(iself.weight)

            iX = iself.X

            ZA = iself(iX) + self.epsilon
            SA = torch.div(R, ZA)

            C = torch.autograd.grad(ZA, iX, SA)[0]
            R = iself.X * C

        # If not, continue as usual
        else:

            # Clone so we don't influence actual layer
            iself = type(self)(self.in_channels, self.out_channels, self.kernel_size, self.name, self.stride,
                               self.padding)
            iself.load_state_dict(self.state_dict())
            iself.X = self.X.clone()

            iX = torch.tensor(iself.X.data, requires_grad=True)
            Z = iself(iX) + self.epsilon
            S = torch.div(R, Z)
            C = torch.autograd.grad(Z, iX, S)[0]
            R = iself.X * C

            # a = torch.tensor(iself.X.data, requires_grad=True)
            # ZA = iself(a)
            # # ZA = iself(a) + self.epsilon
            # SA = torch.div(R, ZA + self.epsilon).data
            #
            # (ZA * SA).sum().backward()
            # c = a.grad
            # R = (a * c).data
            #
            # print('Next convolution', np.allclose(R.detach().cpu().numpy(), r.detach().cpu().numpy()))
            # C = torch.autograd.grad(ZA, iX, SA)[0]
            # R = iself.X * C

        # utils.Logger.save_intermediate_heatmap(torch.sum(R, 1, keepdim=True).detach(), self.name)
        # print('Layer {}: {}'.format(self.name, R.abs().sum().item()))
        return R

    def incorporateBatchNorm(self, bn):

        gamma = bn.weight
        beta = bn.bias
        mean = bn.running_mean
        var = bn.running_var
        eps = bn.eps

        var_sqrt = torch.sqrt(var + eps)

        w = (self.weight * gamma.reshape(self.out_channels, 1, 1, 1)) / var_sqrt.reshape(self.out_channels, 1,
                                                                                         1, 1)
        b = ((self.bias - mean) * gamma) / var_sqrt + beta

        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(b)


class LastConvolutionEps(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, name, stride=1, padding=2, dilation=1, groups=1,
                 bias=True, epsilon=1e-2):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        self.name = name
        # Variables for Relevance Propagation
        self.X = None
        self.epsilon = epsilon

    def forward(self, input, flip=False):
        # Input shape: minibatch x in_channels, iH x iW
        self.X = input.clone()

        if flip:
            self.weight.data *= -1
            self.bias.data *= -1

        output = super().forward(input)

        if flip:
            self.weight.data *= -1
            self.bias.data *= -1

        return output

    def relprop(self, R, flip=False):

        # Is the layer before Batch Norm?
        if type(R) is tuple:
            R, params = R

            gamma, var, eps, beta, mean = params['gamma'], params['var'], params['eps'], params['beta'], \
                                          params['mean']
            var = torch.div(torch.ones(1), (torch.sqrt(var + eps)))

            # Clone so we don't influence actual layer
            iself = type(self)(self.in_channels, self.out_channels, self.kernel_size, self.name, self.stride,
                               self.padding)
            iself.load_state_dict(self.state_dict())
            iself.X = self.X.clone()
            # iself.bias.data *= 0
            iself.weight.data = iself.weight.data * gamma.view(-1, 1, 1, 1).expand_as(iself.weight) \
                                * var.unsqueeze(1).view(-1, 1, 1, 1).expand_as(iself.weight)

            if flip:
                iself.weight.data *= -1
                iself.bias.data *= -1

            iX = torch.tensor(iself.X.data, requires_grad=True)
            Z = iself(iX) + self.epsilon
            S = torch.div(R, Z)
            C = torch.autograd.grad(Z, iX, S)[0]
            R = iself.X * C

            # a = torch.tensor(iself.X.data, requires_grad=True)
            # ZA = iself(a)
            # # ZA = iself(a) + self.epsilon
            # SA = torch.div(R, ZA + self.epsilon).data
            #
            # (ZA * SA).sum().backward()
            # c = a.grad
            # R = (a * c).data
            #
            # print('Last convolution', np.allclose(R.detach().cpu().numpy(), r.detach().cpu().numpy()))
            # C = torch.autograd.grad(ZA, iX, SA)[0]
            # R = iself.X * C

        # If not, continue as usual
        else:

            # Clone so we don't influence actual layer
            iself = type(self)(self.in_channels, self.out_channels, self.kernel_size, self.name, self.stride,
                               self.padding)
            iself.load_state_dict(self.state_dict())
            iself.X = self.X.clone()

            if flip:
                iself.weight.data *= -1
                iself.bias.data *= -1

            iX = torch.tensor(iself.X.data, requires_grad=True)
            Z = iself(iX) + self.epsilon
            S = torch.div(R, Z)
            C = torch.autograd.grad(Z, iX, S)[0]
            R = iself.X * C

            # a = torch.tensor(iself.X.data, requires_grad=True)
            # ZA = iself(a)
            # # ZA = iself(a) + self.epsilon
            # SA = torch.div(R, ZA + self.epsilon).data
            #
            # (ZA * SA).sum().backward()
            # c = a.grad
            # R = (a * c).data

            # print('Last convolution', np.allclose(R.detach().cpu().numpy(), r.detach().cpu().numpy()))

        # utils.Logger.save_intermediate_heatmap(torch.sum(R, 1, keepdim=True).detach(), self.name)
        # print('Last layer', R.abs().sum())
        return R


class ReLu(nn.ReLU):

    def relprop(self, R):
        return R


class LeakyReLU(nn.LeakyReLU):

    def relprop(self, R):
        return R


class BatchNorm2d(nn.BatchNorm2d):

    def relprop(self, R):
        return R, self.getParams()
        return R

    def getParams(self):
        return {'gamma': copy.deepcopy(self.weight), 'var': copy.deepcopy(self.running_var),
                'eps': copy.deepcopy(self.eps), 'beta': copy.deepcopy(self.bias),
                'mean': copy.deepcopy(self.running_mean)}


class GaussianNoise(nn.Module):
    def __init__(self, stddev=0.05):
        super().__init__()
        self.stddev = stddev

    def forward(self, x):
        if not self.training: return x
        return x + torch.Tensor(torch.randn(x.size()) * self.stddev)


class VBN2d(nn.Module):
    """
    Module for Virtual Batch Normalization.
    Implementation borrowed and modified from Rafael_Valle's code + help of SimonW from this discussion thread:
    https://discuss.pytorch.org/t/parameter-grad-of-conv-weight-is-none-after-virtual-batch-normalization/9036
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        # batch statistics
        self.num_features = num_features
        self.eps = eps  # epsilon
        # self.ref_mean = self.register_parameter('ref_mean', None)
        # self.ref_mean_sq = self.register_parameter('ref_mean_sq', None)

        # define gamma and beta parameters
        gamma = torch.normal(mean=torch.ones(1, num_features, 1, 1), std=0.02)
        self.gamma = nn.Parameter(gamma.float())
        self.beta = nn.Parameter(torch.Tensor(1, num_features, 1, 1).fill_(0))
        self.referenceOutput = None

    def get_stats(self, x):
        """
        Calculates mean and mean square for given batch x.
        Args:
            x: tensor containing batch of activations
        Returns:
            mean: mean tensor over features
            mean_sq: squared mean tensor over features
        """
        mean = x.mean(3, keepdim=True).mean(2, keepdim=True).mean(0, keepdim=True)
        mean_sq = (x ** 2).mean(3, keepdim=True).mean(2, keepdim=True).mean(0, keepdim=True)
        return mean, mean_sq

    def forward(self, x, ref_mean: None, ref_mean_sq: None):
        """
        Forward pass of virtual batch normalization.
        Virtual batch normalization require two forward passes
        for reference batch and train batch, respectively.
        The input parameter is_reference should indicate whether it is a forward pass
        for reference batch or not.
        Args:
            x: input tensor
            is_reference(bool): True if forwarding for reference batch
        Result:
            x: normalized batch tensor
        """
        mean, mean_sq = self.get_stats(x)
        if ref_mean is None or ref_mean_sq is None:
            # reference mode - works just like batch norm
            mean = mean.clone().detach()
            mean_sq = mean_sq.clone().detach()
            out = self._normalize(x, mean, mean_sq)
        else:
            # calculate new mean and mean_sq
            batch_size = x.size(0)
            new_coeff = 1. / (batch_size + 1.)
            old_coeff = 1. - new_coeff
            mean = new_coeff * mean + old_coeff * ref_mean
            mean_sq = new_coeff * mean_sq + old_coeff * ref_mean_sq
            out = self._normalize(x, mean, mean_sq)
        return out, mean, mean_sq

    def _normalize(self, x, mean, mean_sq):
        """
        Normalize tensor x given the statistics.
        Args:
            x: input tensor
            mean: mean over features. it has size [1:num_features:]
            mean_sq: squared means over features.
        Result:
            x: normalized batch tensor
        """
        assert mean_sq is not None
        assert mean is not None
        assert len(x.size()) == 4  # specific for 2d VBN
        if mean.size(1) != self.num_features:
            raise Exception(
                'Mean size not equal to number of featuers : given {}, expected {}'
                    .format(mean.size(1), self.num_features))
        if mean_sq.size(1) != self.num_features:
            raise Exception(
                'Squared mean tensor size not equal to number of features : given {}, expected {}'
                    .format(mean_sq.size(1), self.num_features))

        std = torch.sqrt(self.eps + mean_sq - mean ** 2)
        x = x - mean
        x = x / std
        x = x * self.gamma
        x = x + self.beta
        return x

    def __repr__(self):
        return ('{name}(num_features={num_features}, eps={eps}'
                .format(name=self.__class__.__name__, **self.__dict__))


class Dropout(nn.Dropout):

    def __init__(self, p=0.5, inplace=False):
        super().__init__(p, inplace)

    #
    # def forward(self, input):
    #     output = super().forward(input)
    #     return super().forward(input)

    def relprop(self, R):
        return R


class FirstLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias=bias)

        # Variables for Relevance Propagation
        self.X = None

    def forward(self, input):
        self.X = input.clone()
        return super().forward(input)

    def relprop(self, R):
        W = self.weight
        V = torch.max(torch.Tensor(1).zero_(), self.weight)
        U = torch.min(torch.Tensor(1).zero_(), self.weight)
        X = self.X
        L = self.X * 0 + utils.lowest
        H = self.X * 0 + utils.highest

        Z = torch.matmul(X, torch.t(W)) - torch.matmul(L, torch.t(V)) - torch.matmul(H, torch.t(U)) + 1e-9
        S = R / Z
        R = X * torch.matmul(S, W) - L * torch.matmul(S, V) - H * torch.matmul(S, U)
        return R.detach()


class NextLinear(nn.Linear):

    # Disable Bias
    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias=bias)

        # Variables for Relevance Propagation
        self.X = None

    def forward(self, input):
        self.X = input.clone()
        return super().forward(input)

    def relprop(self, R):
        X = self.X.clone()
        V = torch.max(torch.Tensor(1).zero_(), self.weight)
        Z = torch.matmul(X, torch.t(V)) + 1e-9
        S = R / Z
        C = torch.matmul(S, V)
        R = X * C

        return R


class LastLinear(NextLinear):

    def forward(self, input):
        input = torch.reshape(input, (input.size(0), 1, input.size(2) * input.size(3)))
        self.X = input.clone()
        return super().forward(input)


class FlattenLayer(nn.Module):

    def forward(self, input):
        return input.view(-1, 1)


class FlattenToLinearLayer(nn.Module):

    def forward(self, input):
        return input.squeeze(-1).squeeze(-1)

    def relprop(self, R):
        return R.unsqueeze(-1).unsqueeze(-1)


class Pooling(nn.AvgPool2d):

    def __init__(self, kernel_size, name=''):
        super().__init__(kernel_size)
        self.name = name
        self.X = None

    def forward(self, input):
        self.X = input.clone()
        output = super().forward(input) * self.kernel_size
        if self.name == 'global':
            output = output.squeeze()
        return output

    def relprop(self, R):
        if self.name == 'global':
            R.unsqueeze(-1).unsqueeze(-1)

        X = self.X.clone()
        Z = (self.forward(X) + 1e-9)
        S = R / Z
        C = torch.autograd.grad(Z, X, S)[0]
        R = X * C
        return R.detach()


class ReshapeLayer(nn.Module):

    def __init__(self, filters, height, width):
        super().__init__()
        self.filters = filters
        self.height = height
        self.width = width

    def forward(self, input):
        return input.view(-1, self.filters, self.height, self.width)


class RelevanceNetAlternate(nn.Sequential):

    def __init__(self, *args):
        super().__init__(*args)
        self.relevanceOutput = None

    def forward(self, input):

        # self.relevanceOutput = None

        for idx, layer in enumerate(self):
            input = layer.forward(input)
            # save output of second-to-last layer to use in relevance propagation
            # if idx == len(self) - 2:
            #     self.relevanceOutput = input

        return input

    def relprop(self, relevance):
        R = relevance.clone()
        # For all layers
        for layer in self[::-1]:
            R = layer.relprop(R)
        return R


class RelevanceNet(nn.Sequential):

    def __init__(self, *args):
        super().__init__(*args)
        self.relevanceOutput = None

    def forward(self, input):

        self.relevanceOutput = None

        for idx, layer in enumerate(self):
            input = layer.forward(input)

            # save output of second-to-last layer to use in relevance propagation
            if idx == len(self) - 2:
                self.relevanceOutput = input

        return input

    def relprop(self):
        R = self.relevanceOutput.clone()
        # For all layers except the last
        for layer in self[-2::-1]:
            R = layer.relprop(R)
        return R


class Layer(nn.Sequential):
    # def __init__(self, *args):
    #     super().__init__(*args)
    #
    # def forward(self, input):
    #     return super().forward(input)

    def relprop(self, R):
        for layer in self[::-1]:
            R = layer.relprop(R)
        return R


class DiscriminatorNet(nn.Module):

    def __init__(self, ndf, nc, ngpu=1):
        super(DiscriminatorNet, self).__init__()

        self.ngpu = ngpu
        self.net = None

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


