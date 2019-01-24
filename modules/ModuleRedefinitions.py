import torch
from torch import nn
from utils import utils
import copy


class FirstConvolution(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=2, dilation=1, groups=1,
                 bias=True):
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
            var = torch.div(torch.ones(1), (torch.sqrt(var + eps)))

            iself = type(self)(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
            iself.load_state_dict(self.state_dict())
            iself.X = self.X.clone()
            # Include positive biases as neurons
            # iself_biases = copy.deepcopy(iself.bias.data)
            # iself_biases = beta + gamma * (iself_biases - mean) * gamma
            iself.bias.data *= 0
            iself.weight.data = iself.weight.data * gamma.view(-1, 1, 1, 1).expand_as(iself.weight) \
                                * var.view(-1, 1, 1, 1).expand_as(iself.weight)

            nself = type(self)(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
            nself.load_state_dict(self.state_dict())
            nself.X = self.X.clone()
            # Include positive biases as neurons
            # nself_biases = copy.deepcopy(nself.bias.data)
            # nself_biases = beta + gamma * (nself_biases - mean) * gamma
            nself.bias.data *= 0
            nself.weight.data = nself.weight.data * gamma.view(-1, 1, 1, 1).expand_as(nself.weight) \
                                * var.view(-1, 1, 1, 1).expand_as(nself.weight)
            nself.weight.data = torch.min(torch.Tensor(1).zero_(), nself.weight)

            pself = type(self)(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
            pself.load_state_dict(self.state_dict())
            pself.X = self.X.clone()
            # Include positive biases as neurons to normalize over
            # pself_biases = copy.deepcopy(pself.bias.data)
            # incorporate batch norm
            # pself_biases = beta + gamma * (pself_biases - mean) * gamma
            pself.bias.data *= 0
            pself.weight.data = pself.weight.data * gamma.view(-1, 1, 1, 1).expand_as(pself.weight) \
                                * var.view(-1, 1, 1, 1).expand_as(pself.weight)
            pself.weight.data = torch.max(torch.Tensor(1).zero_(), pself.weight)

            X = iself.X
            L = nself.X.fill_(utils.lowest)
            H = pself.X.fill_(utils.highest)

            iself_f = iself.forward(X)
            # Expand bias for addition
            # iself_biases = torch.max(torch.Tensor(1).zero_(), iself_biases).view(1, -1, 1, 1).expand_as(iself_f)
            # iself_f = iself_f + iself_biases
            pself_f = pself.forward(L)
            # Expand bias for addition
            # pself_biases = torch.max(torch.Tensor(1).zero_(), pself_biases).view(1, -1, 1, 1).expand_as(pself_f)
            # pself_f = pself_f + pself_biases
            nself_f = nself.forward(H)
            # Expand bias for addition
            # nself_biases = torch.max(torch.Tensor(1).zero_(), nself_biases).view(1, -1, 1, 1).expand_as(nself_f)
            # nself_f = nself_f + nself_biases

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
            # Include positive biases as neurons
            # iself_biases = copy.deepcopy(iself.bias.data)
            iself.bias.data *= 0

            nself = type(self)(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
            nself.load_state_dict(self.state_dict())
            nself.X = self.X.clone()
            # Include positive biases as neurons
            # nself_biases = copy.deepcopy(nself.bias.data)
            nself.bias.data *= 0
            nself.weight.data = torch.min(torch.Tensor(1).zero_(), nself.weight)

            pself = type(self)(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
            pself.load_state_dict(self.state_dict())
            pself.X = self.X.clone()
            # Include positive biases as neurons
            # pself_biases = copy.deepcopy(pself.bias.data)
            pself.bias.data *= 0
            pself.weight.data = torch.max(torch.Tensor(1).zero_(), pself.weight)

            X = iself.X
            L = nself.X.fill_(utils.lowest)
            H = pself.X.fill_(utils.highest)

            iself_f = iself.forward(X)
            # Expand bias for addition
            # iself_biases = torch.max(torch.Tensor(1).zero_(), iself_biases).view(1, -1, 1, 1).expand_as(iself_f)
            # iself_f = iself_f + iself_biases
            pself_f = pself.forward(L)
            # Expand bias for addition
            # pself_biases = torch.max(torch.Tensor(1).zero_(), pself_biases).view(1, -1, 1, 1).expand_as(pself_f)
            # pself_f = pself_f + pself_biases
            nself_f = nself.forward(H)
            # Expand bias for addition
            # nself_biases = torch.max(torch.Tensor(1).zero_(), nself_biases).view(1, -1, 1, 1).expand_as(nself_f)
            # nself_f = nself_f + nself_biases

            Z = iself_f - pself_f - nself_f + 1e-9
            S = R / Z

            iself_b = torch.autograd.grad(iself_f, X, S, retain_graph=True)[0]
            pself_b = torch.autograd.grad(pself_f, L, S, retain_graph=True)[0]
            nself_b = torch.autograd.grad(nself_f, H, S)[0]

            R = X * iself_b - L * pself_b - H * nself_b

        return R.detach()


class NextConvolution(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, name, stride=1, padding=2, dilation=1, groups=1,
                 bias=True, alpha=1, beta=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        self.name = name
        # Variables for Relevance Propagation
        self.X = None
        self.alpha = alpha
        if beta is None:
            self.beta = alpha - 1
        else:
            self.beta = beta

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

            # Positive weights
            pself = type(self)(self.in_channels, self.out_channels, self.kernel_size, self.name, self.stride,
                               self.padding)
            pself.load_state_dict(self.state_dict())
            pself.X = self.X.clone()
            pself.alpha = self.alpha
            # Include positive biases as neurons to normalize over
            # pself_biases = copy.deepcopy(pself.bias.data)
            # pself_biases = beta + gamma * (pself_biases - mean) * gamma
            pself.bias.data *= 0
            pself.weight.data = pself.weight.data * gamma.view(-1, 1, 1, 1).expand_as(pself.weight) \
                                * var.unsqueeze(1).view(-1, 1, 1, 1).expand_as(pself.weight)
            pself.weight.data = torch.max(torch.Tensor([1e-9]), pself.weight)

            # Negative weights
            nself = type(self)(self.in_channels, self.out_channels, self.kernel_size, self.name, self.stride,
                               self.padding)
            nself.load_state_dict(self.state_dict())
            nself.X = self.X.clone()
            nself.beta = self.beta

            # Include positive biases as neurons to normalize over
            # nself_biases = copy.deepcopy(pself.bias.data)
            # nself_biases = beta + gamma * (nself_biases - mean) * gamma
            nself.bias.data *= 0
            nself.weight.data = nself.weight.data * gamma.view(-1, 1, 1, 1).expand_as(nself.weight) \
                                * var.view(-1, 1, 1, 1).expand_as(nself.weight)
            nself.weight.data = torch.min(torch.Tensor([-1e-9]), nself.weight)

            pX = pself.X + 1e-9
            nX = nself.X + 1e-9

            ZA = pself(pX)
            # expand biases for addition
            # pself_biases = torch.max(torch.Tensor(1).zero_(), pself_biases).view(1, -1, 1, 1).expand_as(ZA)
            # ZA = ZA + pself_biases
            SA = pself.alpha * torch.div(R, ZA)

            ZB = nself(nX)
            # expand biases for addition HERE NEGATIVE BIASES? torch.min???
            # nself_biases = torch.min(torch.Tensor(1).zero_(), nself_biases).view(1, -1, 1, 1).expand_as(ZB)
            # ZB = ZB + nself_biases
            SB = - nself.beta * torch.div(R, ZB)

            C = torch.autograd.grad(ZA, pX, SA)[0] + torch.autograd.grad(ZB, nX, SB)[0]
            R = pself.X * C

        # If not, continue as usual
        else:

            pself = type(self)(self.in_channels, self.out_channels, self.kernel_size, self.name, self.stride,
                               self.padding)
            pself.load_state_dict(self.state_dict())
            pself.X = self.X.clone()
            pself.alpha = self.alpha
            # Include positive biases as neurons
            # pself_biases = copy.deepcopy(pself.bias.data)
            pself.bias.data *= 0
            pself.weight.data = torch.max(torch.Tensor([1e-9]), pself.weight)

            nself = type(self)(self.in_channels, self.out_channels, self.kernel_size, self.name, self.stride,
                               self.padding)
            nself.load_state_dict(self.state_dict())
            nself.X = self.X.clone()
            nself.beta = self.beta
            # Include positive biases as neurons
            # nself_biases = copy.deepcopy(nself.bias.data)
            nself.bias.data *= 0
            nself.weight.data = torch.min(torch.Tensor([-1e-9]), nself.weight)

            pX = pself.X + 1e-9
            nX = nself.X + 1e-9

            ZA = pself(pX)
            # expand biases for addition
            # pself_biases = torch.max(torch.Tensor(1).zero_(), pself_biases).view(1, -1, 1, 1).expand_as(ZA)
            # ZA = ZA + pself_biases
            SA = pself.alpha * torch.div(R, ZA)

            ZB = nself(nX)
            # expand biases for addition HERE NEGATIVE BIASES? torch.min???
            # nself_biases = torch.min(torch.Tensor(1).zero_(), nself_biases).view(1, -1, 1, 1).expand_as(ZB)
            # ZB = ZB + nself_biases
            SB = - nself.beta * torch.div(R, ZB)

            C = torch.autograd.grad(ZA, pX, SA)[0] + torch.autograd.grad(ZB, nX, SB)[0]
            R = pX * C

        return R.detach()


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

class VirtualBatchNorm2d(nn.Module):

    def __init__(self, ref_batch, eps: float=1e-5):
        super().__init__(ref_batch, eps)

        self.batch_size = ref_batch.size(0)
        # self.eps = eps
        self.mean = torch.mean(ref_batch, [0, 2])
        self.ref_mean = self.register_parameter('ref_mean', None)
        self.ref_mean_sq = self.register_parameter('ref_mean_sq', None)


class VBN(nn.Module):
    """
    Virtual Batch Normalization
    """

    def __init__(self, num_features, eps=1e-5):
        super(VBN, self).__init__()
        assert isinstance(eps, float)

        # batch statistics
        self.eps = eps
        self.mean = None
        self.mean_sq = None
        self.batch_size = None
        # reference output
        self.reference_output = None
        self.gamma = None
        self.beta = None
        gamma = torch.normal(mean=torch.ones(1, num_features, 1, 1), std=0.02)
        self.gamma = nn.Parameter(gamma.float())
        self.beta = nn.Parameter(torch.FloatTensor(1, num_features, 1, 1).fill_(0))

    def initialize(self, x):
        # compute batch statistics
        mean = x.mean(3, keepdim=True).mean(2, keepdim=True).mean(0, keepdim=True)
        mean_sq = (x**2).mean(3, keepdim=True).mean(2, keepdim=True).mean(0, keepdim=True)
        self.batch_size = x.size(0)
        assert x is not None
        assert mean is not None
        assert mean_sq is not None
        # build detached variables to avoid backprop to graph to compute mean and mean_sq
        # we will manually backprop those in hooks
        self.mean = mean.clone()
        self.mean_sq = mean_sq.clone()
        self.mean.register_hook(lambda grad: mean.backward(grad, retain_graph = True))  # new code
        self.mean_sq.register_hook(lambda grad: mean_sq.backward(grad, retain_graph = True))  # new code
        # compute reference output
        out = self._normalize(x, mean, mean_sq)
        self.reference_output = out.detach_()  # change, just to remove unnecessary saved graph
        return mean, mean_sq

    def get_ref_batch_stats(self):
        return self.mean, self.mean_sq

    def forward(self, x):
        if self.reference_output is None:
            ref_mean, ref_mean_sq = self.initialize(x)
        else:
            ref_mean, ref_mean_sq = self.get_ref_batch_stats()
        new_coeff = 1. / (self.batch_size + 1.)
        old_coeff = 1. - new_coeff
        new_mean = x.mean(3, keepdim=True).mean(2, keepdim=True).mean(0, keepdim=True)
        new_mean_sq = (x**2).mean(3, keepdim=True).mean(2, keepdim=True).mean(0, keepdim=True)
        mean = new_coeff * new_mean + old_coeff * ref_mean  # change
        mean_sq = new_coeff * new_mean_sq + old_coeff * ref_mean_sq  # change
        x = self._normalize(x, mean, mean_sq)
        return x

    def _normalize(self, x, mean, mean_sq):
        assert self.eps is not None
        assert mean_sq is not None
        assert mean is not None
        assert len(x.size()) == 4
        std = torch.sqrt(self.eps + mean_sq - mean**2)
        x = x - mean
        x = x / std
        x = x * self.gamma
        x = x + self.beta
        return x

    def __repr__(self):
        return ('{name}(num_features={num_features}, eps={eps}'.format(
            name=self.__class__.__name__, **self.__dict__))


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
