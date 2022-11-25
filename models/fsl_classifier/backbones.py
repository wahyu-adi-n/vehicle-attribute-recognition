
from torchvision.models import efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l, resnet18, resnet34
from operator import __add__
import math
import torch

# First model: Efficientnet v2


def effinet(variant, dropout):
    trunk = []
    if variant == 's':
        net = efficientnet_v2_s(weights='DEFAULT')
    elif variant == 'm':
        net = efficientnet_v2_m(weights='DEFAULT')
    elif variant == 'l':
        net = efficientnet_v2_l(weights='DEFAULT')
    else:
        raise ValueError(
            'the version of the effinet you\'re looking for is not found!'
        )

    if dropout:
        trunk.append(torch.nn.Dropout(dropout))
    trunk.append(torch.nn.Flatten())
    net.classifier = torch.nn.Sequential(*trunk)

    return net


# Second model: Resnet

def resnet(variant, dropout):
    trunk = []

    if variant == '18':
        net = resnet18(weights='DEFAULT')
    elif variant == '34':
        net = resnet34(weights='DEFAULT')
        pass
    else:
        raise ValueError(
            'the version of the resnet you\'re looking for is not found!'
        )

    if dropout:
        trunk.append(torch.nn.Dropout(dropout))
    trunk.append(torch.nn.Flatten())
    net.fc = torch.nn.Sequential(*trunk)

    return net

# Third model: ConvNet


def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, torch.nn.Conv2d):
        n = L.kernel_size[0]*L.kernel_size[1]*L.out_channels
        L.weight.data.normal_(0, math.sqrt(2.0/float(n)))
    elif isinstance(L, torch.nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)


class Conv2d_fw(torch.nn.Conv2d):  # used in MAML to forward input with fast weight
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(Conv2d_fw, self).__init__(in_channels, out_channels,
                                        kernel_size, stride=stride, padding=padding, bias=bias)
        self.weight.fast = None
        if not self.bias is None:
            self.bias.fast = None

    def forward(self, x):
        if self.bias is None:
            if self.weight.fast is not None:
                out = torch.nn.functional.conv2d(
                    x, self.weight.fast, None, stride=self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)
        else:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = torch.nn.functional.conv2d(
                    x, self.weight.fast, self.bias.fast, stride=self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)

        return out


# used in MAML to forward input with fast weight
class BatchNorm2d_fw(torch.nn.BatchNorm2d):
    def __init__(self, num_features):
        super(BatchNorm2d_fw, self).__init__(num_features)
        self.weight.fast = None
        self.bias.fast = None

    def forward(self, x):
        running_mean = torch.zeros(x.data.size()[1])
        running_var = torch.ones(x.data.size()[1])
        if self.weight.fast is not None and self.bias.fast is not None:
            out = torch.nn.functional.batch_norm(
                x, running_mean, running_var, self.weight.fast, self.bias.fast, training=True, momentum=1)
            # batch_norm momentum hack: follow hack of Kate Rakelly in pytorch-maml/src/layers.py
        else:
            out = torch.nn.functional.batch_norm(
                x, running_mean, running_var, self.weight, self.bias, training=True, momentum=1)
        return out


class ConvBlock(torch.nn.Module):
    maml = False  # Default

    def __init__(self, indim, outdim, pool=True, padding=1):
        super(ConvBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        if self.maml:
            self.C = Conv2d_fw(indim, outdim, 3, padding=padding)
            self.BN = BatchNorm2d_fw(outdim)
        else:
            self.C = torch.nn.Conv2d(indim, outdim, 3, padding=padding)
            self.BN = torch.nn.BatchNorm2d(outdim)
        self.relu = torch.nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C, self.BN, self.relu]
        if pool:
            self.pool = torch.nn.MaxPool2d(2)
            self.parametrized_layers.append(self.pool)

        for layer in self.parametrized_layers:
            init_layer(layer)

        self.trunk = torch.nn.Sequential(*self.parametrized_layers)

    def forward(self, x):
        out = self.trunk(x)
        return out


class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class ConvNet(torch.nn.Module):
    def __init__(self, depth, dropout, flatten=True):
        super(ConvNet, self).__init__()
        trunk = []
        for i in range(depth):
            indim = 3 if i == 0 else 64
            outdim = 64
            # only pooling for misal:4 layers
            B = ConvBlock(indim, outdim, pool=(i < 4))
            trunk.append(B)

        if flatten:
            if dropout:
                trunk.append(torch.nn.Dropout(dropout))
            trunk.append(Flatten())

        self.trunk = torch.nn.Sequential(*trunk)
        self.final_feat_dim = 640

    def forward(self, x):
        out = self.trunk(x)
        return out
