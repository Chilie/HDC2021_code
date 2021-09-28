import torch
import torch.nn as nn

# wrappers for convenience
import torch.nn as nn
from torch.nn.init import xavier_normal_, kaiming_normal_
import copy
from functools import partial


def get_weight_init_fn(activation_fn):
    """get weight_initialization function according to activation_fn
    Notes
    -------------------------------------
    if activation_fn requires arguments, use partial() to wrap activation_fn
    """
    fn = activation_fn
    if hasattr(activation_fn, 'func'):
        fn = activation_fn.func

    if fn == nn.LeakyReLU:
        negative_slope = 0
        if hasattr(activation_fn, 'keywords'):
            if activation_fn.keywords.get('negative_slope') is not None:
                negative_slope = activation_fn.keywords['negative_slope']
        if hasattr(activation_fn, 'args'):
            if len(activation_fn.args) > 0:
                negative_slope = activation_fn.args[0]
        return partial(kaiming_normal_, a=negative_slope)
    elif fn == nn.ReLU or fn == nn.PReLU:
        return partial(kaiming_normal_, a=0)
    else:
        return xavier_normal_
    return


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0, activation_fn=None, use_batchnorm=False,
         pre_activation=False, bias=True, weight_init_fn=None):
    """pytorch torch.nn.Conv2d wrapper
    Notes
    ---------------------------------------------------------------------
    Arguments:
        activation_fn : use partial() to wrap activation_fn if any argument is needed
        weight_init_fn : a init function, use partial() to wrap the init function if any argument is needed. default None, if None, auto choose init function according to activation_fn

    examples:
        conv(3,32,3,1,1,activation_fn = partial( torch.nn.LeakyReLU , negative_slope = 0.1 ))
    """
    if not pre_activation and use_batchnorm:
        assert not bias

    layers = []
    if pre_activation:
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(in_channels))
        if activation_fn is not None:
            layers.append(activation_fn())
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
    if weight_init_fn is None:
        weight_init_fn = get_weight_init_fn(activation_fn)
    try:
        weight_init_fn(conv.weight)
    except:
        print(conv.weight)
    layers.append(conv)
    if not pre_activation:
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        if activation_fn is not None:
            layers.append(activation_fn())
    return nn.Sequential(*layers)


def deconv(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, activation_fn=None,
           use_batchnorm=False, pre_activation=False, bias=True, weight_init_fn=None):
    """pytorch torch.nn.ConvTranspose2d wrapper
    Notes
    ---------------------------------------------------------------------
    Arguments:
        activation_fn : use partial() to wrap activation_fn if any argument is needed
        weight_init_fn : a init function, use partial() to wrap the init function if any argument is needed. default None, if None, auto choose init function according to activation_fn

    examples:
        deconv(3,32,3,1,1,activation_fn = partial( torch.nn.LeakyReLU , negative_slope = 0.1 ))

    """
    if not pre_activation and use_batchnorm:
        assert not bias

    layers = []
    if pre_activation:
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(in_channels))
        if activation_fn is not None:
            layers.append(activation_fn())
    deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=bias)
    if weight_init_fn is None:
        weight_init_fn = get_weight_init_fn(activation_fn)
    weight_init_fn(deconv.weight)
    layers.append(deconv)
    if not pre_activation:
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        if activation_fn is not None:
            layers.append(activation_fn())
    return nn.Sequential(*layers)


def linear(in_channels, out_channels, activation_fn=None, use_batchnorm=False, pre_activation=False, bias=True,
           weight_init_fn=None):
    """pytorch torch.nn.Linear wrapper
    Notes
    ---------------------------------------------------------------------
    Arguments:
        activation_fn : use partial() to wrap activation_fn if any argument is needed
        weight_init_fn : a init function, use partial() to wrap the init function if any argument is needed. default None, if None, auto choose init function according to activation_fn

    examples:
        linear(3,32,activation_fn = partial( torch.nn.LeakyReLU , negative_slope = 0.1 ))
    """
    if not pre_activation and use_batchnorm:
        assert not bias

    layers = []
    if pre_activation:
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(in_channels))
        if activation_fn is not None:
            layers.append(activation_fn())
    linear = nn.Linear(in_channels, out_channels)
    if weight_init_fn is None:
        weight_init_fn = get_weight_init_fn(activation_fn)
    weight_init_fn(linear.weight)

    layers.append(linear)
    if not pre_activation:
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        if activation_fn is not None:
            layers.append(activation_fn())
    return nn.Sequential(*layers)


class BasicBlock(nn.Module):
    """pytorch torch.nn.Linear wrapper
    Notes
    ---------------------------------------------------------------------
    use partial() to wrap activation_fn if arguments are needed
    examples:
        BasicBlock(32,32,activation_fn = partial( torch.nn.LeakyReLU , negative_slope = 0.1 , inplace = True ))
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, use_batchnorm=False,
                 activation_fn=partial(nn.ReLU, inplace=True), last_activation_fn=partial(nn.ReLU, inplace=True),
                 pre_activation=False, scaling_factor=1.0):
        super(BasicBlock, self).__init__()
        self.conv1 = conv(in_channels, out_channels, kernel_size, stride, kernel_size // 2, activation_fn,
                          use_batchnorm)
        self.conv2 = conv(out_channels, out_channels, kernel_size, 1, kernel_size // 2, None, use_batchnorm,
                          weight_init_fn=get_weight_init_fn(last_activation_fn))
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = conv(in_channels, out_channels, 1, stride, 0, None, use_batchnorm)
        if last_activation_fn is not None:
            self.last_activation = last_activation_fn()
        else:
            self.last_activation = None
        self.scaling_factor = scaling_factor

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(residual)

        out = self.conv1(x)
        out = self.conv2(out)

        out += residual * self.scaling_factor
        if self.last_activation is not None:
            out = self.last_activation(out)

        return out


def conv5x5_relu(in_channels, out_channels, stride):
    return conv(in_channels, out_channels, 5, stride, 2, activation_fn=partial(nn.ReLU, inplace=True))


def deconv5x5_relu(in_channels, out_channels, stride, output_padding):
    return deconv(in_channels, out_channels, 5, stride, 2, output_padding=output_padding,
                  activation_fn=partial(nn.ReLU, inplace=True))


def resblock(in_channels):
    """Resblock without BN and the last activation
    """
    return BasicBlock(in_channels, out_channels=in_channels, kernel_size=5, stride=1, use_batchnorm=False,
                      activation_fn=partial(nn.ReLU, inplace=True), last_activation_fn=None)


class EBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(type(self), self).__init__()
        self.conv = conv5x5_relu(in_channels, out_channels, stride)
        resblock_list = []
        for i in range(3):
            resblock_list.append(resblock(out_channels))
        self.resblock_stack = nn.Sequential(*resblock_list)

    def forward(self, x):
        x = self.conv(x)
        x = self.resblock_stack(x)
        return x


class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, output_padding):
        super(type(self), self).__init__()
        resblock_list = []
        for i in range(3):
            resblock_list.append(resblock(in_channels))
        self.resblock_stack = nn.Sequential(*resblock_list)
        self.deconv = deconv5x5_relu(in_channels, out_channels, stride, output_padding)

    def forward(self, x):
        x = self.resblock_stack(x)
        x = self.deconv(x)
        return x


class OutBlock(nn.Module):
    def __init__(self, in_channels):
        super(type(self), self).__init__()
        resblock_list = []
        for i in range(3):
            resblock_list.append(resblock(in_channels))
        self.resblock_stack = nn.Sequential(*resblock_list)
        self.conv = conv(in_channels, 3, 5, 1, 2, activation_fn=None)

    def forward(self, x):
        x = self.resblock_stack(x)
        x = self.conv(x)
        return x


class SRNDeblurNet(nn.Module):
    """SRN-DeblurNet
    examples:
        net = SRNDeblurNet()
        y = net( x1 , x2 , x3）#x3 is the coarsest image while x1 is the finest image
    """

    def __init__(self, xavier_init_all=True):
        super(type(self), self).__init__()
        self.inblock = EBlock(3, 32, 1)
        self.eblock1 = EBlock(32, 64, 2)
        self.eblock2 = EBlock(64, 128, 2)
        self.dblock1 = DBlock(128, 64, 2, 1)
        self.dblock2 = DBlock(64, 32, 2, 1)
        self.outblock = OutBlock(32)

        self.input_padding = None
        if xavier_init_all:
            for name, m in self.named_modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                    torch.nn.init.xavier_normal_(m.weight)
                    # print(name)

    def forward(self, x):
        e32 = self.inblock(x)
        e64 = self.eblock1(e32)
        e128 = self.eblock2(e64)
        d64 = self.dblock1(e128)
        d32 = self.dblock2(d64 + e64)
        d3 = self.outblock(d32 + e32)
        return d3