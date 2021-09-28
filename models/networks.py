import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
from models.fpn_mobilenet import FPNMobileNet
from models.fpn_inception import FPNInception
from models.fpn_inception_cconditionalsr import FPNInceptionCCSR
from models.fpn_inception_simple import FPNInceptionSimple
from models.fpn_densenet import FPNDense
###############################################################################
# Functions
###############################################################################

class ConditionalBatchNorm2d(nn.Module):
  def __init__(self, num_features, num_classes, bias=True):
    super().__init__()
    self.num_features = num_features
    self.bias = bias
    self.bn = nn.BatchNorm2d(num_features, affine=False)
    if self.bias:
      self.embed = nn.Embedding(num_classes, num_features * 2)
      self.embed.weight.data[:, :num_features].uniform_()  # Initialise scale at N(1, 0.02)
      self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0
    else:
      self.embed = nn.Embedding(num_classes, num_features)
      self.embed.weight.data.uniform_()

  def forward(self, x, y):
    out = self.bn(x)
    if self.bias:
      gamma, beta = self.embed(y).chunk(2, dim=1)
      out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
    else:
      gamma = self.embed(y)
      out = gamma.view(-1, self.num_features, 1, 1) * out
    return out


class ConditionalInstanceNorm2d(nn.Module):
  def __init__(self, num_features, num_classes=20, bias=True):
    super().__init__()
    self.num_features = num_features
    self.bias = bias
    self.instance_norm = nn.InstanceNorm2d(num_features, affine=False, track_running_stats=False)
    if bias:
      self.embed = nn.Embedding(num_classes, num_features * 2)
      self.embed.weight.data[:, :num_features].uniform_()  # Initialise scale at N(1, 0.02)
      self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0
    else:
      self.embed = nn.Embedding(num_classes, num_features)
      self.embed.weight.data.uniform_()

  def forward(self, x, y):
    h = self.instance_norm(x)
    if self.bias:
      gamma, beta = self.embed(y).chunk(2, dim=-1)
      out = gamma.view(-1, self.num_features, 1, 1) * h + beta.view(-1, self.num_features, 1, 1)
    else:
      gamma = self.embed(y)
      out = gamma.view(-1, self.num_features, 1, 1) * h
    return out


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=True, track_running_stats=True) # False
    elif norm_type == 'conditional-batch':
        norm_layer = ConditionalBatchNorm2d
    elif norm_type == 'conditional-instance':
        norm_layer = ConditionalInstanceNorm2d
    elif norm_type == 'conditional-instance10':
        norm_layer = functools.partial(ConditionalInstanceNorm2d, num_classes=10)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

##############################################################################
# Classes
##############################################################################

def concatenation(input1, input2):
    inputs_shapes2 = [input1.shape[2], input2.shape[2]]
    inputs_shapes3 = [input1.shape[3], input2.shape[3]]

    if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(np.array(inputs_shapes3) == min(inputs_shapes3)):
        inputs_ = [input1, input2]
    else:
        target_shape2 = min(inputs_shapes2)
        target_shape3 = min(inputs_shapes3) # Get the minimal size of the input.

        inputs_ = []
        for inp in [input1, input2]:
            diff2 = (inp.size(2) - target_shape2) // 2
            diff3 = (inp.size(3) - target_shape3) // 2
            inputs_.append(inp[:, :, diff2: diff2 + target_shape2, diff3:diff3 + target_shape3]) # Cut the redundant dimensions

    return torch.cat(inputs_, dim=1)

# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc=1, output_nc=1, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, use_parallel=True, learn_residual=True, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.use_parallel = use_parallel
        self.learn_residual = learn_residual
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        # model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        output = self.model(input)
        if self.learn_residual:
            output = input + output
            output = torch.clamp(output,min = -1,max = 1)
        return output


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class DicsriminatorTail(nn.Module):
    def __init__(self, nf_mult, n_layers, ndf=64, norm_layer=nn.BatchNorm2d, use_parallel=True):
        super(DicsriminatorTail, self).__init__()
        self.use_parallel = use_parallel
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw-1)/2))

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence = [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, norm_layer=nn.BatchNorm2d, use_parallel=True):
        super(MultiScaleDiscriminator, self).__init__()
        self.use_parallel = use_parallel
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        for n in range(1, 3):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        self.scale_one = nn.Sequential(*sequence)
        self.first_tail = DicsriminatorTail(nf_mult=nf_mult, n_layers=3)
        nf_mult_prev = 4
        nf_mult = 8

        self.scale_two = nn.Sequential(
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=2, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True))
        nf_mult_prev = nf_mult
        self.second_tail = DicsriminatorTail(nf_mult=nf_mult, n_layers=4)
        self.scale_three = nn.Sequential(
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True))
        self.third_tail = DicsriminatorTail(nf_mult=nf_mult, n_layers=5)

    def forward(self, input):
        x = self.scale_one(input)
        x_1 = self.first_tail(x)
        x = self.scale_two(x)
        x_2 = self.second_tail(x)
        x = self.scale_three(x)
        x = self.third_tail(x)
        return [x_1, x_2, x]


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, use_parallel=True):
        super(NLayerDiscriminator, self).__init__()
        self.use_parallel = use_parallel
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


def get_fullD(model_config):
    model_d = NLayerDiscriminator(n_layers=5,
                                  # norm_layer=get_norm_layer(norm_type=model_config['norm_layer']),
                                  use_sigmoid=False)
    return model_d


def get_generator(model_config):
    generator_name = model_config['g_name']
    if generator_name == 'resnet':
        model_g = ResnetGenerator(norm_layer=get_norm_layer(norm_type=model_config['norm_layer']),
                                  use_dropout=model_config['dropout'],
                                  n_blocks=model_config['blocks'],
                                  learn_residual=model_config['learn_residual'])
    elif generator_name == 'fpn_mobilenet':
        model_g = FPNMobileNet(norm_layer=get_norm_layer(norm_type=model_config['norm_layer']))
    elif generator_name == 'fpn_inception':
        model_g = FPNInception(norm_layer=get_norm_layer(norm_type=model_config['norm_layer']),output_ch=3)
    elif generator_name == 'fpn_inceptioncc+sr':
        model_g = FPNInceptionCCSR(norm_layer=get_norm_layer(norm_type=model_config['norm_layer']),output_ch=3)
    elif generator_name == 'fpn_inception_simple':
        model_g = FPNInceptionSimple(norm_layer=get_norm_layer(norm_type=model_config['norm_layer']))
    elif generator_name == 'fpn_dense':
        model_g = FPNDense()
    else:
        raise ValueError("Generator Network [%s] not recognized." % generator_name)

    return nn.DataParallel(model_g)


def get_discriminator(model_config):
    discriminator_name = model_config['d_name']
    if discriminator_name == 'no_gan':
        model_d = None
    elif discriminator_name == 'patch_gan':
        model_d = NLayerDiscriminator(n_layers=model_config['d_layers'],
                                      norm_layer=get_norm_layer(norm_type=model_config['norm_layer']),
                                      use_sigmoid=False)
        model_d = nn.DataParallel(model_d)
    elif discriminator_name == 'double_gan':
        patch_gan = NLayerDiscriminator(n_layers=model_config['d_layers'],
                                        # norm_layer=get_norm_layer(norm_type=model_config['norm_layer']),
                                        use_sigmoid=False)
        patch_gan = nn.DataParallel(patch_gan)
        full_gan = get_fullD(model_config)
        full_gan = nn.DataParallel(full_gan)
        model_d = {'patch': patch_gan,
                   'full': full_gan}
    elif discriminator_name == 'multi_scale':
        model_d = MultiScaleDiscriminator(norm_layer=get_norm_layer(norm_type=model_config['norm_layer']))
        model_d = nn.DataParallel(model_d)
    else:
        raise ValueError("Discriminator Network [%s] not recognized." % discriminator_name)

    return model_d


def get_nets(model_config):
    return get_generator(model_config), get_discriminator(model_config)
