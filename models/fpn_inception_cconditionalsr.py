import torch
import torch.nn as nn
import yaml
from pretrainedmodels import inceptionresnetv2
from torchsummary import summary
import torch.nn.functional as F


class FPNHead(nn.Module):
    def __init__(self, num_in, num_mid, num_out):
        super().__init__()
        self.block0 = nn.Conv2d(num_in, num_mid, kernel_size=3, padding=1, bias=False)
        self.embed0 = nn.Embedding(20, num_mid)
        self.embed0.weight.data.uniform_()  # Initialise scale at N(1, 0.02)
        self.block1 = nn.Conv2d(num_mid, num_out, kernel_size=3, padding=1, bias=False)
        self.embed1 = nn.Embedding(20, num_out)
        self.embed1.weight.data.uniform_()  # Initialise scale at N(1, 0.02)
        self.num_mid = num_mid
        self.num_out = num_out

    def forward(self, x, y):
        x = self.block0(x)
        gamma = self.embed0(y)
        x = gamma.view(-1, self.num_mid, 1, 1) * x
        x = nn.functional.relu(x, inplace=True)
        x = self.block1(x)
        gamma = self.embed1(y)
        x = gamma.view(-1, self.num_out, 1, 1) * x
        x = nn.functional.relu(x, inplace=True)
        return x

class ConvBlockC(nn.Module):
    def __init__(self, num_in, num_out, norm_layer):
        super().__init__()
        self.conv = nn.Conv2d(num_in, num_out, kernel_size=3, padding=1, bias=False)
        self.embed = nn.Embedding(20, num_out * 2)
        self.embed.weight.data[:, :num_out].uniform_()  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_out:].zero_()  # Initialise bias at 0
        self.num_out = num_out
        self.norm = norm_layer(num_out)
        self.act = nn.ReLU(inplace=True)
        # self.block = nn.Sequential(nn.Conv2d(num_in, num_out, kernel_size=3, padding=1),
        #                          norm_layer(num_out),
        #                          nn.ReLU(inplace=True))

    def forward(self, x, y):
        x = self.conv(x)
        gamma, beta = self.embed(y).chunk(2, dim=-1)
        x = gamma.view(-1, self.num_out, 1, 1) * x + beta.view(-1, self.num_out, 1, 1)
        x = self.norm(x,y)
        x = self.act(x)
        return x


class FPNInceptionCC(nn.Module):

    def __init__(self, norm_layer, output_ch=3, num_filters=128, num_filters_fpn=256):
        super().__init__()

        # Feature Pyramid Network (FPN) with four feature maps of resolutions
        # 1/4, 1/8, 1/16, 1/32 and `num_filters` filters for all feature maps.
        self.fpn = FPNC(num_filters=num_filters_fpn, norm_layer=norm_layer)

        # The segmentation heads on top of the FPN

        self.head1 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head2 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head3 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head4 = FPNHead(num_filters_fpn, num_filters, num_filters)

        self.smooth = ConvBlockC(4 * num_filters, num_filters, norm_layer)
        # self.smooth = nn.Sequential(
        #     nn.Conv2d(4 * num_filters, num_filters, kernel_size=3, padding=1),
        #     norm_layer(num_filters),
        #     nn.ReLU(),
        # )

        self.smooth2 = ConvBlockC(num_filters, num_filters // 2, norm_layer)
        # self.smooth2 = nn.Sequential(
        #     nn.Conv2d(num_filters, num_filters // 2, kernel_size=3, padding=1),
        #     norm_layer(num_filters // 2),
        #     nn.ReLU(),
        # )

        self.final = nn.Conv2d(num_filters // 2, output_ch, kernel_size=3, padding=1)

    def unfreeze(self):
        self.fpn.unfreeze()

    def forward(self, x, cond):
        # y = x.repeat(1,3,1,1)
        map0, map1, map2, map3, map4 = self.fpn(x, cond)

        map4 = nn.functional.upsample(self.head4(map4, cond), scale_factor=8, mode="bilinear") #nearest
        map3 = nn.functional.upsample(self.head3(map3, cond), scale_factor=4, mode="bilinear")
        map2 = nn.functional.upsample(self.head2(map2, cond), scale_factor=2, mode="bilinear")
        map1 = nn.functional.upsample(self.head1(map1, cond), scale_factor=1, mode="bilinear")

        smoothed = self.smooth(torch.cat([map4, map3, map2, map1], dim=1), cond)
        smoothed = nn.functional.upsample(smoothed, scale_factor=2, mode="bilinear")
        smoothed = self.smooth2(smoothed + map0, cond)
        smoothed = nn.functional.upsample(smoothed, scale_factor=2, mode="bilinear")

        final = self.final(smoothed)
        res = torch.tanh(final) + x
        #
        # return torch.clamp(res, min = -1,max = 1)
        # res = final + x

        return torch.clamp(res, min = -1,max = 1)

class FPNInceptionCCSR(nn.Module):

    def __init__(self, norm_layer, output_ch=3, num_filters=128, num_filters_fpn=256):
        super().__init__()

        self.fpninceptioncc = FPNInceptionCC(norm_layer=norm_layer,output_ch=output_ch)
        # self.fpninceptioncc = self.fpninceptioncc.cuda()
        
        # The segmentation heads on top of the FPN
        # with open('config/config_sr_0-9.yaml') as cfg:
        #     config = yaml.load(cfg)
        # state_dict = torch.load(config['model']['weights_path'])['model']  # 模型可以保存为pth文件，也可以为pt文件。
        # # create new OrderedDict that does not contain `module.`
        # from collections import OrderedDict
        # new_state_dict = OrderedDict()
        # for k, v in state_dict.items():
        #     name = k[7:]  # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
        #     new_state_dict[name] = v  # 新字典的key值对应的value为一一对应的值。
        # # load params
        # # model.load_state_dict(new_state_dict)  # 从新加载这个模型。
        # self.fpninceptioncc.load_state_dict(new_state_dict)

        self.fpninceptioncc.train(True)
        self.smooth = ConvBlockC(output_ch, num_filters, norm_layer)
        # self.smooth = nn.Sequential(
        #     nn.Conv2d(4 * num_filters, num_filters, kernel_size=3, padding=1),
        #     norm_layer(num_filters),
        #     nn.ReLU(),
        # )

        self.smooth2 = ConvBlockC(num_filters, num_filters // 2, norm_layer)
        # self.smooth2 = nn.Sequential(
        #     nn.Conv2d(num_filters, num_filters // 2, kernel_size=3, padding=1),
        #     norm_layer(num_filters // 2),
        #     nn.ReLU(),
        # )

        self.final = nn.Conv2d(num_filters // 2, output_ch, kernel_size=3, padding=1)
        for param in self.fpninceptioncc.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.fpninceptioncc.parameters():
            param.requires_grad = True

    def forward(self, x, cond):
        out = self.fpninceptioncc(x, cond)
        out_up = nn.functional.upsample(out, scale_factor=2, mode="bilinear")

        out_up1 = self.smooth(out_up, cond)
        out_up1 = self.smooth2(out_up1, cond)

        final = self.final(out_up1)
        res = torch.tanh(final) + out_up
        #
        # return torch.clamp(res, min = -1,max = 1)
        # res = final + x

        return [out, torch.clamp(res, min = -1,max = 1)]

class FPNC(nn.Module):

    def __init__(self, norm_layer, num_filters=256):
        """Creates an `FPN` instance for feature extraction.
        Args:
          num_filters: the number of filters in each output pyramid level
          pretrained: use ImageNet pre-trained backbone feature extractor
        """

        super().__init__()
        self.inception = inceptionresnetv2(num_classes=1000, pretrained=None) #'imagenet')

        self.enc0 = self.inception.conv2d_1a
        self.enc1 = nn.Sequential(
            self.inception.conv2d_2a,
            self.inception.conv2d_2b,
            # nn.MaxPool2d(kernel_size=3, stride=1),
            # antialiased_cnns.BlurPool(64, stride=2),#]
            self.inception.maxpool_3a,
        ) # 64
        self.enc2 = nn.Sequential(
            self.inception.conv2d_3b,
            self.inception.conv2d_4a,
            self.inception.maxpool_5a,
        )  # 192
        self.enc3 = nn.Sequential(
            self.inception.mixed_5b,
            self.inception.repeat,
            self.inception.mixed_6a,
        )   # 1088
        self.enc4 = nn.Sequential(
            self.inception.repeat_1,
            self.inception.mixed_7a,
        ) #2080

        self.td1 = ConvBlockC(num_filters, num_filters, norm_layer)
        self.td2 = ConvBlockC(num_filters, num_filters, norm_layer)
        self.td3 = ConvBlockC(num_filters, num_filters, norm_layer)
        # self.td1 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
        #                          norm_layer(num_filters),
        #                          nn.ReLU(inplace=True))
        # self.td2 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
        #                          norm_layer(num_filters),
        #                          nn.ReLU(inplace=True))
        # self.td3 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
        #                          norm_layer(num_filters),
        #                          nn.ReLU(inplace=True))
        self.pad = nn.ReflectionPad2d(1)
        self.lateral4 = nn.Conv2d(2080, num_filters, kernel_size=1, bias=False)
        self.lateral3 = nn.Conv2d(1088, num_filters, kernel_size=1, bias=False)
        self.lateral2 = nn.Conv2d(192, num_filters, kernel_size=1, bias=False)
        self.lateral1 = nn.Conv2d(64, num_filters, kernel_size=1, bias=False)
        self.lateral0 = nn.Conv2d(32, num_filters // 2, kernel_size=1, bias=False)

        for param in self.inception.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.inception.parameters():
            param.requires_grad = True

    def forward(self, x, cond):

        # Bottom-up pathway, from ResNet
        enc0 = self.enc0(x)

        enc1 = self.enc1(enc0) # 256

        enc2 = self.enc2(enc1) # 512

        enc3 = self.enc3(enc2) # 1024

        enc4 = self.enc4(enc3) # 2048

        # Lateral connections

        lateral4 = self.pad(self.lateral4(enc4))
        lateral3 = self.pad(self.lateral3(enc3))
        lateral2 = self.lateral2(enc2)
        lateral1 = self.pad(self.lateral1(enc1))
        lateral0 = self.lateral0(enc0)

        # Top-down pathway
        pad = (1, 2, 1, 2)  # pad last dim by 1 on each side
        pad1 = (0, 1, 0, 1)
        map4 = lateral4
        map3 = self.td1(lateral3 + nn.functional.upsample(map4, scale_factor=2, mode="bilinear"),cond)  # bilinear , nearest
        map2 = self.td2(F.pad(lateral2, pad, "reflect") + nn.functional.upsample(map3, scale_factor=2, mode="bilinear"),cond)
        map1 = self.td3(lateral1 + nn.functional.upsample(map2, scale_factor=2, mode="bilinear"), cond)
        return F.pad(lateral0, pad1, "reflect"), map1, map2, map3, map4

        # map4 = lateral4
        # map3 = self.td1(lateral3 + nn.functional.upsample(map4, scale_factor=2, mode="nearest"), cond) # bilinear , nearest
        # map2 = self.td2(F.pad(lateral2, pad, "reflect") + nn.functional.upsample(map3, scale_factor=2, mode="bilinear"), cond)
        # map1 = self.td3(lateral1 + nn.functional.upsample(map2, scale_factor=2, mode="nearest"), cond)
        # return F.pad(lateral0, pad1, "reflect"), map1, map2, map3, map4
