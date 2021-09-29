import numpy as np
import torch.nn as nn
from skimage.measure import compare_ssim as SSIM

from util.metrics import PSNR


class DeblurModel(nn.Module):
    def __init__(self):
        super(DeblurModel, self).__init__()

    def get_input(self, data):
        img = data['a']
        inputs = img
        targets = data['b']
        # if data['cond']:
        cond = data['cond'] if data.get('cond') is not None else None
        # else:
        #     cond = data['cond']
        inputs, targets, cond = inputs.cuda(), targets.cuda(), cond.cuda()
        return inputs, targets, cond

    def tensor2im(self, image_tensor, imtype=np.uint8):
        image_numpy = image_tensor[0].cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        # image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
        return image_numpy.astype(imtype)

    def get_images_and_metrics(self, inp, output, target) -> (float, float, np.ndarray):
        inp = self.tensor2im(inp)
        fake = self.tensor2im(output.data)
        real = self.tensor2im(target.data)
        psnr = PSNR(fake, real)
        ssim = SSIM(fake, real, multichannel=True)
        vis_img = np.hstack((inp, fake, real))
        return psnr, ssim, vis_img, fake, real

class DeblurModelsr(nn.Module):
    def __init__(self):
        super(DeblurModelsr, self).__init__()

    def get_input(self, data):
        img = data['a']
        inputs = img
        targets = data['b']
        # if data['cond']:
        cond = data['cond'] if data.get('cond') is not None else None
        # else:
        #     cond = data['cond']
        
        # inputs, cond = inputs.cuda(), cond.cuda()
        # targets = [t.cuda() for t in targets]
        return inputs, targets, cond

    def tensor2im(self, image_tensor, imtype=np.uint8):
        image_numpy = image_tensor[0].cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        # image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
        return image_numpy.astype(imtype)

    def get_images_and_metrics(self, inp, output, target) -> (float, float, np.ndarray):
        inp = self.tensor2im(inp)
        fake = self.tensor2im(output[0].data)
        real = self.tensor2im(target[0].data)
        psnr = PSNR(fake, real)
        ssim = SSIM(fake, real, multichannel=True)
        vis_img = np.hstack((inp, fake, real))
        return psnr, ssim, vis_img, fake, real

def get_model(model_config):
    return DeblurModel()

def get_modelsr(model_config):
    return DeblurModelsr()
