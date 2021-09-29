# the main file for HDC2021
import os
from glob import glob
from typing import Optional

import cv2
import numpy as np
import torch
from torch.nn.modules import instancenorm
import yaml
from fire import Fire
from tqdm import tqdm
import time
import logging
from utils import from_gdrive_download, get_paths_from_images

import argparse

import re

from aug import get_normalize, get_resize
from models.networks import get_generator

torch.set_num_threads(3)

parser = argparse.ArgumentParser()
parser.add_argument('input_path', type=str, default='/home/jili_cw4/FDC_data/step[0-9]/*/CAM02/focusStep_*_[vt]*009*.tif', help='the input blurred image path')
parser.add_argument('output_path', type=str, default='submit/HDC2021_0-9_sr/', help='output path of the deblurred images')
parser.add_argument('blur_level', type=int, default= None, help='the blur level of input')
parser.add_argument('--gpu', dest='gpu_id', type=str, default='0', help='use gpu or cpu')
parser.add_argument('--side_by_side', action='store_false', default=False, help='put the blur/deblur iamges side-by-side')
opt = parser.parse_args()

# opt.side_by_side = False
def sorted_glob(pattern):
        return sorted(pattern)
# import os
# set gpu/cpu mode
opt.gpu_id = opt.gpu_id if torch.cuda.is_available() else '-1'
if int(opt.gpu_id) >=0:
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.makedirs(opt.output_path, exist_ok=True)
logging.basicConfig(filename= opt.output_path + 'runtime.log',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

imgs = sorted_glob(opt.input_path)
names = sorted([os.path.basename(x) for x in glob(opt.input_path)])

# to check the existences, wether 
if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints',exist_ok=True)
if not os.path.exists(os.path.join('checkpoints', 'hdc2021_0_9_best.h5')) or not os.path.exists(os.path.join('checkpoints', 'hdc2021_10_19_best.h5')):
    from_gdrive_download(save_path='checkpoints')

if not opt.blur_level:
    out = re.findall(r'\d+', names[0])
    out2 = [int(s) for s in out]
    opt.blur_level = out2[1]
        
if opt.blur_level >= 10:
    opt.checkpoint_path = 'checkpoints/hdc2021_10_19_best.h5'
else:
    opt.checkpoint_path = 'checkpoints/hdc2021_0_9_best.h5'

device = 'cuda' if int(opt.gpu_id) >=0 else 'cpu'

class Predictor:
    def __init__(self, weights_path: str, model_name: str = ''):
        with open('config/config_sr_0-9.yaml') as cfg:
            config = yaml.load(cfg)
        model = get_generator(model_name or config['model'])
        model.load_state_dict(torch.load(weights_path, map_location=device)['model'])

        self.model = model.cuda() if int(opt.gpu_id) >= 0 else model
        if isinstance(self.model, torch.nn.DataParallel):
            self.model = self.model.module
        # print(self.model.device)
        # self.model.train(True)
        # # GAN inference should be in train mode to use actual stats in norm layers,
        # # it's not a bug
        self.normalize_fn = get_normalize()
        self.resize = get_resize()
    @staticmethod
    def _array_to_batch(x):
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, 0)
        return torch.from_numpy(x)

    def _preprocess(self, x: np.ndarray, mask: Optional[np.ndarray]):

        x, _ = self.resize(x,x)
        x, _ = self.normalize_fn(x, x)
        if mask is None:
            mask = np.ones_like(x, dtype=np.float32)
        else:
            mask = np.round(mask.astype('float32') / 255)

        h, w, _ = x.shape
        block_size = 32
        min_height = (h // block_size + 1) * block_size
        min_width = (w // block_size + 1) * block_size

        # pad_params = {'mode': 'constant',
        #               'constant_values': 0,
        #               'pad_width': ((0, min_height - h), (0, min_width - w), (0, 0))
        #               }
        # pad_params = {'mode': 'edge',
        #               # 'constant_values': 0,
        #               'pad_width': ((0, min_height - h), (0, min_width - w), (0, 0))
        #               }
        half_h = (min_height - h) // 2
        half_w = (min_width - w) // 2
        pad_params = {'mode': 'reflect',
                      # 'constant_values': 0,
                      'pad_width': ((half_h, min_height - h - half_h), (half_w, min_width - w - half_w), (0, 0))
                      }
        x = np.pad(x, **pad_params)
        mask = np.pad(mask, **pad_params)

        return map(self._array_to_batch, (x, mask)), 2*h, 2*w

    @staticmethod
    def _postprocess(x: torch.Tensor) -> np.ndarray:
        x, = x
        x = x.detach().cpu().float().numpy()
        x = (np.transpose(x, (1, 2, 0)) + 1) / 2.0 * 255.0
        return x.astype('uint8')
    

    def __call__(self, img: np.ndarray, cond: np.ndarray,mask: Optional[np.ndarray], ignore_mask=True) -> np.ndarray:
        (img, mask), h, w = self._preprocess(img, mask)
        # print(img.shape)
        with torch.no_grad():
            # inputs = [img.cuda(), torch.from_numpy(np.expand_dims(cond, 0)).cuda()]
            inputs = [img.to(device), torch.from_numpy(np.expand_dims(cond, 0)).to(device)]
            if not ignore_mask:
                inputs += [mask]
            pred = self.model(*inputs)
            pred = pred[1]
            

        return self._postprocess(pred)[:h, :w, :]



def main(img_pattern: str= opt.input_path,#'/home/jili_cw4/FDC_data/step7/Times/CAM02/focusStep_7_timesR_size_30_sample_009*.tif',#'/home/jili_cw4/FDC_data/Patches/Test/step7/Times_large_img/CAM02/*.png',#'/home/jili_cw4/FDC_data/step7/Times/CAM02/focusStep_7_timesR_size_30_sample_009*.tif',#'/home/jili_cw4/FDC_data/Patches/Test/step7/Times/CAM02/*.png',#step9/Times_large_img/CAM02/*.png',#'/home/jili_cw4/FDC_data/step7/Times/CAM02/focusStep_7_timesR_size_30_sample_009*.tif', #'/home/jili/Downloads/dd_dp_dataset_png/train_c/source/*.png', #/home/jili/real_dataset/*.jpg', #= '/home/jili/GOPRO_Large/train/*/blur/*.png'
         weights_path= opt.checkpoint_path,#'fdc_step7_dp_full_unet_l2_c1_01_last_my_fpn.h5',#'fdc_dp_last_my_fpn.h5',  #'defocus_dp_last_my_fpn.h5', #'best_fpn.h5.ori', #'defocus_best_my_fpn.h5', #best_fpn.h5.ori',
         out_dir= opt.output_path, #'/home/jili_cw4/yangziyi/test_more/',  #/home/jili/SelfDeblur/results/DeblurGAN-v2/Lai_REAL/',
         blur_level = opt.blur_level,
         side_by_side: bool = False):
    
    img_path = get_paths_from_images(img_pattern)
    imgs = sorted_glob(img_path)
    # print(imgs)
    # print(len(imgs))
    # masks = sorted_glob(mask_pattern) if mask_pattern is not None else [None for _ in imgs]
    names = [os.path.basename(x) for x in imgs]
    # print(len(names))
    # print(names)
    predictor = Predictor(weights_path=weights_path)

    os.makedirs(out_dir, exist_ok=True)
    
    for name, f_img in tqdm(zip(names, imgs), total=len(names)):
        # print(f_img)
        img = cv2.imread(f_img)
        # mask = cv2.imread(None)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # out = re.findall(r'\d+', f_img)
        # out2 = [int(s) for s in out]
        if blur_level >= 10:
            blur_level = blur_level-10
        cond = np.array(blur_level)
        # print(type(cond))
        # img = z['image']
        t = time.time()
        # img = rgb2lin_v1(img)
        # print(img.shape)
        # print(cond.shape)
        pred = predictor(img, cond, None)
        elapsed = time.time() - t
        logging.info('names%s time%.5f' % (name, elapsed))
        if side_by_side:
            pred = np.hstack((img, pred))
        # print(pred.shape)
        # c,d = os.path.split(f_img)
        # app = os.sep.join(c.split(os.sep)[-3:])
        # os.makedirs(out_dir+app, exist_ok=True)
        name = os.path.basename(f_img)
        filename = os.path.splitext(name)[0]
        cv2.imwrite(os.path.join(out_dir, filename + '.png'),
                    pred[:,:,0])


if __name__ == '__main__':
    # Fire(main)
    main()