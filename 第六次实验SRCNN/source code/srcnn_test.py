'''
Test
'''

from __future__ import print_function
import argparse
import torch
import math
from torch.autograd import Variable
from PIL import Image

from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import cv2

# PSNR compute
def psnr(img1, img2):
    mse = np.mean((img1-img2)**2)
    if mse == 0:
        return float('inf')
    else:
        return 20*np.log10(255/np.sqrt(mse))


#Settings
#Totally 5 test images you can use in this foler.
input_image = './set5/set5/img_005.png'
#model = '/root/wyf/SRCNN/checkpoint/SRCNN_last.pth'    #your model dir
model = '/root/wyf/SRCNN/checkpoint/VDSR_last.pth'    #your model dir
ori_filename = os.path.join('/root/wyf/SRCNN/results', 'img_005LR.png')
output_filename = os.path.join('/root/wyf/SRCNN/results', 'img_005.png')
scale_factor = 3
use_cuda = 1

# Bellowing code supports VDSR model test, if you want to test SRCNN, please change check_point path & test image's channels
img1 = Image.open(input_image)#.convert('YCbCr')
#y, cb, cr = img1.split()

max_size0 = img1.size[0] - (img1.size[0] % scale_factor)
max_size1 = img1.size[1] - (img1.size[1] % scale_factor)
img = img1.crop((0,0,max_size0,max_size1))

img = img.resize((int(img.size[0]//scale_factor),int(img.size[1]//scale_factor)),Image.BICUBIC)
img = img.resize((int(img.size[0]*scale_factor),int(img.size[1]*scale_factor)),Image.BICUBIC)

if not os.path.exists('/root/wyf/SRCNN/results'):
    os.makedirs('/root/wyf/SRCNN/results')
img.save(ori_filename)
img_ori = img
#img = img.convert('YCbCr')
#y, cb, cr = img.split()
#img = y

model = torch.load(model)
input = Variable(ToTensor()(img)).view(1, -1, img.size[1], img.size[0])

if use_cuda:
    model = model.cuda()
    input = input.cuda()

out = model(input)
out = out.cpu()

print ("type = ",type(out))
tt = transforms.ToPILImage()

img_out = tt(out.data[0])

#img_out=Image.merge("YCbCr", (img_out, cb, cr)).convert('RGB')
#img_out.save(output_filename)

out_img_y = out.data[0].numpy()
out_img_y *= 255.0
out_img_y = out_img_y.clip(0, 255)
#out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')
out_img_y = Image.fromarray(np.uint8(out_img_y).transpose(1, 2, 0), mode='RGB')

#out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
#out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
#out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')

out_img = out_img_y
out_img.save(output_filename)

out_img = out_img.resize((img1.size[0], img1.size[1]), Image.BICUBIC)
print(psnr(np.array(out_img).astype(np.float64), np.array(img1).astype(np.float64)))
#img_ori = img_ori.resize((img1.size[0], img1.size[1]), Image.BICUBIC)
#print(psnr(np.array(img_ori).astype(np.float64), np.array(img1).astype(np.float64)))

print('output image saved to ', output_filename)
