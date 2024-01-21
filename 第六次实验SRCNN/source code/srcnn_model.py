import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.datasets as d_sets
from torch.utils.data import DataLoader as d_loader
import matplotlib.pyplot as plt
from PIL import Image

# Do not modify this function without guarantee.
def default_conv(in_channels, out_channels, kernel_size, padding, bias=False, init_scale=0.1):
	basic_conv = nn.Conv2d(in_channels, out_channels, kernel_size,padding=padding, bias=bias)
	nn.init.kaiming_normal_(basic_conv.weight.data, a=0, mode='fan_in')
	basic_conv.weight.data *= init_scale
	if basic_conv.bias is not None:
		basic_conv.bias.data.zero_()
	return basic_conv


class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = default_conv(1, 64, kernel_size=9, padding=9 // 2, bias=False, init_scale=0.1)
        self.conv2 = default_conv(64, 32, kernel_size=1, padding=1 // 2, bias=False, init_scale=0.1)
        self.conv3 = default_conv(32, 1, kernel_size=3, padding=3 // 2, bias=False, init_scale=0.1)
        self.relu = nn.ReLU(inplace=True)
        # Your code here. 
        # For better result, please use 'default_conv(*)' function above to represent conv layer instead of nn.conv2d(*)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        out = self.relu(self.conv3(x))
        return out


'''
1.This is VDSR algorithm, but the follow network is not deep as original paper.
2.After you finish the SRCNN model, you can train the VDSR model bellow to observe the 
  difference between two models' result and training process.
3.You can download this paper by searching keywords 'VDSR super resolution' in Google.
''' 


class VDSR(nn.Module):
    def __init__(self):
        super(VDSR, self).__init__()
        self.conv1 = default_conv(in_channels=3, out_channels=64, kernel_size=3, padding=1, bias=False, init_scale=0.1)
        self.body = nn.Sequential(
                default_conv(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False, init_scale=0.1),
                nn.ReLU(),
                default_conv(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False, init_scale=0.1),
                nn.ReLU(),
                default_conv(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False, init_scale=0.1),
                nn.ReLU(),
                default_conv(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False, init_scale=0.1),
                nn.ReLU(),
                default_conv(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False, init_scale=0.1),
                nn.ReLU(),
                default_conv(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False, init_scale=0.1),
            )
        self.conv3 = default_conv(in_channels=64, out_channels=3, kernel_size=3, padding=1, bias=False, init_scale=0.1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.body(out)
        out = self.conv3(out)

        return out + x


