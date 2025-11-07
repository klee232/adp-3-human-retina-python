# Created by Kuan-Min Lee
# Created date: Oct, 27th 2025 
# All rights reserved to Leelab.ai
# still need tested

# Brief User Introduction:
# This script is created for storing different customized blocks used in segmentation models


import math
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F



# The library is created for the following reference
# H. Zhang, C. Wu, Z. Zhang, Y. Zhu, H. Lin, Z. Zhang, Y. Sun, T. He, J. Mueller, R. Manmatha, M. Li, & A. Smola, "ResNeSt: Split-Attention Networks,"  In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops , pp.2736-2746
# Source Code Reference:
# Creator: Hang Zhang
# Email: zhanghang0704@gmail.com
# Github: https://github.com/zhanghang1989/ResNeSt?tab=readme-ov-file#pretrained-models
import resnest # import renest blocks for construction
from resnest.torch.splat import SplAtConv2d



# ----- feature enlargement blocks -----
# Brief User Introduction:
# this block is used in multiple models to enlarge the input image to a certain number of features
# original feature enlargement block (also the first convolution)
class first_layer_conv_block(nn.Module):
    # initialize function that initializes the following model
    def __init__(self, num_chn, num_feat):
        super(first_layer_conv_block, self).__init__()  
        # enlarge feature module
        self.enlarge_feat_1 = nn.Sequential(
            nn.Conv2d(num_chn, num_feat, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(num_feat),
            nn.ReLU(inplace=True)
        )
        self.enlarge_feat_2 = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=1, padding=1 ,bias=False),
            # nn.BatchNorm2d(num_feat),
            nn.ReLU(inplace=True) 
        )  
    
    
    # forward function that takes input x and process through the bottleneck
    def forward(self, x):
        output1 = self.enlarge_feat_1(x)
        output2 = self.enlarge_feat_2(output1)
        output = output1 + output2
        return output 
        


# ----- encoder blocks -----
class encoder_block(nn.Module):
    def __init__(self, in_ch, num_feat, num_group=2, num_radix=2, reduction=8):
        super().__init__()
        # construct pre_splat block
        w_mid = max(num_feat // reduction, 1)
        self.pre_splat_conv = nn.Sequential(
            nn.Conv2d(in_ch, w_mid, kernel_size=1, bias=False),
            # nn.BatchNorm2d(w_mid),
            nn.ReLU(inplace=True)
        )
        # construct split attention convolution block
        self.splat_conv = nn.Sequential(
            SplAtConv2d(w_mid, w_mid, kernel_size=3,padding=1,groups=num_group,radix=num_radix,norm_layer=nn.BatchNorm2d),
            nn.ReLU(inplace=True)
        )  
        # construct pro_splat block
        self.post_splat_conv = nn.Sequential(
            nn.Conv2d(w_mid, num_feat, kernel_size=1, bias=False),
            # nn.BatchNorm2d(num_feat)
        )
        # construct residual block
        self.res_block = None
        if in_ch != num_feat:
            self.res_block = nn.Sequential(
                nn.Conv2d(in_ch, num_feat, kernel_size=1, bias=False),
                # nn.BatchNorm2d(num_feat)
            )
        self.res_block_act = nn.ReLU(inplace=True)
        
                
        
    def forward(self,x):
        identity = x
        output = self.pre_splat_conv(x)
        output = self.splat_conv(output)
        output = self.post_splat_conv(output)
        if self.res_block is not None:
            identity = self.res_block(identity)
        output = output + identity
        output = self.res_block_act(output)
        return output



# ----- decoder blocks -----
# --- Standard ResNet bottleneck (post-activation, torchvision style) ---
# Brief User Introduction:
# this block is used in multiple models in decoder branches 
# original resnest decoder block
class UpsampleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mode="nearest"):
        super().__init__()
        self.mode = mode
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
    def forward(self, x, size=None):
        output = F.interpolate(x, scale_factor=2 if size is None else None,
                          size=size, mode=self.mode, align_corners=False if self.mode=="bilinear" else None)
        return self.conv(output)

class decoder_block(nn.Module):
    # initialize function that initializes the following model
    def __init__(self, num_chn, num_feat, num_block):
        super().__init__()
        # 1) flatten concat'd channels down to num_feat (cheap bottleneck)
        self.flat_conv = nn.Sequential(
            nn.Conv2d(num_chn, num_feat, kernel_size=1, bias=False),
            # norm_layer(num_feat),
            nn.ReLU(inplace=True)
        )

        # 2) depthwise-separable residual unit (cheap at high spatial sizes)
        class DWSeparableRes(nn.Module):
            def __init__(self, c, g=8):
                super().__init__()
                self.conv1 = nn.Conv2d(c, c, kernel_size=3, padding=1, groups=g, bias=False)
                self.act1 = nn.ReLU(inplace=True)
                # self.bn1 = norm_layer(c)
                self.conv2 = nn.Conv2d(c, c, kernel_size=3, padding=1, groups=g, bias=False)
                # self.bn2 = norm_layer(c)
                self.act2 = nn.ReLU(inplace=True)
            def forward(self, x):
                output = self.act1(self.conv1(x))
                output = self.conv2(output)
                return self.act2(x+output)

        self.bottleneck_blocks = nn.Sequential(*[DWSeparableRes(num_feat) for _ in range(max(int(num_block), 1))])

        # 3) light smoothing 3x3 (kept name 'splat_conv' to match your forward)
        self.splat_conv = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1, bias=False),
            # norm_layer(num_feat),
            nn.ReLU(inplace=True)
        )
        
              
    
    
    # forward function that takes input x and process through the bottleneck
    def forward(self, x):
        output = self.flat_conv(x)
        output = self.bottleneck_blocks(output)
        output = self.splat_conv(output)
        return output
        
        
        




