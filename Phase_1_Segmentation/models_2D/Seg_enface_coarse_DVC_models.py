# Created by Kuan-Min Lee
# Created date: Oct, 24th 2025 
# All rights reserved to Leelab.ai
# still need tested

# Brief User Introduction:
# This script is created for constructing different DVC coarse stage models



import torch # for model construction
import torch.nn as nn
from torch.nn import init
from torch.utils.data import TensorDataset, DataLoader
import numpy as np # for variable storage construction
import matplotlib.pyplot as plt # for plotting


# import random cropping
from .input_feature_blocks import random_crop_block_ROSE1_SVC

import os
import time

# import resume chacking function
from .resume_checking import CheckpointManager
from .losses import TverskyPlusMSE


'''
# The library is created for the following reference
# H. Zhang, C. Wu, Z. Zhang, Y. Zhu, H. Lin, Z. Zhang, Y. Sun, T. He, J. Mueller, R. Manmatha, M. Li, & A. Smola, "ResNeSt: Split-Attention Networks,"  In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops , pp.2736-2746
# Source Code Reference:
# Creator: Hang Zhang
# Email: zhanghang0704@gmail.com
# Github: https://github.com/zhanghang1989/ResNeSt?tab=readme-ov-file#pretrained-models
import resnest # import renest blocks for construction
'''



# customized library
from .input_feature_blocks import pad_align_block
from .model_blocks import first_layer_conv_block, encoder_block, UpsampleConv, decoder_block
from .attention_blocks import channel_attention_SE_block



# ----- models ----- #
# Brief User Introduction:
# this model is equivalent to coarse prototype 9 for DVC layer.
# I only picked up the best overall model to replicate in python.
class DVC_coarse_prototype_net_1(nn.Module):
    def __init__(self):
        super(DVC_coarse_prototype_net_1, self).__init__()
        # ----- Encoder part -----
        # ----- layer 0 -----
        # construct feature map enlarge module (output feature map: (h/2, w/2, 64))
        num_chn = 1
        num_out_feat = 64
        self.enlarge_feat_block = first_layer_conv_block(num_chn, num_out_feat) # (h/2, w/2, 64)
        # downsample module
        self.downsample = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # ----- layer 1 (output feature map: (h/4, w/4, 256)) -----
        chn_in = 64
        num_feat = 256
        self.encoder1 = encoder_block(chn_in, num_feat) # (h/2, w/2, 256)
        self.downsample1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2) # (h/4, w/4, 256)
        )
        # ----- layer 2 (output feature map: (h/8, w/8, 512)) -----
        chn_in = 256
        num_feat = 512
        self.encoder2 = encoder_block(chn_in, num_feat) # (h/4, w/4, 512)
        
        # ----- Decoder part -----      
        # ----- layer 1 (output feature map: (h/2, w/2, 256)) -----
        # setup upsampling module
        num_chn = 512
        num_feat = 256
        num_block = 1
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(num_chn, num_feat, kernel_size=2, stride=2)
        )
        self.padblock1 = pad_align_block()
        self.chnatten1 = channel_attention_SE_block(num_chn)
        # self.spatten1 = spatial_attention_block(kernel_size=7)
        self.decoder1 = decoder_block(num_chn, num_feat, num_block)
        # ----- layer 0 (output feature map: (h, w, 64)) -----
        num_chn = 256
        num_in_chn = 128
        num_feat = 64
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(num_chn, num_feat, kernel_size=2, stride=2)
        )
        self.padblock = pad_align_block()
        self.chnatten = channel_attention_SE_block(num_in_chn)
        # self.spatten = spatial_attention_block(kernel_size=7)
        self.first_flat_block = first_layer_conv_block(num_in_chn, num_feat) # (h/2, w/2, 64)
        # final flatten layer (output feature map: (h, w, 1))
        num_chn = 64
        num_feat = 1
        self.final_flat_block = nn.Sequential(
            nn.Conv2d(num_chn, num_feat, kernel_size=1, stride=1, padding='same', bias=True),
            nn.Sigmoid()
        )
            
        
    def forward(self, x):
        # encoder part
        # level 0
        output_l0 = self.enlarge_feat_block(x)
        output = self.downsample(output_l0)
        # level 1
        output_l1 = self.encoder1(output)
        output = self.downsample1(output_l1)
        # level 2
        output = self.encoder2(output)
        
        # decoder part
        # level 1
        output = self.upsample1(output)
        output = self.padblock1(output, output_l1)
        output = torch.cat((output, output_l1), dim=1).contiguous(memory_format=torch.channels_last)
        output = self.chnatten1(output)
        # output = self.spatten1(output)
        output = self.decoder1(output)
        # level 0
        output = self.upsample(output)
        output = self.padblock(output, output_l0)
        output = torch.cat((output, output_l0), dim=1).contiguous(memory_format=torch.channels_last)
        output = self.chnatten(output)
        # output = self.spatten(output)
        output = self.first_flat_block(output)
        # flatten
        output = self.final_flat_block(output)
        
        return output