# Created by Kuan-Min Lee
# Created date: Oct, 27th 2025 
# All rights reserved to Leelab.ai
# still need tested

# Brief User Introduction:
# This script is created for storing different SVC coarse stage models



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
from .attention_blocks import channel_attention_SE_block, spatial_attention_block



# ----- models ----- #
# Brief User Introduction:
# this model is equivalent to coarse prototype 4 for SVC layer.
# I only picked up the best overall model to replicate in python.
class SVC_coarse_prototype_net_1(nn.Module):
    def __init__(self):
        super(SVC_coarse_prototype_net_1, self).__init__()
        # ----- Encoder part ------
        # ----- layer 0 -----
        # construct feature map enlarge module (output feature map: (h/2, w/2, 64))
        num_chn = 1
        num_out_feat = 64
        self.enlarge_feat_block = first_layer_conv_block(num_chn, num_out_feat) # (h/2, w/2, 64)
        # downsample module
        self.downsample = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # construct encoder modules
        '''        
        backbone = resnest50(pretrained=False)
        '''
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
        self.downsample2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2) # (h/8, w/8, 512)
        )
        # ----- layer 3 (output feature map: (h/16, h/16, 1024)) -----
        chn_in = 512
        num_feat = 1024
        self.encoder3 = encoder_block(chn_in, num_feat) # (h/8, w/8, 1024)
        self.downsample3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2) # (h/16, w/16, 1024)
        )
        # ----- layer 4 (output feature map: (h/16, w/16, 2048)) -----
        chn_in = 1024
        num_feat = 2048
        self.encoder4 = encoder_block(chn_in, num_feat) # (h/16, w/16, 2048)
        
        # ----- Decoder part -----
        # ----- thick branch -----
        # construct dencoder modules (thick vessel)  
        # layer 3 (output feature map: (h/8, w/8, 1024))
        # setup upsampling module
        num_chn = 2048
        num_feat = 1024
        num_block = 1
        # upsample module
        self.thick_upsample3 = UpsampleConv(num_chn, num_feat)
        self.thick_padblock3 = pad_align_block()
        self.thick_chnatten3 = channel_attention_SE_block(num_chn)
        self.thick_spatten3 = spatial_attention_block(kernel_size=7)
        self.thick_decoder3 = decoder_block(num_chn, num_feat, num_block)
        # layer 2 (output feature map: (h/4, w/4, 512))
        # setup upsampling module
        num_chn = 1024
        num_feat = 512
        num_block = 1
        self.thick_upsample2 = UpsampleConv(num_chn, num_feat)
        self.thick_padblock2 = pad_align_block()
        self.thick_chnatten2 = channel_attention_SE_block(num_chn)
        self.thick_spatten2 = spatial_attention_block(kernel_size=7)
        self.thick_decoder2 = decoder_block(num_chn, num_feat, num_block)
        # layer 1 (output feature map: (h/2, w/2, 256))
        # setup upsampling module
        num_chn = 512
        num_feat = 256
        num_block = 1
        self.thick_upsample1 = UpsampleConv(num_chn, num_feat)
        self.thick_padblock1 = pad_align_block()
        self.thick_chnatten1 = channel_attention_SE_block(num_chn)
        # self.thick_spatten1 = spatial_attention_block(kernel_size=7)
        self.thick_decoder1 = decoder_block(num_chn, num_feat, num_block)
        # layer 0 (output feature map: (h, w, 64))
        num_chn = 256
        num_in_chn = 128
        num_feat = 64
        self.thick_upsample = UpsampleConv(num_chn, num_feat)
        self.thick_padblock = pad_align_block()
        self.thick_chnatten = channel_attention_SE_block(num_in_chn)
        # self.thick_spatten = spatial_attention_block(kernel_size=7)
        self.thick_first_flat_block = first_layer_conv_block(num_in_chn, num_feat) # (h/2, w/2, 64)
        # final flatten layer (output feature map: (h, w, 1))
        num_chn = 64
        num_feat = 1
        self.thick_final_flat_block = nn.Sequential(
            nn.Conv2d(num_chn, num_feat, kernel_size=1, stride=1, padding='same', bias=True),
            nn.Sigmoid()
        )
        
        # ----- thin branch -----
        # construct dencoder modules (thin vessel)                
        # layer 2
        # setup upsampling module
        num_chn = 1024
        num_feat = 512
        num_block = 1
        self.thin_upsample2 = UpsampleConv(num_chn, num_feat)
        self.thin_padblock2 = pad_align_block()
        self.thin_chnatten2 = channel_attention_SE_block(num_chn)
        self.thin_spatten2 = spatial_attention_block(kernel_size=7)     
        self.thin_decoder2 = decoder_block(num_chn, num_feat, num_block)
        # layer 1
        # setup upsampling module
        num_chn = 512
        num_feat = 256
        num_block = 1
        self.thin_upsample1 = UpsampleConv(num_chn, num_feat)
        self.thin_padblock1 = pad_align_block()
        self.thin_chnatten1 = channel_attention_SE_block(num_chn)
        # self.thin_spatten1 = spatial_attention_block(kernel_size=7)
        self.thin_decoder1 = decoder_block(num_chn, num_feat, num_block)
        # layer 0
        num_chn = 256
        num_in_chn = 128
        num_feat = 64
        self.thin_upsample = UpsampleConv(num_chn, num_feat)
        self.thin_padblock = pad_align_block()
        self.thin_chnatten = channel_attention_SE_block(num_in_chn)
        # self.thin_spatten = spatial_attention_block(kernel_size=7)
        self.thin_first_flat_block = first_layer_conv_block(num_in_chn, num_feat) # (h/2, w/2, 64)
        # final flatten layer
        num_chn = 64
        num_feat = 1
        self.thin_final_flat_block = nn.Sequential(
            nn.Conv2d(num_chn, num_feat, kernel_size=3, stride=1, padding='same', bias=True),
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
        output_l2 = self.encoder2(output)
        output = self.downsample2(output_l2)
        # level 3
        output_l3 = self.encoder3(output)
        output = self.downsample3(output_l3)
        # level 4
        output = self.encoder4(output)
        
        # decoder part
        # thick vessel branch
        # level 3
        thick_output = self.thick_upsample3(output)
        thick_output = self.thick_padblock3(thick_output, output_l3)
        thick_output = torch.cat((thick_output, output_l3), dim=1).contiguous(memory_format=torch.channels_last)
        thick_output = self.thick_chnatten3(thick_output)
        thick_output = self.thick_spatten3(thick_output)
        thick_output = self.thick_decoder3(thick_output)
        '''
        output = F.interpolate(output, scale_factor=2, mode='nearest')
        thick_output = self.thick_decoder3(torch.cat([output, output_l3], dim=1))
        '''
        # level 2
        thick_output = self.thick_upsample2(thick_output)
        thick_output = self.thick_padblock2(thick_output, output_l2)
        thick_output = torch.cat((thick_output, output_l2), dim=1).contiguous(memory_format=torch.channels_last)
        thick_output = self.thick_chnatten2(thick_output)
        thick_output = self.thick_spatten2(thick_output)
        thick_output = self.thick_decoder2(thick_output)
        '''
        thick_output = F.interpolate(thick_output, scale_factor=2, mode='nearest')
        thick_output = self.thick_decoder2(torch.cat([thick_output, output_l2], dim=1))
        '''
        # level 1
        thick_output = self.thick_upsample1(thick_output)
        thick_output = self.thick_padblock1(thick_output, output_l1)
        thick_output = torch.cat((thick_output, output_l1), dim=1).contiguous(memory_format=torch.channels_last)
        thick_output = self.thick_chnatten1(thick_output)
        # thick_output = self.thick_spatten1(thick_output)
        thick_output = self.thick_decoder1(thick_output)
        '''
        thick_output = F.interpolate(thick_output, scale_factor=2, mode='nearest')
        thick_output = self.thick_decoder1(torch.cat([thick_output, output_l1], dim=1))
        '''
        # level 0
        thick_output = self.thick_upsample(thick_output)
        thick_output = self.thick_padblock(thick_output, output_l0)
        thick_output = torch.cat((thick_output, output_l0), dim=1).contiguous(memory_format=torch.channels_last)
        thick_output = self.thick_chnatten(thick_output)
        # thick_output = self.thick_spatten(thick_output)
        thick_output = self.thick_first_flat_block(thick_output)
        # flatten
        thick_output = self.thick_final_flat_block(thick_output)
        
        # thin vessel branch
        # level 2
        thin_output = self.thin_upsample2(output_l3)
        thin_output = self.thin_padblock2(thin_output, output_l2)       
        thin_output = torch.cat((thin_output, output_l2), dim=1).contiguous(memory_format=torch.channels_last)
        thin_output = self.thin_chnatten2(thin_output)
        thin_output = self.thin_spatten2(thin_output)
        thin_output = self.thin_decoder2(thin_output)
        '''
        thin_output = F.interpolate(output_l3, scale_factor=2, mode='nearest')
        thin_output = self.thin_decoder2(torch.cat([thin_output, output_l2], dim=1))
        '''
        # level 1
        thin_output = self.thin_upsample1(thin_output)
        thin_output = self.thin_padblock1(thin_output, output_l1)       
        thin_output = torch.cat((thin_output, output_l1), dim=1).contiguous(memory_format=torch.channels_last)
        thin_output = self.thin_chnatten1(thin_output)
        # thin_output = self.thin_spatten1(thin_output)
        thin_output = self.thin_decoder1(thin_output)
        '''
        thin_output = F.interpolate(thin_output, scale_factor=2, mode='nearest')
        thin_output = self.thin_decoder1(torch.cat([thin_output, output_l1], dim=1))
        '''
        # level 0
        thin_output = self.thin_upsample(thin_output)
        thin_output = self.thin_padblock(thin_output, output_l0)       
        thin_output = torch.cat((thin_output, output_l0), dim=1).contiguous(memory_format=torch.channels_last)
        thin_output = self.thin_chnatten(thin_output) 
        # thin_output = self.thin_spatten(thin_output)
        thin_output = self.thin_first_flat_block(thin_output)
        # flatten
        thin_output = self.thin_final_flat_block(thin_output)
        
        return thick_output, thin_output
        
        
        
        
# Brief User Introduction:
# this model is equivalent to coarse prototype 4 for SVC layer.
# I only picked up the best overall model to replicate in python.
class SVC_coarse_prototype_net_2(nn.Module):
    def __init__(self):
        super(SVC_coarse_prototype_net_1, self).__init__()
        # ----- Encoder part ------
        # ----- layer 0 -----
        # construct feature map enlarge module (output feature map: (h/2, w/2, 64))
        num_chn = 1
        num_out_feat = 64
        self.enlarge_feat_block = first_layer_conv_block(num_chn, num_out_feat) # (h/2, w/2, 64)
        # downsample module
        self.downsample = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # construct encoder modules
        '''        
        backbone = resnest50(pretrained=False)
        '''
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
        self.downsample2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2) # (h/8, w/8, 512)
        )
        # ----- layer 3 (output feature map: (h/16, h/16, 1024)) -----
        chn_in = 512
        num_feat = 1024
        self.encoder3 = encoder_block(chn_in, num_feat) # (h/8, w/8, 1024)
        self.downsample3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2) # (h/16, w/16, 1024)
        )
        # ----- layer 4 (output feature map: (h/16, w/16, 2048)) -----
        chn_in = 1024
        num_feat = 2048
        self.encoder4 = encoder_block(chn_in, num_feat) # (h/16, w/16, 2048)
        
        # ----- Decoder part -----
        # ----- thick branch -----
        # construct dencoder modules (thick vessel)  
        # layer 3 (output feature map: (h/8, w/8, 1024))
        # setup upsampling module
        num_chn = 2048
        num_feat = 1024
        num_block = 1
        # upsample module
        self.thick_upsample3 = UpsampleConv(num_chn, num_feat)
        self.thick_padblock3 = pad_align_block()
        self.thick_chnatten3 = channel_attention_SE_block(num_chn)
        self.thick_spatten3 = spatial_attention_block(kernel_size=7)
        self.thick_decoder3 = decoder_block(num_chn, num_feat, num_block)
        # layer 2 (output feature map: (h/4, w/4, 512))
        # setup upsampling module
        num_chn = 1024
        num_feat = 512
        num_block = 1
        self.thick_upsample2 = UpsampleConv(num_chn, num_feat)
        self.thick_padblock2 = pad_align_block()
        self.thick_chnatten2 = channel_attention_SE_block(num_chn)
        self.thick_spatten2 = spatial_attention_block(kernel_size=7)
        self.thick_decoder2 = decoder_block(num_chn, num_feat, num_block)
        # layer 1 (output feature map: (h/2, w/2, 256))
        # setup upsampling module
        num_chn = 512
        num_feat = 256
        num_block = 1
        self.thick_upsample1 = UpsampleConv(num_chn, num_feat)
        self.thick_padblock1 = pad_align_block()
        self.thick_chnatten1 = channel_attention_SE_block(num_chn)
        # self.thick_spatten1 = spatial_attention_block(kernel_size=7)
        self.thick_decoder1 = decoder_block(num_chn, num_feat, num_block)
        # layer 0 (output feature map: (h, w, 64))
        num_chn = 256
        num_in_chn = 128
        num_feat = 64
        self.thick_upsample = UpsampleConv(num_chn, num_feat)
        self.thick_padblock = pad_align_block()
        self.thick_chnatten = channel_attention_SE_block(num_in_chn)
        # self.thick_spatten = spatial_attention_block(kernel_size=7)
        self.thick_first_flat_block = first_layer_conv_block(num_in_chn, num_feat) # (h/2, w/2, 64)
        # final flatten layer (output feature map: (h, w, 1))
        num_chn = 64
        num_feat = 1
        self.thick_final_flat_block = nn.Sequential(
            nn.Conv2d(num_chn, num_feat, kernel_size=1, stride=1, padding='same', bias=True),
            nn.Sigmoid()
        )
        
        # ----- thin branch -----
        # construct dencoder modules (thin vessel)                
        # layer 2
        # setup upsampling module
        num_chn = 1024
        num_feat = 512
        num_block = 1
        self.thin_upsample2 = UpsampleConv(num_chn, num_feat)
        self.thin_padblock2 = pad_align_block()
        self.thin_chnatten2 = channel_attention_SE_block(num_chn)
        self.thin_spatten2 = spatial_attention_block(kernel_size=7)     
        self.thin_decoder2 = decoder_block(num_chn, num_feat, num_block)
        # layer 1
        # setup upsampling module
        num_chn = 512
        num_feat = 256
        num_block = 1
        self.thin_upsample1 = UpsampleConv(num_chn, num_feat)
        self.thin_padblock1 = pad_align_block()
        self.thin_chnatten1 = channel_attention_SE_block(num_chn)
        # self.thin_spatten1 = spatial_attention_block(kernel_size=7)
        self.thin_decoder1 = decoder_block(num_chn, num_feat, num_block)
        # layer 0
        num_chn = 256
        num_in_chn = 128
        num_feat = 64
        self.thin_upsample = UpsampleConv(num_chn, num_feat)
        self.thin_padblock = pad_align_block()
        self.thin_chnatten = channel_attention_SE_block(num_in_chn)
        # self.thin_spatten = spatial_attention_block(kernel_size=7)
        self.thin_first_flat_block = first_layer_conv_block(num_in_chn, num_feat) # (h/2, w/2, 64)
        # final flatten layer
        num_chn = 64
        num_feat = 1
        self.thin_final_flat_block = nn.Sequential(
            nn.Conv2d(num_chn, num_feat, kernel_size=3, stride=1, padding='same', bias=True),
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
        output_l2 = self.encoder2(output)
        output = self.downsample2(output_l2)
        # level 3
        output_l3 = self.encoder3(output)
        output = self.downsample3(output_l3)
        # level 4
        output = self.encoder4(output)
        
        # decoder part
        # thick vessel branch
        # level 3
        thick_output = self.thick_upsample3(output)
        thick_output = self.thick_padblock3(thick_output, output_l3)
        thick_output = torch.cat((thick_output, output_l3), dim=1).contiguous(memory_format=torch.channels_last)
        thick_output = self.thick_chnatten3(thick_output)
        thick_output = self.thick_spatten3(thick_output)
        thick_output = self.thick_decoder3(thick_output)
        '''
        output = F.interpolate(output, scale_factor=2, mode='nearest')
        thick_output = self.thick_decoder3(torch.cat([output, output_l3], dim=1))
        '''
        # level 2
        thick_output = self.thick_upsample2(thick_output)
        thick_output = self.thick_padblock2(thick_output, output_l2)
        thick_output = torch.cat((thick_output, output_l2), dim=1).contiguous(memory_format=torch.channels_last)
        thick_output = self.thick_chnatten2(thick_output)
        thick_output = self.thick_spatten2(thick_output)
        thick_output = self.thick_decoder2(thick_output)
        '''
        thick_output = F.interpolate(thick_output, scale_factor=2, mode='nearest')
        thick_output = self.thick_decoder2(torch.cat([thick_output, output_l2], dim=1))
        '''
        # level 1
        thick_output = self.thick_upsample1(thick_output)
        thick_output = self.thick_padblock1(thick_output, output_l1)
        thick_output = torch.cat((thick_output, output_l1), dim=1).contiguous(memory_format=torch.channels_last)
        thick_output = self.thick_chnatten1(thick_output)
        # thick_output = self.thick_spatten1(thick_output)
        thick_output = self.thick_decoder1(thick_output)
        '''
        thick_output = F.interpolate(thick_output, scale_factor=2, mode='nearest')
        thick_output = self.thick_decoder1(torch.cat([thick_output, output_l1], dim=1))
        '''
        # level 0
        thick_output = self.thick_upsample(thick_output)
        thick_output = self.thick_padblock(thick_output, output_l0)
        thick_output = torch.cat((thick_output, output_l0), dim=1).contiguous(memory_format=torch.channels_last)
        thick_output = self.thick_chnatten(thick_output)
        # thick_output = self.thick_spatten(thick_output)
        thick_output = self.thick_first_flat_block(thick_output)
        # flatten
        thick_output = self.thick_final_flat_block(thick_output)
        
        # thin vessel branch
        # level 2
        thin_output = self.thin_upsample2(output_l3)
        thin_output = self.thin_padblock2(thin_output, output_l2)       
        thin_output = torch.cat((thin_output, output_l2), dim=1).contiguous(memory_format=torch.channels_last)
        thin_output = self.thin_chnatten2(thin_output)
        thin_output = self.thin_spatten2(thin_output)
        thin_output = self.thin_decoder2(thin_output)
        '''
        thin_output = F.interpolate(output_l3, scale_factor=2, mode='nearest')
        thin_output = self.thin_decoder2(torch.cat([thin_output, output_l2], dim=1))
        '''
        # level 1
        thin_output = self.thin_upsample1(thin_output)
        thin_output = self.thin_padblock1(thin_output, output_l1)       
        thin_output = torch.cat((thin_output, output_l1), dim=1).contiguous(memory_format=torch.channels_last)
        thin_output = self.thin_chnatten1(thin_output)
        # thin_output = self.thin_spatten1(thin_output)
        thin_output = self.thin_decoder1(thin_output)
        '''
        thin_output = F.interpolate(thin_output, scale_factor=2, mode='nearest')
        thin_output = self.thin_decoder1(torch.cat([thin_output, output_l1], dim=1))
        '''
        # level 0
        thin_output = self.thin_upsample(thin_output)
        thin_output = self.thin_padblock(thin_output, output_l0)       
        thin_output = torch.cat((thin_output, output_l0), dim=1).contiguous(memory_format=torch.channels_last)
        thin_output = self.thin_chnatten(thin_output) 
        # thin_output = self.thin_spatten(thin_output)
        thin_output = self.thin_first_flat_block(thin_output)
        # flatten
        thin_output = self.thin_final_flat_block(thin_output)
        
        return thick_output, thin_output