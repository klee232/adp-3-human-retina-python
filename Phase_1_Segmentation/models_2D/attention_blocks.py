# Created by Kuan-Min Lee
# Created date: Nov., 2nd 2025 
# All rights reserved to Leelab.ai
# still need tested

# Brief User Introduction:
# This script is created for storing different customized blocks used in segmentation models (input feature spatial and channel attention blocks)



import math
import torch
import torch.nn as nn
from torch.nn import init




# ----- Channel attention blocks -----
# Brief User Introduction:
# this block is used for channel attention module
# SE (Squeeze-and-Excitation) block
class channel_attention_SE_block(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, act=nn.ReLU):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1  = nn.Conv2d(channels, hidden, 1, bias=True)
        self.act  = act(inplace=True)
        self.fc2  = nn.Conv2d(hidden,  channels, 1, bias=True)
        self.gate = nn.Sigmoid()
        
        
    def forward(self, x):
        w = self.gate(self.fc2(self.act(self.fc1(self.pool(x)))))
        return x * w
        
        

# ECA (Squeeze-and-Excitation) block
class channel_attention_ECA_block(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, adaptive_k: bool = False):
        super().__init__()
        if adaptive_k:
            k = int(abs((math.log2(max(channels, 2)) / 2) + 1))
            kernel_size = k if k % 2 == 1 else k + 1
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.gate = nn.Sigmoid()
        
        
    def forward(self, x):
        y = self.pool(x)                     # [N,C,1,1]
        y = y.squeeze(-1).transpose(1, 2)    # [N,1,C]
        y = self.conv1d(y)                   # [N,1,C]
        y = self.gate(y.transpose(1, 2).unsqueeze(-1))  # [N,C,1,1]
        return x * y



# ----- spatial attention block -----
class spatial_attention_block(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        k = kernel_size; p = k // 2
        # depthwise 7x7 on the 2-channel map, then pointwise to 1
        self.dw = nn.Conv2d(2, 2, k, padding=p, bias=False)
        self.pw = nn.Conv2d(2, 1, 1, bias=False)
        
        
        self.gate = nn.Sigmoid()
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        a = torch.cat([avg, mx], dim=1)   # [N,2,H,W]
        w = self.gate(self.pw(self.dw(a)))
        return x * w
        
 
# spatial squeeze excitation block
class spatial_attention_SE_block(nn.Module):
    """
    sSE: projects channels -> 1 with a 1x1 conv to get a spatial gate.
    Very cheap; often helpful for thin structures (vessels).
    """
    def __init__(self):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1)
        self.gate = nn.Sigmoid()


    def forward(self, x):
        # collapse channels by mean first (to avoid knowing C); then 1x1 refine
        m = torch.mean(x, dim=1, keepdim=True)         # [N,1,H,W]
        w = self.gate(self.conv1x1(m))                 # [N,1,H,W]
        return x * w