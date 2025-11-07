# Created by Kuan-Min Lee
# Created date: Oct, 28th 2025 
# All rights reserved to Leelab.ai
# passed

# Brief User Introducttion:
# This script is constructed to store different loss functions for training neural network. 



import torch
import torch.nn as nn
import torch.nn.functional as F



class TverskyPlusMSE(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=None, # Tversky weights (alpha=beta=0.5 ≈ Dice)
                 lam_tversky=1.0, lam_mse=0.01, eps=1e-7, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lam_t = lam_tversky
        self.lam_m = lam_mse
        self.eps = eps
        self.reduction = reduction


    def forward(self, logits, target):
        # logits: [B,1,H,W] (or [B,H,W]); target: float {0,1}
        if target.dtype != torch.float32 and target.dtype != torch.float64:
            target = target.float()

        # Ensure shapes are [B,1,H,W]
        if logits.dim() == 3:  # [B,H,W]
            logits = logits.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)

        # Tversky components
        dims = (2, 3)  # spatial, per-channel
        TP = (logits * target).sum(dims)
        FP = (logits * (1 - target)).sum(dims)
        FN = ((1 - logits) * target).sum(dims)
        tversky = (TP + self.eps) / (TP + self.alpha * FP + self.beta * FN + self.eps)
        tversky_loss = (1.0 - tversky).mean(dim=1)  # per-channel
        if self.gamma and self.gamma > 1.0:
            tversky_loss = tversky_loss.pow(self.gamma)

        # Reduce
        '''
        if self.reduction == 'mean':
            tversky_loss = tversky_loss.mean()
        elif self.reduction == 'sum':
            tversky_loss = tversky_loss.sum()
        '''
        # else: 'none' keeps vector

        mse_loss_pixel = (logits - target) ** 2
        mse_loss = mse_loss_pixel.mean(dim=(1,2,3))
        '''
        mse_loss = mse_loss_persample.mean()
        '''

        return self.lam_t * tversky_loss + self.lam_m * mse_loss
