# Created by Kuan-Min Lee
# Created date: Nov., 2nd 2025 
# All rights reserved to Leelab.ai
# still need tested

# Brief User Introduction:
# This script is created for storing different customized blocks used in segmentation models (input feature padding, cropping blocks)



import math
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.utils.data import Dataset




# ----- Pad align block -----
class pad_align_block(nn.Module):
    def __init__(self, mode='nearest', align_corners=False):
        super().__init__()
        self.mode = mode
        self.align_corners = align_corners
    
    
    def forward(self, x, ref):
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode=self.mode, align_corners=self.align_corners)
        return x


# ----- cropping block -----
def _pad_if_needed(x, size, pad_mode="reflect"):
    H, W = x.shape[-2], x.shape[-1]
    th, tw = size
    pad_h, pad_w = max(0, th - H), max(0, tw - W)
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode=pad_mode)
    return x

def random_crop_triplet_ROSE1_SVC(img, gt1, gt2, size=(256,256)):
    th, tw = size
    img, gt1, gt2 = _pad_if_needed(img, size), _pad_if_needed(gt1, size), _pad_if_needed(gt2, size)
    H, W = img.shape[-2], img.shape[-1]
    i = torch.randint(0, H - th + 1, (1,)).item()
    j = torch.randint(0, W - tw + 1, (1,)).item()
    return (
        img[..., i:i+th, j:j+tw],
        gt1[..., i:i+th, j:j+tw],
        gt2[..., i:i+th, j:j+tw],
    )
    
def random_crop_triplet(img, gt1, size=(256,256)):
    th, tw = size
    img, gt1 = _pad_if_needed(img, size), _pad_if_needed(gt1, size)
    H, W = img.shape[-2], img.shape[-1]
    i = torch.randint(0, H - th + 1, (1,)).item()
    j = torch.randint(0, W - tw + 1, (1,)).item()
    return (
        img[..., i:i+th, j:j+tw],
        gt1[..., i:i+th, j:j+tw],
    )

def center_crop_triplet(img, *rest, size=(256,256), pad_mode="reflect"):
    """
    Supports:
      center_crop_triplet(img, gt, size=(h,w), pad_mode=...)
      center_crop_triplet(img, gt1, gt2, size=(h,w), pad_mode=...)

    img:  Tensor [C,H,W] or [H,W]
    gts:  list of Tensors [1,H,W] or [H,W]
    returns: (img_c, *gts_c) with same dimensionality convention as inputs
    """
    assert isinstance(img, torch.Tensor), "img must be a torch.Tensor"
    gts = list(rest)

    # Normalize shapes to channel-first
    img_was_2d = (img.ndim == 2)
    if img_was_2d:
        img = img.unsqueeze(0)  # [1,H,W]
    C, H, W = img.shape

    norm_masks = []
    orig_is_2d = []
    for m in gts:
        if m is None:
            norm_masks.append(None)
            orig_is_2d.append(True)
            continue
        assert isinstance(m, torch.Tensor), "mask must be a torch.Tensor"
        orig_is_2d.append(m.ndim == 2)
        if m.ndim == 2:
            m = m.unsqueeze(0)   # [1,H,W]
        elif m.ndim == 3 and m.shape[0] != 1:
            m = m[:1, ...]       # keep 1 channel for mask
        # keep masks binary defensively
        m = (m > 0).to(torch.float32)
        norm_masks.append(m)

    crop_h, crop_w = int(size[0]), int(size[1])

    # Pad if needed
    pad_h = max(0, crop_h - H)
    pad_w = max(0, crop_w - W)
    if pad_h or pad_w:
        pad = (0, pad_w, 0, pad_h)  # (left,right,top,bottom)
        mode = pad_mode if pad_mode in ("constant", "reflect", "replicate") else "reflect"
        img = F.pad(img, pad, mode=mode)
        padded_masks = []
        for m in norm_masks:
            if m is None:
                padded_masks.append(None)
            else:
                padded_masks.append(F.pad(m, pad, mode="constant", value=0.0))
        norm_masks = padded_masks
        _, H, W = img.shape  # update shapes

    # Center crop indices (deterministic)
    top  = max(0, (H - crop_h) // 2)
    left = max(0, (W - crop_w) // 2)

    img_c  = img[:, top:top+crop_h, left:left+crop_w].contiguous()
    masks_c = []
    for m in norm_masks:
        if m is None:
            masks_c.append(None)
        else:
            mc = m[:, top:top+crop_h, left:left+crop_w].contiguous()
            mc = (mc > 0.5).to(torch.float32)  # ensure binary
            masks_c.append(mc)

    # Restore original dimensionality
    if img_was_2d:
        img_c = img_c.squeeze(0)  # [H,W]

    outs = [img_c]
    for was_2d, mc in zip(orig_is_2d, masks_c):
        if mc is None:
            outs.append(None)
        else:
            outs.append(mc.squeeze(0) if was_2d else mc)
    return tuple(outs)

# ----- Cropping block -----
class random_crop_block_ROSE1_SVC(Dataset):
    def __init__(self, base_ds: Dataset, size=(256,256), pad_mode="reflect"):
        self.base = base_ds
        self.size = size
        self.pad_mode = pad_mode
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        img, gt1, gt2 = self.base[idx]
        return random_crop_triplet_ROSE1_SVC(img, gt1, gt2, self.size)
        
     
     
class random_crop_block(Dataset):
    def __init__(self, base_ds: Dataset, size=(256,256), pad_mode="reflect"):
        self.base = base_ds
        self.size = size
        self.pad_mode = pad_mode
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        img, gt = self.base[idx]
        return random_crop_triplet(img, gt, self.size)
            
class val_crop_block_ROSE1_SVC(Dataset):
    """
    Wraps a base dataset that returns (img, gt1, gt2).
    Applies a deterministic center crop with padding if needed.
    """
    def __init__(self, base_ds: Dataset, size=(256,256), pad_mode="reflect"):
        self.base = base_ds
        self.size = size
        self.pad_mode = pad_mode

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, gt1, gt2 = self.base[idx]
        return center_crop_triplet(img, gt1, gt2, size=self.size, pad_mode=self.pad_mode)


class val_crop_block(Dataset):
    """
    Wraps a base dataset that returns (img, gt).
    Applies a deterministic center crop with padding if needed.
    """
    def __init__(self, base_ds: Dataset, size=(256,256), pad_mode="reflect"):
        self.base = base_ds
        self.size = size
        self.pad_mode = pad_mode

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, gt = self.base[idx]
        return center_crop_triplet(img, gt, size=self.size, pad_mode=self.pad_mode)