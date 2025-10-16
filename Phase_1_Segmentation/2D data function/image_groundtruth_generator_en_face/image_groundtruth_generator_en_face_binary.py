# Created by Kuan-Min Lee
# Created date: Oct. 15th, 2025

# Brief User Introduction:
# This module utilize a global thresholding method of creating a
# first-phase binarization result for the input 3D image (OCTA or OCT
# image)

# 3D image array format: [num_z_slice, num_x_coord, num_y_coord]

# Input parameter:
# surface_deep_image: OCTA image with only surface and deep capillary layer (3D image array)
# mask: mask for segmentation layer (3D mask array)
# surf_thres_ratio: threshold for binarization for surface capillary layer
# deep_thres_ratio: threshold for binarization for deep capillary layer

# Output:
# binary_surf_data_image: outcome of binarization image (binarized 3D image) 

# import necessary
from skimage.filters import threshold_otsu
import numpy as np

def image_groundtruth_generator_en_face_binary_global()