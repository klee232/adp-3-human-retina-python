# Created by Kuan-Min Lee
# All rights reserved to Leelab.ai
# passed

# Brief User Introducttion:
# This script includes methods for handling image dataset


import numpy as np



# Created date: Oct. 17th, 2025
# Brief User Introduction:
# This function will convert images into single-channel and store them into a 3D np array. 
# Input Parameter
# image: input image array (np array: [x,y,chn,num_image])
# Output Parameter:
# out_image: processed image array (np array: [x,y,num_image])
def image_SingleChannel_converter(image):
    # convert image to single-channel grayscale image 
    num_row = image.shape[1]
    num_col = image.shape[2]
    num_files = image.shape[0]
    out_image = np.zeros((num_files, num_row, num_col))
    for i_file in range(num_files):
        current_file = image[i_file, :, :, :]
        current_gray_file = np.mean(current_file, axis=-1)
        out_image[i_file, :, :] = current_gray_file
    return out_image



# Created date: Oct. 20th, 2025
# Brief User Introduction:
# this method is created to read safely with the .tif image files
# Input Parameter
# p: path to image files
# Output Parameter:
# np.array of the read .tif images
def image_safe_read_tif(p):
    path = str(p)
    # check the input image type (1-byte(binary) vs 8-byte(0-255))
    import tifffile
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        # if it's a binary image, convert it to 8-bit first and return
        if page.bitspersample == 1:
            raise ValueError("Run matlab convert_ccit_to_uint8.m first")
        else:
            out_img = tifffile.imread(path)
            return out_img