# Created by Kuan-Min Lee
# All rights reserved to Leelab.ai
# passed

# Brief User Introducttion:
# This script includes the augmentations methods for rose data



import numpy as np # for array computation (including output arrays, array dimension)




# Created date: Oct. 16th, 2025
# passed
# Brief User Introduction:
# This function will conduct rotation for data augmentation. 
# Input Parameter
# num_aug: number of data augmentation (2*1 integer array with one smaller value and one bigger valeu)
# angles: maximum angle for rotation (double)
# image: input image array (np array: [num_image,x,y])
# goundtruth: input groundtruth array (np array: [num_image,x,y])
# Output Parameter:
# out_img: processed image array (np array: [num_image,x,y])
# out_gt: processed groundtruth array (np array: [num_image,x,y])
def img_rotator(num_aug, angles, image, groundtruth):
    import cv2 # for image rotation, image flipping
    
    # check if the image is single channel
    # if it is not, convert it to single-channel grayscale image first
    if image.ndim != 3:
        from image_processors import image_SingleChannel_converter # import single channel converter if necessary
        converted_image = image_SingleChannel_converter(image)
    # if it's, just stay still with input image
    else:
        converted_image = image
          
    # check if the groundtruth is single channel
    if groundtruth.ndim != 3:
        from image_processors import image_SingleChannel_converter # import single channel converter if necessary
        converted_gt = image_SingleChannel_converter(groundtruth)
    # if it's, just stay still with input image
    else:
        converted_gt = groundtruth
        
        
    # check if the groundtruth is boolean
    if converted_gt.dtype == bool:
        converted_gt = converted_gt.astype(np.uint8) * 255
    # if it's, just stay still with input image
    else:
        converted_gt = converted_gt
        
        
    # conduct angle rotation
    # generate angles for rotation 
    # check if the input array has two distinct values
    unique_vals = np.unique(angles)
    # if the angles doesn't have exact two distince values, raise an error message
    if len(unique_vals) != 2:
        raise ValueError("Incorrected angles input found. Please insert an array with two distinct values for angles")
    # only conduct rotation when given two distince angle values
    else:
        # generate angles for rotation
        upper_angle = angles.max()
        lower_angle = angles.min()
        rotation_angles = np.linspace(lower_angle, upper_angle, num_aug)
        rotation_angles = np.concatenate(([0], rotation_angles)) 
        
        # conduct angle rotation 
        num_row = image.shape[1]
        num_col = image.shape[2]
        num_files = image.shape[0]
        out_img = np.zeros((num_files*(num_aug+1), num_row, num_col)) # it's num_aug+1, because I stored the original image + the assigned number of augmentation 
                                                                      # (for example, If I had three augmentation angles, I will have output as [x, y, (num_aug+1)])
        out_gt = np.zeros((num_files*(num_aug+1), num_row, num_col)) 
        for i_file in range(num_files):
            # grab out current file
            current_file = converted_image[i_file, :, :]
            current_gt = converted_gt[i_file, :, :]
            
            # conduct angle rotation: 0: original
            for i_angle in range(num_aug+1):
                angle = rotation_angles[i_angle]
                M = cv2.getRotationMatrix2D((num_row/2.0, num_col/2.0), angle, 1.0) # retrieve rotational matrix
                rotated_current_file = cv2.warpAffine(current_file, M, (int(num_row), int(num_col)), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                rotated_current_gt = cv2.warpAffine(current_gt, M, (int(num_row), int(num_col)), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                out_img[i_angle+(i_file)*(num_aug+1), :, :] = rotated_current_file
                out_gt[i_angle+(i_file)*(num_aug+1), :, :] = rotated_current_gt
                
                
    return out_img, out_gt
    
    
    
# Created date: Oct. 17th, 2025
# passed
# Brief User Introduction:
# This function will conduct image flipping for data augmentation. 
# Input Parameter
# image: input image array (np array: [num_image,x,y])
# goundtruth: input groundtruth array (np array: [num_image,x,y])
# Output Parameter:
# out_img: processed image array (np array: [num_image,x,y])
# out_gt: processed groundtruth array (np array: [num_image,x,y])
def img_flipper(image, groundtruth):
    import cv2 # for image rotation, image flipping
    
    
    # check if the image is single channel
    # if it is not, convert it to single-channel grayscale image first
    if image.ndim != 3:
        from image_processors import image_SingleChannel_converter # import single channel converter if necessary
        converted_image = image_SingleChannel_converter(image)
    # if it's, just stay still with input image
    else:
        converted_image = image
    
    # check if the groundtruth is single channel
    if groundtruth.ndim != 3:
        from image_processors import image_SingleChannel_converter # import single channel converter if necessary
        converted_gt = image_SingleChannel_converter(groundtruth)
    # if it's, just stay still with input image
    else:
        converted_gt = groundtruth  
        
        
    # check if the groundtruth is boolean
    if converted_gt.dtype == bool:
        converted_gt = converted_gt.astype(np.uint8) * 255
    # if it's, just stay still with input image
    else:
        converted_gt = converted_gt
    
    
    # conduct image flipping
    num_row = image.shape[1]
    num_col = image.shape[2]
    num_files = image.shape[0]
    out_img = np.zeros((num_files*(2+1), num_row, num_col)) # here, I constructed to conduct two flipping (horizontal and vertical). 
                                                            # So in the end, I will have output as [x, y, (2+1)]
    out_gt = np.zeros((num_files*(2+1), num_row, num_col))        
    for i_file in range(num_files):
        # grab out current file
        current_file = converted_image[i_file, :, :]
        current_gt = converted_gt[i_file, :, :]
        
        # augmentation 0: original image, augmentation 1: flip in vertical, augmentation 2: flip in horizontal
        for i_aug in range(2+1):
            if i_aug == 0:
                flipped_current_file = current_file
                flipped_current_gt = current_gt
            else:
                flipped_current_file = np.flip(current_file, axis=(i_aug-1))
                flipped_current_gt = np.flip(current_gt, axis=(i_aug-1))
            out_img[i_aug+(i_file)*(2+1), :, :] = flipped_current_file
            out_gt[i_aug+(i_file)*(2+1), :, :] = flipped_current_gt


    return out_img, out_gt
    
    
    
# Created date: Oct. 17th, 2025
# passed
# Brief User Introduction:
# This function will conduct image elastic deformation for data augmentation. 
# Input Parameter
# image: input image array (np array: [num_image,x,y])
# Output Parameter:
# out_img: processed image array (np array: [num_image,x,y])
# out_gt: processed groundtruth array (np array: [num_image,x,y])
def img_elastic_deformer(image, groundtruth):
    # for image elastic deformation
    import albumentations as A

    
    # check if the image is single channel
    # if it is not, convert it to single-channel grayscale image first
    if image.ndim != 3:
        from image_processors import image_SingleChannel_converter # import single channel converter if necessary
        converted_image = image_SingleChannel_converter(image)
    # if it's, just stay still with input image
    else:
        converted_image = image

    # check if the groundtruth is single channel
    if groundtruth.ndim != 3:
        from image_processors import image_SingleChannel_converter # import single channel converter if necessary
        converted_gt = image_SingleChannel_converter(groundtruth)
    # if it's, just stay still with input image
    else:
        converted_gt = groundtruth    


    # check if the groundtruth is boolean
    if converted_gt.dtype == bool:
        converted_gt = converted_gt.astype(np.uint8) * 255
    # if it's, just stay still with input image
    else:
        converted_gt = converted_gt
    
    
    # conduct image elastic deformation
    aug = A.Compose([
    A.ElasticTransform(alpha=40, sigma=6, 
                       interpolation=1, border_mode=0, p=0.5)  # zeros outside
    ])
    num_row = image.shape[1]
    num_col = image.shape[2]
    num_files = image.shape[0]
    out_img = np.zeros((num_files*(1+1), num_row, num_col)) # like in the image flipping, this time, I only conduct one augmentation (elastic deformation).
                                                            # so, its outut as [x, y, (1+1)]
    out_gt = np.zeros((num_files*(1+1), num_row, num_col))        
    for i_file in range(num_files):
        # grab out current file
        current_file = converted_image[i_file, :, :]
        current_gt = converted_gt[i_file, :, :]
        
        # augmentation 0: original, augmentation: deformed image
        for i_aug in range(1+1):
            if i_aug == 0:
                deformed_current_file = current_file
                deformed_current_gt = current_gt
            else:
                deformed_current_file = aug(image=current_file)["image"]
                deformed_current_gt = aug(image=current_gt)["image"]
            out_img[i_aug+(i_file)*(1+1), :, :] = deformed_current_file
            out_gt[i_aug+(i_file)*(1+1), :, :] = deformed_current_gt


    return out_img, out_gt
    
    
    
# Created date: Oct. 17th, 2025
# passed
# Brief User Introduction:
# This function will conduct image contrast jitter for data augmentation. 
# Input Parameter
# image: input image array (np array: [num_image,x,y])
# Output Parameter:
# out_img: processed image array (np array: [num_image,x,y])   
# out_gt: processed groundtruth array (np array: [num_image,x,y])
def img_contrast_jitter(image, groundtruth):
    from PIL import Image, ImageEnhance # for image contrast jitter 
    
    
    # check if the image is single channel
    # if it is not, convert it to single-channel grayscale image first
    if image.ndim != 3: 
        from image_processors import image_SingleChannel_converter # import single channel converter if necessary
        converted_image = image_SingleChannel_converter(image)
    # if it's, just stay still with input image
    else:
        converted_image = image

    # check if the groundtruth is single channel
    if groundtruth.ndim != 3:
        from image_processors import image_SingleChannel_converter # import single channel converter if necessary
        converted_gt = image_SingleChannel_converter(groundtruth)
    # if it's, just stay still with input image
    else:
        converted_gt = groundtruth     


    # check if the groundtruth is boolean
    if converted_gt.dtype == bool:
        converted_gt = converted_gt.astype(np.uint8) * 255
    # if it's, just stay still with input image
    else:
        converted_gt = converted_gt      
    
    
    # conduct image contrast jitter
    num_row = image.shape[1]
    num_col = image.shape[2]
    num_files = image.shape[0]
    out_img = np.zeros((num_files*(2+1), num_row, num_col)) # like in the image flipping, this time, I conduct two augmentation (higher contrast and lower contrast).
    out_gt = np.zeros((num_files*(2+1), num_row, num_col))        
    for i_file in range(num_files):
        # grab out current file
        current_file = converted_image[i_file, :, :]
        current_gt = converted_gt[i_file, :, :]
        
        # setup enhancer for images
        enhancer = ImageEnhance.Contrast(Image.fromarray(current_file.astype(np.uint8)))
        
        # augmentation 0: original, augmentation 1: enhanced contrast, augmentation 2: decreased contrast
        for i_aug in range(2+1):
            if i_aug == 0:
                out_img[i_aug+(i_file)*(2+1), :, :] = current_file
            elif i_aug == 1:
                out_img[i_aug+(i_file)*(2+1), :, :] = np.array(enhancer.enhance(1.5), dtype=np.uint8)
            else:
                out_img[i_aug+(i_file)*(2+1), :, :] = np.array(enhancer.enhance(0.8), dtype=np.uint8)
            out_gt[i_aug+(i_file)*(2+1), :, :] = current_gt 

            
    return out_img, out_gt
