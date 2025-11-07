# Created by Kuan-Min Lee
# All rights reserved to Leelab.ai
# still need tested

# Brief User Introducttion:
# This script includes the augmentations methods for rose data



import numpy as np # for array computation (including output arrays, array dimension)




# Created date: Oct. 21st, 2025
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
def img_rotator_ROSEO(num_aug, angles, SVC_image, DVC_image, IVC_image, FAZ_groundtruth, junct_groundtruth, vessel_groundtruth):
    import cv2 # for image rotation, image flipping
    
    
    # check if the image is single channel
    # if it is not, convert it to single-channel grayscale image first
    if SVC_image.ndim != 3:
        from image_processors import image_SingleChannel_converter # import single channel converter if necessary
        converted_SVC_image = image_SingleChannel_converter(SVC_image)
    # if it's, just stay still with input image
    else:
        converted_SVC_image = SVC_image
        
    # check if the image is single channel
    # if it is not, convert it to single-channel grayscale image first
    if DVC_image.ndim != 3:
        from image_processors import image_SingleChannel_converter # import single channel converter if necessary
        converted_DVC_image = image_SingleChannel_converter(DVC_image)
    # if it's, just stay still with input image
    else:
        converted_DVC_image = DVC_image
        
    # check if the image is single channel
    # if it is not, convert it to single-channel grayscale image first
    if IVC_image.ndim != 3:
        from image_processors import image_SingleChannel_converter # import single channel converter if necessary
        converted_IVC_image = image_SingleChannel_converter(IVC_image)
    # if it's, just stay still with input image
    else:
        converted_IVC_image = IVC_image
    
    # check if the groundtruth is single channel
    if FAZ_groundtruth.ndim != 3:
        from image_processors import image_SingleChannel_converter # import single channel converter if necessary
        converted_FAZ_gt = image_SingleChannel_converter(FAZ_groundtruth)
    # if it's, just stay still with input image
    else:
        converted_FAZ_gt = FAZ_groundtruth
    
    # check if the thick groundtruth is single channel
    if junct_groundtruth.ndim != 3:
        from image_processors import image_SingleChannel_converter # import single channel converter if necessary
        converted_junct_gt = image_SingleChannel_converter(junct_groundtruth)
    # if it's, just stay still with input image
    else:
        converted_junct_gt = junct_groundtruth 

    # check if the thin groundtruth is single channel
    if vessel_groundtruth.ndim != 3:
        from image_processors import image_SingleChannel_converter # import single channel converter if necessary
        converted_vessel_gt = image_SingleChannel_converter(vessel_groundtruth)
    # if it's, just stay still with input image
    else:
        converted_vessel_gt = vessel_groundtruth           
        
        
    # check if the groundtruth is boolean
    if converted_FAZ_gt.dtype == bool:
        converted_FAZ_gt = converted_FAZ_gt.astype(np.uint8) * 255
    # if it's, just stay still with input image
    else:
        converted_FAZ_gt = converted_FAZ_gt
        
    # check if the groundtruth is boolean
    if converted_junct_gt.dtype == bool:
        converted_junct_gt = converted_junct_gt.astype(np.uint8) * 255
    # if it's, just stay still with input image
    else:
        converted_junct_gt = converted_junct_gt
        
    # check if the groundtruth is boolean
    if converted_vessel_gt.dtype == bool:
        converted_vessel_gt = converted_vessel_gt.astype(np.uint8) * 255
    # if it's, just stay still with input image
    else:
        converted_vessel_gt = converted_vessel_gt

        
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
        num_row = SVC_image.shape[1]
        num_col = SVC_image.shape[2]
        num_files = SVC_image.shape[0]
        out_SVC_img = np.zeros((num_files*(num_aug+1), num_row, num_col)) # it's num_aug+1, because I stored the original image + the assigned number of augmentation 
                                                                          # (for example, If I had three augmentation angles, I will have output as [x, y, (num_aug+1)])
        out_DVC_img = np.zeros((num_files*(num_aug+1), num_row, num_col)) 
        out_IVC_img = np.zeros((num_files*(num_aug+1), num_row, num_col)) 
        out_FAZGt = np.zeros((num_files*(num_aug+1), num_row, num_col)) 
        out_junctGt = np.zeros((num_files*(num_aug+1), num_row, num_col)) 
        out_vesselGt = np.zeros((num_files*(num_aug+1), num_row, num_col)) 
        for i_file in range(num_files):
            # grab out current file
            current_SVC_file = converted_SVC_image[i_file, :, :]
            current_DVC_file = converted_DVC_image[i_file, :, :]
            current_IVC_file = converted_IVC_image[i_file, :, :]
            current_FAZgt = converted_FAZ_gt[i_file, :, :]
            current_FAZgt = converted_junct_gt[i_file, :, :]
            current_FAZgt = converted_FAZ_gt[i_file, :, :]
            current_junctgt = converted_junct_gt[i_file, :, :]
            current_vesselgt = converted_vessel_gt[i_file, :, :]
            
            # conduct angle rotation: 0: original
            for i_angle in range(num_aug+1):
                angle = rotation_angles[i_angle]
                M = cv2.getRotationMatrix2D((num_row/2.0, num_col/2.0), angle, 1.0) # retrieve rotational matrix
                rotated_current_SVC_file = cv2.warpAffine(current_SVC_file, M, (int(num_row), int(num_col)), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                rotated_current_DVC_file = cv2.warpAffine(current_DVC_file, M, (int(num_row), int(num_col)), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                rotated_current_IVC_file = cv2.warpAffine(current_IVC_file, M, (int(num_row), int(num_col)), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                rotated_current_FAZgt = cv2.warpAffine(current_FAZgt, M, (int(num_row), int(num_col)), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                rotated_current_junctgt = cv2.warpAffine(current_junctgt, M, (int(num_row), int(num_col)), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                rotated_current_vesselgt = cv2.warpAffine(current_vesselgt, M, (int(num_row), int(num_col)), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                out_SVC_img[i_angle+(i_file)*(num_aug+1), :, :] = rotated_current_SVC_file
                out_DVC_img[i_angle+(i_file)*(num_aug+1), :, :] = rotated_current_DVC_file
                out_IVC_img[i_angle+(i_file)*(num_aug+1), :, :] = rotated_current_IVC_file
                out_FAZGt[i_angle+(i_file)*(num_aug+1), :, :] = rotated_current_FAZgt
                out_junctGt[i_angle+(i_file)*(num_aug+1), :, :] = rotated_current_junctgt
                out_vesselGt[i_angle+(i_file)*(num_aug+1), :, :] = rotated_current_vesselgt

                
    return out_SVC_img, out_DVC_img, out_IVC_img, out_FAZGt, out_junctGt, out_vesselGt
    
    
    
# Created date: Oct. 21st, 2025
# passed
# Brief User Introduction:
# This function will conduct image flipping for data augmentation. 
# Input Parameter
# image: input image array (np array: [num_image,x,y])
# goundtruth: input groundtruth array (np array: [num_image,x,y])
# Output Parameter:
# out_img: processed image array (np array: [num_image,x,y])
# out_gt: processed groundtruth array (np array: [num_image,x,y])
def img_flipper_ROSEO(SVC_image, DVC_image, IVC_image, FAZ_groundtruth, junct_groundtruth, vessel_groundtruth):
    import cv2 # for image rotation, image flipping
    
    
    # check if the image is single channel
    # if it is not, convert it to single-channel grayscale image first
    if SVC_image.ndim != 3:
        from image_processors import image_SingleChannel_converter # import single channel converter if necessary
        converted_SVC_image = image_SingleChannel_converter(SVC_image)
    # if it's, just stay still with input image
    else:
        converted_SVC_image = SVC_image
        
    # check if the image is single channel
    # if it is not, convert it to single-channel grayscale image first
    if DVC_image.ndim != 3:
        from image_processors import image_SingleChannel_converter # import single channel converter if necessary
        converted_DVC_image = image_SingleChannel_converter(DVC_image)
    # if it's, just stay still with input image
    else:
        converted_DVC_image = DVC_image
    
    # check if the image is single channel
    # if it is not, convert it to single-channel grayscale image first
    if IVC_image.ndim != 3:
        from image_processors import image_SingleChannel_converter # import single channel converter if necessary
        converted_IVC_image = image_SingleChannel_converter(IVC_image)
    # if it's, just stay still with input image
    else:
        converted_IVC_image = IVC_image
    
    # check if the groundtruth is single channel
    if FAZ_groundtruth.ndim != 3:
        from image_processors import image_SingleChannel_converter # import single channel converter if necessary
        converted_FAZ_gt = image_SingleChannel_converter(FAZ_groundtruth)
    # if it's, just stay still with input image
    else:
        converted_FAZ_gt = FAZ_groundtruth 

    # check if the groundtruth is single channel
    if junct_groundtruth.ndim != 3:
        from image_processors import image_SingleChannel_converter # import single channel converter if necessary
        converted_junct_gt = image_SingleChannel_converter(junct_groundtruth)
    # if it's, just stay still with input image
    else:
        converted_junct_gt = junct_groundtruth         
    
    # check if the groundtruth is single channel
    if vessel_groundtruth.ndim != 3:
        from image_processors import image_SingleChannel_converter # import single channel converter if necessary
        converted_vessel_gt = image_SingleChannel_converter(vessel_groundtruth)
    # if it's, just stay still with input image
    else:
        converted_vessel_gt = vessel_groundtruth 
        
        
    # check if the groundtruth is boolean
    if converted_FAZ_gt.dtype == bool:
        converted_FAZ_gt = converted_FAZ_gt.astype(np.uint8) * 255
    # if it's, just stay still with input image
    else:
        converted_FAZ_gt = converted_FAZ_gt
        
    # check if the groundtruth is boolean
    if converted_junct_gt.dtype == bool:
        converted_junct_gt = converted_junct_gt.astype(np.uint8) * 255
    # if it's, just stay still with input image
    else:
        converted_junct_gt = converted_junct_gt
        
    # check if the groundtruth is boolean
    if converted_vessel_gt.dtype == bool:
        converted_vessel_gt = converted_vessel_gt.astype(np.uint8) * 255
    # if it's, just stay still with input image
    else:
        converted_vessel_gt = converted_vessel_gt    
        
        
    # conduct image flipping
    num_row = SVC_image.shape[1]
    num_col = SVC_image.shape[2]
    num_files = SVC_image.shape[0]
    out_SVC_img = np.zeros((num_files*(2+1), num_row, num_col)) # here, I constructed to conduct two flipping (horizontal and vertical). 
                                                                # So in the end, I will have output as [x, y, (2+1)]
    out_DVC_img = np.zeros((num_files*(2+1), num_row, num_col)) 
    out_IVC_img = np.zeros((num_files*(2+1), num_row, num_col)) 
    out_FAZGt = np.zeros((num_files*(2+1), num_row, num_col))   
    out_junctGt = np.zeros((num_files*(2+1), num_row, num_col))        
    out_vesselGt = np.zeros((num_files*(2+1), num_row, num_col))            
    for i_file in range(num_files):
        # grab out current file
        current_SVC_file = converted_SVC_image[i_file, :, :]
        current_DVC_file = converted_DVC_image[i_file, :, :]
        current_IVC_file = converted_IVC_image[i_file, :, :]
        current_FAZgt = converted_FAZ_gt[i_file, :, :]
        current_junctgt = converted_junct_gt[i_file, :, :]
        current_vesselgt = converted_vessel_gt[i_file, :, :]
        
        # augmentation 0: original image, augmentation 1: flip in vertical, augmentation 2: flip in horizontal
        for i_aug in range(2+1):
            if i_aug == 0:
                flipped_current_SVC_file = current_SVC_file
                flipped_current_DVC_file = current_DVC_file
                flipped_current_IVC_file = current_IVC_file
                flipped_current_FAZgt = current_FAZgt
                flipped_current_junctgt = current_junctgt
                flipped_current_vesselgt = current_vesselgt
            else:
                flipped_current_SVC_file = np.flip(current_SVC_file, axis=(i_aug-1))
                flipped_current_DVC_file = np.flip(current_DVC_file, axis=(i_aug-1))
                flipped_current_IVC_file = np.flip(current_IVC_file, axis=(i_aug-1))
                flipped_current_FAZgt = np.flip(current_FAZgt, axis=(i_aug-1))
                flipped_current_junctgt = np.flip(current_junctgt, axis=(i_aug-1))
                flipped_current_vesselgt = np.flip(current_vesselgt, axis=(i_aug-1))            
            out_SVC_img[i_aug+(i_file)*(2+1), :, :] = flipped_current_SVC_file
            out_DVC_img[i_aug+(i_file)*(2+1), :, :] = flipped_current_DVC_file
            out_IVC_img[i_aug+(i_file)*(2+1), :, :] = flipped_current_IVC_file
            out_FAZGt[i_aug+(i_file)*(2+1), :, :] = flipped_current_FAZgt
            out_junctGt[i_aug+(i_file)*(2+1), :, :] = flipped_current_junctgt
            out_vesselGt[i_aug+(i_file)*(2+1), :, :] = flipped_current_vesselgt


    return out_SVC_img, out_DVC_img, out_IVC_img, out_FAZGt, out_junctGt, out_vesselGt
    
    
    
# Created date: Oct. 21st, 2025
# passed
# Brief User Introduction:
# This function will conduct image elastic deformation for data augmentation. 
# Input Parameter
# image: input image array (np array: [num_image,x,y])
# Output Parameter:
# out_img: processed image array (np array: [num_image,x,y])
# out_gt: processed groundtruth array (np array: [num_image,x,y])
def img_elastic_deformer_ROSEO(SVC_image, DVC_image, IVC_image, FAZ_groundtruth, junct_groundtruth, vessel_groundtruth):
    # for image elastic deformation
    import albumentations as A

    
    # check if the image is single channel
    # if it is not, convert it to single-channel grayscale image first
    if SVC_image.ndim != 3:
        from image_processors import image_SingleChannel_converter # import single channel converter if necessary
        converted_SVC_image = image_SingleChannel_converter(SVC_image)
    # if it's, just stay still with input image
    else:
        converted_SVC_image = SVC_image

    # check if the image is single channel
    # if it is not, convert it to single-channel grayscale image first
    if DVC_image.ndim != 3:
        from image_processors import image_SingleChannel_converter # import single channel converter if necessary
        converted_DVC_image = image_SingleChannel_converter(DVC_image)
    # if it's, just stay still with input image
    else:
        converted_DVC_image = DVC_image 
        
    # check if the image is single channel
    # if it is not, convert it to single-channel grayscale image first
    if IVC_image.ndim != 3:
        from image_processors import image_SingleChannel_converter # import single channel converter if necessary
        converted_IVC_image = image_SingleChannel_converter(IVC_image)
    # if it's, just stay still with input image
    else:
        converted_IVC_image = IVC_image

    # check if the groundtruth is single channel
    if FAZ_groundtruth.ndim != 3:
        from image_processors import image_SingleChannel_converter # import single channel converter if necessary
        converted_FAZ_gt = image_SingleChannel_converter(FAZ_groundtruth)
    # if it's, just stay still with input image
    else:
        converted_FAZ_gt = FAZ_groundtruth    
        
    # check if the groundtruth is single channel
    if junct_groundtruth.ndim != 3:
        from image_processors import image_SingleChannel_converter # import single channel converter if necessary
        converted_junct_gt = image_SingleChannel_converter(junct_groundtruth)
    # if it's, just stay still with input image
    else:
        converted_junct_gt = junct_groundtruth  
    
    # check if the groundtruth is single channel
    if vessel_groundtruth.ndim != 3:
        from image_processors import image_SingleChannel_converter # import single channel converter if necessary
        converted_vessel_gt = image_SingleChannel_converter(vessel_groundtruth)
    # if it's, just stay still with input image
    else:
        converted_vessel_gt = vessel_groundtruth 


    # check if the groundtruth is boolean
    if converted_FAZ_gt.dtype == bool:
        converted_FAZ_gt = converted_FAZ_gt.astype(np.uint8) * 255
    # if it's, just stay still with input image
    else:
        converted_FAZ_gt = converted_FAZ_gt
        
    # check if the groundtruth is boolean
    if converted_junct_gt.dtype == bool:
        converted_junct_gt = converted_junct_gt.astype(np.uint8) * 255
    # if it's, just stay still with input image
    else:
        converted_junct_gt = converted_junct_gt
        
    # check if the groundtruth is boolean
    if converted_vessel_gt.dtype == bool:
        converted_vessel_gt = converted_vessel_gt.astype(np.uint8) * 255
    # if it's, just stay still with input image
    else:
        converted_vessel_gt = converted_vessel_gt
        
        
    # conduct image elastic deformation
    aug = A.Compose([
    A.ElasticTransform(alpha=40, sigma=6, 
                       interpolation=1, border_mode=0, p=0.5)  # zeros outside
    ])
    num_row = SVC_image.shape[1]
    num_col = SVC_image.shape[2]
    num_files = SVC_image.shape[0]
    out_SVC_img = np.zeros((num_files*(1+1), num_row, num_col)) # like in the image flipping, this time, I only conduct one augmentation (elastic deformation).
                                                                # so, its outut as [x, y, (1+1)]
    out_DVC_img = np.zeros((num_files*(1+1), num_row, num_col)) 
    out_IVC_img = np.zeros((num_files*(1+1), num_row, num_col)) 
    out_FAZGt = np.zeros((num_files*(1+1), num_row, num_col))       
    out_junctGt = np.zeros((num_files*(1+1), num_row, num_col))        
    out_vesselGt = np.zeros((num_files*(1+1), num_row, num_col))        
    for i_file in range(num_files):
        # grab out current file
        current_SVC_file = converted_SVC_image[i_file, :, :]
        current_DVC_file = converted_SVC_image[i_file, :, :]
        current_IVC_file = converted_SVC_image[i_file, :, :]
        current_FAZgt = converted_FAZ_gt[i_file, :, :]
        current_junct_gt = converted_junct_gt[i_file, :, :]
        current_vessel_gt = converted_vessel_gt[i_file, :, :]
        
        # augmentation 0: original, augmentation: deformed image
        for i_aug in range(1+1):
            if i_aug == 0:
                deformed_current_SVC_file = current_SVC_file
                deformed_current_DVC_file = current_DVC_file
                deformed_current_IVC_file = current_IVC_file
                deformed_current_FAZgt = current_FAZgt
                deformed_current_junct_gt = current_junct_gt
                deformed_current_vessel_gt = current_vessel_gt
            else:
                deformed_current_SVC_file = aug(image=current_SVC_file)["image"]
                deformed_current_DVC_file = aug(image=current_DVC_file)["image"]
                deformed_current_IVC_file = aug(image=current_IVC_file)["image"]
                deformed_current_FAZgt = aug(image=current_FAZgt)["image"]
                deformed_current_junct_gt = aug(image=current_junct_gt)["image"]
                deformed_current_vessel_gt = aug(image=current_vessel_gt)["image"]
            out_SVC_img[i_aug+(i_file)*(1+1), :, :] = deformed_current_SVC_file
            out_DVC_img[i_aug+(i_file)*(1+1), :, :] = deformed_current_DVC_file
            out_IVC_img[i_aug+(i_file)*(1+1), :, :] = deformed_current_IVC_file
            out_FAZGt[i_aug+(i_file)*(1+1), :, :] = deformed_current_FAZgt
            out_junctGt[i_aug+(i_file)*(1+1), :, :] = deformed_current_junct_gt
            out_vesselGt[i_aug+(i_file)*(1+1), :, :] = deformed_current_vessel_gt


    return out_SVC_img, out_DVC_img, out_IVC_img, out_FAZGt, out_junctGt, out_vesselGt
    
    
    
# Created date: Oct. 21st, 2025
# passed
# Brief User Introduction:
# This function will conduct image contrast jitter for data augmentation. 
# Input Parameter
# image: input image array (np array: [num_image,x,y])
# Output Parameter:
# out_img: processed image array (np array: [num_image,x,y])   
# out_gt: processed groundtruth array (np array: [num_image,x,y])
def img_contrast_jitter_ROSEO(SVC_image, DVC_image, IVC_image, FAZ_groundtruth, junct_groundtruth, vessel_groundtruth):
    from PIL import Image, ImageEnhance # for image contrast jitter 
    
    
    # check if the image is single channel
    # if it is not, convert it to single-channel grayscale image first
    if SVC_image.ndim != 3: 
        from image_processors import image_SingleChannel_converter # import single channel converter if necessary
        converted_SVC_image = image_SingleChannel_converter(SVC_image)
    # if it's, just stay still with input image
    else:
        converted_SVC_image = SVC_image
        
    # check if the image is single channel
    # if it is not, convert it to single-channel grayscale image first
    if DVC_image.ndim != 3: 
        from image_processors import image_SingleChannel_converter # import single channel converter if necessary
        converted_DVC_image = image_SingleChannel_converter(DVC_image)
    # if it's, just stay still with input image
    else:
        converted_DVC_image = DVC_image

    # check if the image is single channel
    # if it is not, convert it to single-channel grayscale image first
    if IVC_image.ndim != 3: 
        from image_processors import image_SingleChannel_converter # import single channel converter if necessary
        converted_IVC_image = image_SingleChannel_converter(IVC_image)
    # if it's, just stay still with input image
    else:
        converted_IVC_image = IVC_image

    # check if the groundtruth is single channel
    if FAZ_groundtruth.ndim != 3:
        from image_processors import image_SingleChannel_converter # import single channel converter if necessary
        converted_FAZ_gt = image_SingleChannel_converter(FAZ_groundtruth)
    # if it's, just stay still with input image
    else:
        converted_FAZ_gt = FAZ_groundtruth      

    # check if the groundtruth is single channel
    if junct_groundtruth.ndim != 3:
        from image_processors import image_SingleChannel_converter # import single channel converter if necessary
        converted_junct_gt = image_SingleChannel_converter(junct_groundtruth)
    # if it's, just stay still with input image
    else:
        converted_junct_gt = junct_groundtruth 

    # check if the groundtruth is single channel
    if vessel_groundtruth.ndim != 3:
        from image_processors import image_SingleChannel_converter # import single channel converter if necessary
        converted_vessel_gt = image_SingleChannel_converter(vessel_groundtruth)
    # if it's, just stay still with input image
    else:
        converted_vessel_gt = vessel_groundtruth         
    
    
    # check if the groundtruth is boolean
    if converted_FAZ_gt.dtype == bool:
        converted_FAZ_gt = converted_FAZ_gt.astype(np.uint8) * 255
    # if it's, just stay still with input image
    else:
        converted_FAZ_gt = converted_FAZ_gt
        
    # check if the groundtruth is boolean
    if converted_junct_gt.dtype == bool:
        converted_junct_gt = converted_junct_gt.astype(np.uint8) * 255
    # if it's, just stay still with input image
    else:
        converted_junct_gt = converted_junct_gt
        
    # check if the groundtruth is boolean
    if converted_vessel_gt.dtype == bool:
        converted_vessel_gt = converted_vessel_gt.astype(np.uint8) * 255
    # if it's, just stay still with input image
    else:
        converted_vessel_gt = converted_vessel_gt
        
    
    # conduct image contrast jitter
    num_row = SVC_image.shape[1]
    num_col = SVC_image.shape[2]
    num_files = SVC_image.shape[0]
    out_SVC_img = np.zeros((num_files*(2+1), num_row, num_col)) # like in the image flipping, this time, I conduct two augmentation (higher contrast and lower contrast).
    out_DVC_img = np.zeros((num_files*(2+1), num_row, num_col)) 
    out_IVC_img = np.zeros((num_files*(2+1), num_row, num_col)) 
    out_FAZGt = np.zeros((num_files*(2+1), num_row, num_col))        
    out_junctGt = np.zeros((num_files*(2+1), num_row, num_col))        
    out_vesselGt = np.zeros((num_files*(2+1), num_row, num_col))        
    for i_file in range(num_files):
        # grab out current file
        current_SVC_file = converted_SVC_image[i_file, :, :]
        current_DVC_file = converted_DVC_image[i_file, :, :]
        current_IVC_file = converted_IVC_image[i_file, :, :]
        current_FAZgt = converted_FAZ_gt[i_file, :, :]
        current_junct_gt = converted_junct_gt[i_file, :, :]
        current_vessel_gt = converted_vessel_gt[i_file, :, :]
        
        # setup enhancer for images
        enhancer_SVC = ImageEnhance.Contrast(Image.fromarray(current_SVC_file.astype(np.uint8)))
        enhancer_DVC = ImageEnhance.Contrast(Image.fromarray(current_DVC_file.astype(np.uint8)))
        enhancer_IVC = ImageEnhance.Contrast(Image.fromarray(current_IVC_file.astype(np.uint8)))
        
        # augmentation 0: original, augmentation 1: enhanced contrast, augmentation 2: decreased contrast
        for i_aug in range(2+1):
            if i_aug == 0:
                out_SVC_img[i_aug+(i_file)*(2+1), :, :] = current_SVC_file
                out_DVC_img[i_aug+(i_file)*(2+1), :, :] = current_DVC_file
                out_IVC_img[i_aug+(i_file)*(2+1), :, :] = current_IVC_file
            elif i_aug == 1:
                out_SVC_img[i_aug+(i_file)*(2+1), :, :] = np.array(enhancer_SVC.enhance(1.5), dtype=np.uint8)
                out_DVC_img[i_aug+(i_file)*(2+1), :, :] = np.array(enhancer_DVC.enhance(1.5), dtype=np.uint8)
                out_IVC_img[i_aug+(i_file)*(2+1), :, :] = np.array(enhancer_IVC.enhance(1.5), dtype=np.uint8)
            else:
                out_SVC_img[i_aug+(i_file)*(2+1), :, :] = np.array(enhancer_SVC.enhance(0.8), dtype=np.uint8)
                out_DVC_img[i_aug+(i_file)*(2+1), :, :] = np.array(enhancer_DVC.enhance(0.8), dtype=np.uint8)
                out_IVC_img[i_aug+(i_file)*(2+1), :, :] = np.array(enhancer_IVC.enhance(0.8), dtype=np.uint8)
            out_FAZGt[i_aug+(i_file)*(2+1), :, :] = current_FAZgt 
            out_junctGt[i_aug+(i_file)*(2+1), :, :] = current_junct_gt 
            out_vesselGt[i_aug+(i_file)*(2+1), :, :] = current_vessel_gt 

            
    return out_SVC_img, out_DVC_img, out_IVC_img, out_FAZGt, out_junctGt, out_vesselGt
