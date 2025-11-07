# Created by Kuan-Min Lee
# Created date: Oct. 17th, 2025
# All rights reserved to Leelab.ai
# passed

# Brief User Introducttion:
# this function will create a k-fold dataset for ROSE-O dataset

# Inputs of this script:
# original dataset:
# train_ROSEO_SVC_org: Training image for SVC (np array: [num_image,x,y])
# train_ROSEO_DVC_org: Training image for DVC (np array: [num_image,x,y])
# train_ROSEO_IVC_org: Training image for SDVC (np array: [num_image,x,y])
# 
# groundtruth:
# train_ROSEO_FAZ_gt: Training image for Foveal area segmentation groundtruth (np array: [num_image,x,y])
# train_ROSEO_junct_gt: Training image for junction location segmentation groundtruth (np array: [num_image,x,y])
# train_ROSEO_vessel_gt: Training image for vessel segmentation groundtruth (np array: [num_image,x,y])
#
# Outputs of this script:
# original dataset:
# fold_train_ROSEO_SVC_org: folded Training image for SVC (np array: [num_image,x,y,num_fold])
# fold_train_ROSEO_DVC_org: folded Training image for DVC (np array: [num_image,x,y,num_fold])
# fold_train_ROSEO_IVC_org: folded Training image for SDVC (np array: [num_image,x,y,num_fold])
# fold_valid_ROSEO_SVC_org: folded validation image for SVC (np array: [num_image,x,y,num_fold])
# fold_valid_ROSEO_DVC_org: folded validation image for DVC (np array: [num_image,x,y,num_fold])
# fold_valid_ROSEO_IVC_org: folded validation image for SDVC (np array: [num_image,x,y,num_fold])
# 
# groundtruth:
# fold_train_ROSEO_FAZ_gt: folded Training image for Foveal area segmentation groundtruth (np array: [num_image,x,y,num_fold])
# fold_train_ROSEO_junct_gt: folded Training image for junction location segmentation groundtruth (np array: [num_image,x,y,num_fold])
# fold_train_ROSEO_vessel_gt: folded Training image for vessel segmentation groundtruth (np array: [num_image,x,y,num_fold])
# fold_valid_ROSEO_FAZ_gt: folded validation image for Foveal area segmentation groundtruth (np array: [num_image,x,y,num_fold])
# fold_valid_ROSEO_junct_gt: folded validation image for junction location segmentation groundtruth (np array: [num_image,x,y,num_fold])
# fold_valid_ROSEO_vessel_gt: folded validation image for vessel segmentation groundtruth (np array: [num_image,x,y,num_fold])



import numpy as np # for array computation (including output arrays, array dimension)



def partition_rose_O_dataset(train_ROSEO_SVC_org, 
                             train_ROSEO_DVC_org,
                             train_ROSEO_IVC_org,
                             train_ROSEO_FAZ_gt,
                             train_ROSEO_junct_gt,
                             train_ROSEO_vessel_gt):
    # check if the input images are all single channel --- if it is not, convert it to single-channel grayscale image first
    converted_images = single_channel_checker_rose_O_dataset(train_ROSEO_SVC_org, 
                                                             train_ROSEO_DVC_org,
                                                             train_ROSEO_IVC_org,
                                                             train_ROSEO_FAZ_gt,
                                                             train_ROSEO_junct_gt,
                                                             train_ROSEO_vessel_gt)      
    converted_train_ROSEO_SVC_org = converted_images[0]
    converted_train_ROSEO_DVC_org = converted_images[1]
    converted_train_ROSEO_IVC_org = converted_images[2]
    converted_train_ROSEO_FAZ_gt = converted_images[3]
    converted_train_ROSEO_junct_gt = converted_images[4]
    converted_train_ROSEO_vessel_gt = converted_images[5]
    
    
    # grab out the number of files and determine the amount of training and validation dataset
    num_files = converted_train_ROSEO_SVC_org.shape[0]
    k = 4.0
    ratio = 1 / k
    num_valid_files = int(num_files * ratio)
    num_train_files = num_files - num_valid_files
    
    
    # conduct k-fold dataset formation
    num_row = converted_train_ROSEO_SVC_org.shape[1]
    num_col = converted_train_ROSEO_SVC_org.shape[2]
    fold_valid_ROSEO_SVC_org = np.zeros((num_valid_files, num_row, num_col, int(k)), dtype=np.uint8)
    fold_valid_ROSEO_DVC_org = np.zeros((num_valid_files, num_row, num_col, int(k)), dtype=np.uint8)
    fold_valid_ROSEO_IVC_org = np.zeros((num_valid_files, num_row, num_col, int(k)), dtype=np.uint8)
    fold_valid_ROSEO_FAZ_gt = np.zeros((num_valid_files, num_row, num_col, int(k)), dtype=np.uint8)
    fold_valid_ROSEO_junct_gt = np.zeros((num_valid_files, num_row, num_col, int(k)), dtype=np.uint8)
    fold_valid_ROSEO_vessel_gt = np.zeros((num_valid_files, num_row, num_col, int(k)), dtype=np.uint8) 
    fold_train_ROSEO_SVC_org = np.zeros((num_train_files, num_row, num_col, int(k)), dtype=np.uint8)
    fold_train_ROSEO_DVC_org = np.zeros((num_train_files, num_row, num_col, int(k)), dtype=np.uint8)
    fold_train_ROSEO_IVC_org = np.zeros((num_train_files, num_row, num_col, int(k)), dtype=np.uint8)
    fold_train_ROSEO_FAZ_gt = np.zeros((num_train_files, num_row, num_col, int(k)), dtype=np.uint8)
    fold_train_ROSEO_junct_gt = np.zeros((num_train_files, num_row, num_col, int(k)), dtype=np.uint8)
    fold_train_ROSEO_vessel_gt = np.zeros((num_train_files, num_row, num_col, int(k)), dtype=np.uint8)
    start_valid_ind = 0
    end_valid_ind = num_valid_files 
    for i_k in range(int(k)):
        # setupt index
        valid_index = np.arange(start_valid_ind, end_valid_ind)
        train_index = np.setdiff1d(np.arange(num_files), valid_index)
        # grab out dataset
        # validation dataset
        # original dataset
        current_fold_valid_ROSEO_SVC_org = converted_train_ROSEO_SVC_org[valid_index, :, :]
        current_fold_valid_ROSEO_DVC_org = converted_train_ROSEO_DVC_org[valid_index, :, :]
        current_fold_valid_ROSEO_IVC_org = converted_train_ROSEO_IVC_org[valid_index, :, :]
        
        # groundtruth dataset
        current_fold_valid_ROSEO_FAZ_gt = converted_train_ROSEO_FAZ_gt[valid_index, :, :]
        current_fold_valid_ROSEO_junct_gt = converted_train_ROSEO_junct_gt[valid_index, :, :]
        current_fold_valid_ROSEO_vessel_gt = converted_train_ROSEO_vessel_gt[valid_index, :, :]
        
        # training dataset
        # original dataset
        current_fold_train_ROSEO_SVC_org = converted_train_ROSEO_SVC_org[train_index, :, :]
        current_fold_train_ROSEO_DVC_org = converted_train_ROSEO_DVC_org[train_index, :, :]
        current_fold_train_ROSEO_IVC_org = converted_train_ROSEO_IVC_org[train_index, :, :]
        
        # groundtruth dataset
        current_fold_train_ROSEO_FAZ_gt = converted_train_ROSEO_FAZ_gt[train_index, :, :]
        current_fold_train_ROSEO_junct_gt = converted_train_ROSEO_junct_gt[train_index, :, :]
        current_fold_train_ROSEO_vessel_gt = converted_train_ROSEO_vessel_gt[train_index, :, :]
        
        # store in the outcome
        # validation dataset
        # original dataset
        fold_valid_ROSEO_SVC_org[:, :, :, i_k] = current_fold_valid_ROSEO_SVC_org
        fold_valid_ROSEO_DVC_org[:, :, :, i_k] = current_fold_valid_ROSEO_DVC_org
        fold_valid_ROSEO_IVC_org[:, :, :, i_k] = current_fold_valid_ROSEO_IVC_org
        
        # groundtruth dataset
        fold_valid_ROSEO_FAZ_gt[:, :, :, i_k] = current_fold_valid_ROSEO_FAZ_gt
        fold_valid_ROSEO_junct_gt[:, :, :, i_k] = current_fold_valid_ROSEO_junct_gt
        fold_valid_ROSEO_vessel_gt[:, :, :, i_k] = current_fold_valid_ROSEO_vessel_gt
        
        # training dataset
        # original dataset
        fold_train_ROSEO_SVC_org[:, :, :, i_k] = current_fold_train_ROSEO_SVC_org
        fold_train_ROSEO_DVC_org[:, :, :, i_k] = current_fold_train_ROSEO_DVC_org
        fold_train_ROSEO_IVC_org[:, :, :, i_k] = current_fold_train_ROSEO_IVC_org
        
        # groundtruth dataset
        fold_train_ROSEO_FAZ_gt[:, :, :, i_k] = current_fold_train_ROSEO_FAZ_gt
        fold_train_ROSEO_junct_gt[:, :, :, i_k] = current_fold_train_ROSEO_junct_gt
        fold_train_ROSEO_vessel_gt[:, :, :, i_k] = current_fold_train_ROSEO_vessel_gt
        
        # update validation start and end indices
        start_valid_ind = start_valid_ind + num_valid_files
        end_valid_ind = end_valid_ind + num_valid_files
        
        
    return {"fold_train_ROSEO_SVC_org": fold_train_ROSEO_SVC_org,
            "fold_train_ROSEO_DVC_org": fold_train_ROSEO_DVC_org,
            "fold_train_ROSEO_IVC_org": fold_train_ROSEO_IVC_org,
            "fold_train_ROSEO_FAZ_gt": fold_train_ROSEO_FAZ_gt,
            "fold_train_ROSEO_junct_gt": fold_train_ROSEO_junct_gt,
            "fold_train_ROSEO_vessel_gt": fold_train_ROSEO_vessel_gt,
            "fold_valid_ROSEO_SVC_org": fold_valid_ROSEO_SVC_org,
            "fold_valid_ROSEO_DVC_org": fold_valid_ROSEO_DVC_org,
            "fold_valid_ROSEO_IVC_org": fold_valid_ROSEO_IVC_org,
            "fold_valid_ROSEO_FAZ_gt": fold_valid_ROSEO_FAZ_gt,
            "fold_valid_ROSEO_junct_gt": fold_valid_ROSEO_junct_gt,
            "fold_valid_ROSEO_vessel_gt": fold_valid_ROSEO_vessel_gt}
    
    
 
# Created date: Oct. 17th, 2025
# Brief User Introduction:
# This function will convert image into single channels and stored them into a 3D np array.
# Input Parameter
# image: input image array (np array: [x,y,chn,num_image])
# Output Parameter:
# out_image: processed image array (np array: [num_image,x,y])
def single_channel_checker_rose_O_dataset(
    train_ROSEO_SVC_org, 
    train_ROSEO_DVC_org,
    train_ROSEO_IVC_org,
    train_ROSEO_FAZ_gt,
    train_ROSEO_junct_gt,
    train_ROSEO_vessel_gt
):
    # train SVC
    if train_ROSEO_SVC_org.ndim != 3:
        from image_processors import image_SingleChannel_converter # import single channel converter if necessary
        converted_train_ROSEO_SVC_org = image_SingleChannel_converter(train_ROSEO_SVC_org)
    # if it's, just stay still with input image
    else:
        converted_train_ROSEO_SVC_org = train_ROSEO_SVC_org
        
    # train DVC
    if train_ROSEO_DVC_org.ndim != 3:
        from image_processors import image_SingleChannel_converter # import single channel converter if necessary
        converted_train_ROSEO_DVC_org = image_SingleChannel_converter(train_ROSEO_DVC_org)
    # if it's, just stay still with input image
    else:
        converted_train_ROSEO_DVC_org = train_ROSEO_DVC_org
    
    # train IVC
    if train_ROSEO_IVC_org.ndim != 3:
        from image_processors import image_SingleChannel_converter # import single channel converter if necessary
        converted_train_ROSEO_IVC_org = image_SingleChannel_converter(train_ROSEO_IVC_org)
    # if it's, just stay still with input image
    else:
        converted_train_ROSEO_IVC_org = train_ROSEO_IVC_org

    # train FAZ
    if train_ROSEO_FAZ_gt.ndim != 3:
        from image_processors import image_SingleChannel_converter # import single channel converter if necessary
        converted_train_ROSEO_FAZ_gt = image_SingleChannel_converter(train_ROSEO_FAZ_gt)
    # if it's, just stay still with input image
    else:
        converted_train_ROSEO_FAZ_gt = train_ROSEO_FAZ_gt
    
    # train junction
    if train_ROSEO_junct_gt.ndim != 3:
        from image_processors import image_SingleChannel_converter # import single channel converter if necessary
        converted_train_ROSEO_junct_gt = image_SingleChannel_converter(train_ROSEO_junct_gt)
    # if it's, just stay still with input image
    else:
        converted_train_ROSEO_junct_gt = train_ROSEO_junct_gt
    
    # train vessel
    if train_ROSEO_vessel_gt.ndim != 3:
        from image_processors import image_SingleChannel_converter # import single channel converter if necessary
        converted_train_ROSEO_vessel_gt = image_SingleChannel_converter(train_ROSEO_vessel_gt)
    # if it's, just stay still with input image
    else:
        converted_train_ROSEO_vessel_gt = train_ROSEO_vessel_gt
        
    return (converted_train_ROSEO_SVC_org, 
            converted_train_ROSEO_DVC_org, 
            converted_train_ROSEO_IVC_org,
            converted_train_ROSEO_FAZ_gt,
            converted_train_ROSEO_junct_gt,
            converted_train_ROSEO_vessel_gt)