# Created by Kuan-Min Lee
# All rights reserved to Leelab.ai
# still need tested

# Brief User Introducttion:
# This script includes some processing functions designed for rose-1 dataset

import numpy as np # for array computation (including output arrays, array dimension)



# Created date: Oct. 17th, 2025
# Brief User Introduction:
# this function will create a k-fold dataset for ROSE-1 dataset
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
    num_files = converted_train_ROSEO_SVC_org.shape[3]
    k = 4.0
    ratio = 1 / k
    num_train_files = int(num_files * ratio)
    num_valid_files = num_files - num_train_files
    
    # conduct k-fold dataset formation
    num_row = converted_train_ROSE_SVC_org.shape[0]
    num_col = converted_train_ROSE_SVC_org.shape[1]
    fold_valid_ROSEO_SVC_org = np.zeros((num_row, num_col, num_valid_files, k))
    fold_valid_ROSEO_DVC_org = np.zeros((num_row, num_col, num_valid_files, k))
    fold_valid_ROSEO_IVC_org = np.zeros((num_row, num_col, num_valid_files, k))
    fold_valid_ROSEO_FAZ_gt = np.zeros((num_row, num_col, num_valid_files, k))
    fold_valid_ROSEO_junct_gt = np.zeros((num_row, num_col, num_valid_files, k))
    fold_valid_ROSEO_vessel_gt = np.zeros((num_row, num_col, num_valid_files, k)) 
    fold_train_ROSEO_SVC_org = np.zeros((num_row, num_col, num_train_files, k))
    fold_train_ROSEO_DVC_org = np.zeros((num_row, num_col, num_train_files, k))
    fold_train_ROSEO_IVC_org = np.zeros((num_row, num_col, num_train_files, k))
    fold_train_ROSEO_FAZ_gt = np.zeros((num_row, num_col, num_train_files, k))
    fold_train_ROSEO_junct_gt = np.zeros((num_row, num_col, num_train_files, k))
    fold_train_ROSEO_vessel_gt = np.zeros((num_row, num_col, num_train_files, k))
    start_valid_ind = 0
    end_valid_ind = num_valid_files - 1
    for i_k in range(k):
        # setupt index
        valid_index = np.arange(start_valid_ind, end_valid_ind)
        train_intex = np.setdiff1d(np.arange(num_files), valid_index)
        # grab out dataset
        # validation dataset
        current_fold_valid_ROSEO_SVC_org = converted_train_ROSEO_SVC_org[:, :, valid_index]
        current_fold_valid_ROSEO_DVC_org = converted_train_ROSEO_DVC_org[:, :, valid_index]
        current_fold_valid_ROSEO_IVC_org = converted_train_ROSEO_IVC_org[:, :, valid_index]
        current_fold_valid_ROSEO_FAZ_gt = converted_train_ROSEO_FAZ_gt[:, :, valid_index]
        current_fold_valid_ROSEO_junct_gt = converted_train_ROSEO_junct_gt[:, :, valid_index]
        current_fold_valid_ROSEO_vessel_gt = converted_train_ROSEO_vessel_gt[:, :, valid_index]
        # training dataset
        current_fold_train_ROSEO_SVC_org = converted_train_ROSEO_SVC_org[:, :, train_index]
        current_fold_train_ROSEO_DVC_org = converted_train_ROSEO_DVC_org[:, :, train_index]
        current_fold_train_ROSEO_IVC_org = converted_train_ROSEO_IVC_org[:, :, train_index]
        current_fold_train_ROSEO_FAZ_gt = converted_train_ROSEO_FAZ_gt[:, :, train_index]
        current_fold_train_ROSEO_junct_gt = converted_train_ROSEO_junct_gt[:, :, train_index]
        current_fold_train_ROSEO_vessel_gt = converted_train_ROSEO_vessel_gt[:, :, train_index]
        # store in the outcome
        # validation dataset
        fold_valid_ROSEO_SVC_org[:, :, :, i_k] = current_fold_valid_ROSEO_SVC_org
        fold_valid_ROSEO_DVC_org[:, :, :, i_k] = current_fold_valid_ROSEO_DVC_org
        fold_valid_ROSEO_IVC_org[:, :, :, i_k] = current_fold_valid_ROSEO_IVC_org
        fold_valid_ROSEO_FAZ_gt[:, :, :, i_k] = current_fold_valid_ROSEO_FAZ_gt
        fold_valid_ROSEO_junct_gt[:, :, :, i_k] = current_fold_valid_ROSEO_junct_gt
        fold_valid_ROSEO_vessel_gt[:, :, :, i_k] = current_fold_valid_ROSEO_vessel_gt
        # training dataset
        fold_train_ROSEO_SVC_org[:, :, :, i_k] = current_fold_train_ROSEO_SVC_org
        fold_train_ROSEO_DVC_org[:, :, :, i_k] = current_fold_train_ROSEO_DVC_org
        fold_train_ROSEO_IVC_org[:, :, :, i_k] = current_fold_train_ROSEO_IVC_org
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
            "fold_valid_ROSE_SVC_org": fold_valid_ROSE_SVC_org,
            "fold_valid_ROSE_DVC_org": fold_valid_ROSE_DVC_org,
            "fold_valid_ROSE_IVC_org": fold_valid_ROSE_IVC_org,
            "fold_valid_ROSEO_FAZ_gt": fold_valid_ROSEO_FAZ_gt,
            "fold_valid_ROSEO_junct_gt": fold_valid_ROSEO_junct_gt,
            "fold_valid_ROSEO_vessel_gt": fold_valid_ROSEO_vessel_gt}
    
    
 
# Created date: Oct. 17th, 2025
# Brief User Introduction:
# This function will convert image into single channels and stored them into a 3D np array.
# Input Parameter
# image: input image array (np array: [x,y,chn,num_image])
# Output Parameter:
# out_image: processed image array (np array: [x,y,num_image])
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
        
    return (converted_train_ROSE_SVC_org, 
            converted_train_ROSE_DVC_org, 
            converted_train_ROSE_IVC_org,
            converted_train_ROSEO_FAZ_gt,
            converted_train_ROSEO_junct_gt,
            converted_train_ROSEO_vessel_gt)