# Created by Kuan-Min Lee
# Created date: Oct. 17th, 2025
# All rights reserved to Leelab.ai
# passed

# Brief User Introducttion:
# this function will create a k-fold dataset for ROSE-2 dataset

# Inputs of this script:
# train_ROSE2_org: Training image (np array: [num_image,x,y])
# train_ROSE2_orgGt: Ground truth for training image (np array: [num_image,x,y])
#
# Outputs of this script:
# SVC dataset:
# fold_train_ROSE2_org: folded Training image (np array: [num_image,x,y,num_fold])
# fold_train_ROSE2_orgGt: folded Ground truth for training image (np array: [num_image,x,y,num_fold])
# fold_valid_ROSE2_org: folded validation image (np array: [num_image,x,y,num_fold])
# fold_valid_ROSE2_orgGt: folded ground truth for validation image (np array: [num_image,x,y,num_fold])



import numpy as np # for array computation (including output arrays, array dimension)



def partition_rose_2_dataset(train_ROSE2_org, 
                             train_ROSE2_orgGt):
    # check if the input images are all single channel --- if it is not, convert it to single-channel grayscale image first
    converted_images = single_channel_checker_rose_2_dataset(train_ROSE2_org, 
                                                             train_ROSE2_orgGt)      
    converted_train_ROSE2_org = converted_images[0]
    converted_train_ROSE2_orgGt = converted_images[1]
    
    # grab out the number of files and determine the amount of training and validation dataset
    num_files = converted_train_ROSE2_org.shape[0]
    k = 4.0
    ratio = 1 / k
    num_valid_files = int(num_files * ratio)
    num_train_files = num_files - num_valid_files
    
    # conduct k-fold dataset formation
    num_row = converted_train_ROSE2_org.shape[1]
    num_col = converted_train_ROSE2_org.shape[2]
    fold_valid_ROSE2_org = np.zeros((num_valid_files, num_row, num_col, int(k)), dtype=np.uint8)
    fold_valid_ROSE2_orgGt = np.zeros((num_valid_files, num_row, num_col, int(k)), dtype=np.uint8)
    fold_train_ROSE2_org = np.zeros((num_train_files, num_row, num_col, int(k)), dtype=np.uint8)
    fold_train_ROSE2_orgGt = np.zeros((num_train_files, num_row, num_col, int(k)), dtype=np.uint8)
    start_valid_ind = 0
    end_valid_ind = num_valid_files 
    for i_k in range(int(k)):
        # setupt index
        valid_index = np.arange(start_valid_ind, end_valid_ind)
        train_index = np.setdiff1d(np.arange(num_files), valid_index)
        
        # grab out dataset
        # validation dataset
        current_fold_valid_ROSE2_org = converted_train_ROSE2_org[valid_index, :, :]
        current_fold_valid_ROSE2_orgGt = converted_train_ROSE2_orgGt[valid_index, :, :]
        
        # training dataset
        current_fold_train_ROSE2_org = converted_train_ROSE2_org[train_index, :, :]
        current_fold_train_ROSE2_orgGt = converted_train_ROSE2_orgGt[train_index, :, :]
        
        # store in the outcome
        # validation dataset
        fold_valid_ROSE2_org[:, :, :, i_k] = current_fold_valid_ROSE2_org
        fold_valid_ROSE2_orgGt[:, :, :, i_k] = current_fold_valid_ROSE2_orgGt
        
        # training dataset
        fold_train_ROSE2_org[:, :, :, i_k] = current_fold_train_ROSE2_org
        fold_train_ROSE2_orgGt[:, :, :, i_k] = current_fold_train_ROSE2_orgGt
        
        # update validation start and end indices
        start_valid_ind = start_valid_ind + num_valid_files
        end_valid_ind = end_valid_ind + num_valid_files
        
        
    return {"fold_train_ROSE2_org": fold_train_ROSE2_org,
            "fold_train_ROSE2_orgGt": fold_train_ROSE2_orgGt,
            "fold_valid_ROSE2_org": fold_valid_ROSE2_org,
            "fold_valid_ROSE2_orgGt": fold_valid_ROSE2_orgGt}
    
 
# Created date: Oct. 17th, 2025
# Brief User Introduction:
# This function will convert image into single channels and stored them into a 3D np array.
# Input Parameter
# image: input image array (np array: [x,y,chn,num_image])
# Output Parameter:
# out_image: processed image array (np array: [num_image,x,y])
def single_channel_checker_rose_2_dataset(train_ROSE2_org, train_ROSE2_orgGt):
    # train SVC
    if train_ROSE2_org.ndim != 3:
        from image_processors import image_SingleChannel_converter # import single channel converter if necessary
        converted_train_ROSE2_org = image_SingleChannel_converter(train_ROSE2_org)
    # if it's, just stay still with input image
    else:
        converted_train_ROSE2_org = train_ROSE2_org
        
    # train SVC groundtruth
    if train_ROSE2_orgGt.ndim != 3:
        from image_processors import image_SingleChannel_converter # import single channel converter if necessary
        converted_train_ROSE2_orgGt = image_SingleChannel_converter(train_ROSE2_orgGt)
    # if it's, just stay still with input image
    else:
        converted_train_ROSE2_orgGt = train_ROSE2_orgGt
     
     
    return (converted_train_ROSE2_org, 
            converted_train_ROSE2_orgGt)