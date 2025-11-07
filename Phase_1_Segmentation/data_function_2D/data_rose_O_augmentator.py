# Created by Kuan-Min Lee
# Created date: Oct, 21st 2025 
# All rights reserved to Leelab.ai
# passed

# Brief User Introduction:
# This script is created to conduct data augmentation on folded ROSE-O dataset

# Inputs of this script:
# original dataset:
# train_ROSEO_SVC_org: Training image for SVC (np array: [x,y,num_image])
# train_ROSEO_DVC_org: Training image for DVC (np array: [x,y,num_image])
# train_ROSEO_IVC_org: Training image for SDVC (np array: [x,y,num_image])
#
# groundtruth dataset:
# train_ROSEO_FAZ_gt: Groundtruth image for Foveal Area Segmentation (np array: [x,y,num_image])
# train_ROSEO_junct_org: Groundtruth image for junction location (np array: [x,y,num_image])
# train_ROSEO_vessel_org: Groundtruth image for vessel segmentation (np array: [x,y,num_image])
#
# Outputs of this script
# original dataset:
# augmentated_train_ROSEO_SVC_org: augmentated Training image for SVC (np array: [x,y,num_image])
# augmentated_train_ROSEO_DVC_org: augmentated Training image for DVC (np array: [x,y,num_image])
# augmentated_train_ROSEO_IVC_org: augmentated Training image for SDVC (np array: [x,y,num_image])
#
# groundtruth dataset:
# augmentated_train_ROSEO_FAZ_gt: augmentated Groundtruth image for Foveal Area Segmentation (np array: [x,y,num_image])
# augmentated_train_ROSEO_junct_org: augmentated Groundtruth image for junction location (np array: [x,y,num_image])
# augmentatedtrain_ROSEO_vessel_org: augmentated Groundtruth image for vessel segmentation (np array: [x,y,num_image])



from .image_rose_O_data_augmentator import img_rotator_ROSEO, img_flipper_ROSEO, img_elastic_deformer_ROSEO, img_contrast_jitter_ROSEO # augmentation function
import numpy as np


def augmentate_rose_O_dataset(train_ROSEO_SVC_org,
                              train_ROSEO_DVC_org,
                              train_ROSEO_IVC_org,
                              train_ROSEO_FAZ_gt,
                              train_ROSEO_junct_gt,
                              train_ROSEO_vessel_gt):
    # conduct rotation augmentation
    num_aug = 3
    angles = np.array([-10, 10])
    augmentated_train_ROSEO_SVC_org, augmentated_train_ROSEO_DVC_org, augmentated_train_ROSEO_IVC_org, augmentated_train_ROSEO_FAZ_gt, augmentated_train_ROSEO_junct_gt, augmentated_train_ROSEO_vessel_gt = img_rotator_ROSEO(num_aug, angles, 
                                                                                                                                                                                                                               train_ROSEO_SVC_org, 
                                                                                                                                                                                                                               train_ROSEO_DVC_org,
                                                                                                                                                                                                                               train_ROSEO_IVC_org,
                                                                                                                                                                                                                               train_ROSEO_FAZ_gt,
                                                                                                                                                                                                                               train_ROSEO_junct_gt,
                                                                                                                                                                                                                               train_ROSEO_vessel_gt)
    

    # conduct flipping augmentation
    augmentated_train_ROSEO_SVC_org, augmentated_train_ROSEO_DVC_org, augmentated_train_ROSEO_IVC_org, augmentated_train_ROSEO_FAZ_gt, augmentated_train_ROSEO_junct_gt, augmentated_train_ROSEO_vessel_gt = img_flipper_ROSEO(augmentated_train_ROSEO_SVC_org, 
                                                                                                                                                                                                                               augmentated_train_ROSEO_DVC_org,
                                                                                                                                                                                                                               augmentated_train_ROSEO_IVC_org,
                                                                                                                                                                                                                               augmentated_train_ROSEO_FAZ_gt,
                                                                                                                                                                                                                               augmentated_train_ROSEO_junct_gt,
                                                                                                                                                                                                                               augmentated_train_ROSEO_vessel_gt)
    
    
    # conduct elastic deformation augmentation
    augmentated_train_ROSEO_SVC_org, augmentated_train_ROSEO_DVC_org, augmentated_train_ROSEO_IVC_org, augmentated_train_ROSEO_FAZ_gt, augmentated_train_ROSEO_junct_gt, augmentated_train_ROSEO_vessel_gt = img_elastic_deformer_ROSEO(augmentated_train_ROSEO_SVC_org, 
                                                                                                                                                                                                                                        augmentated_train_ROSEO_DVC_org,
                                                                                                                                                                                                                                        augmentated_train_ROSEO_IVC_org,
                                                                                                                                                                                                                                        augmentated_train_ROSEO_FAZ_gt,
                                                                                                                                                                                                                                        augmentated_train_ROSEO_junct_gt,
                                                                                                                                                                                                                                        augmentated_train_ROSEO_vessel_gt)
    
    
    # conduct contrast jitter augmentation
    augmentated_train_ROSEO_SVC_org, augmentated_train_ROSEO_DVC_org, augmentated_train_ROSEO_IVC_org, augmentated_train_ROSEO_FAZ_gt, augmentated_train_ROSEO_junct_gt, augmentated_train_ROSEO_vessel_gt = img_contrast_jitter_ROSEO(augmentated_train_ROSEO_SVC_org, 
                                                                                                                                                                                                                                       augmentated_train_ROSEO_DVC_org,
                                                                                                                                                                                                                                       augmentated_train_ROSEO_IVC_org,
                                                                                                                                                                                                                                       augmentated_train_ROSEO_FAZ_gt,
                                                                                                                                                                                                                                       augmentated_train_ROSEO_junct_gt,
                                                                                                                                                                                                                                       augmentated_train_ROSEO_vessel_gt)

    
    return {"augmentated_train_ROSEO_SVC_org": augmentated_train_ROSEO_SVC_org, 
            "augmentated_train_ROSEO_DVC_org": augmentated_train_ROSEO_DVC_org,
            "augmentated_train_ROSEO_IVC_org": augmentated_train_ROSEO_IVC_org,
            "augmentated_train_ROSEO_FAZ_gt": augmentated_train_ROSEO_FAZ_gt,
            "augmentated_train_ROSEO_junct_gt": augmentated_train_ROSEO_junct_gt,
            "augmentated_train_ROSEO_vessel_gt": augmentated_train_ROSEO_vessel_gt}