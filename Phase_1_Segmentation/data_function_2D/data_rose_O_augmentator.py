# Created by Kuan-Min Lee
# Created date: Oct, 21st 2025 
# All rights reserved to Leelab.ai
# passed

# Brief User Introduction:
# This script is created to conduct data augmentation on folded ROSE-O dataset

# Created by Kuan-Min Lee
# Created date: Oct, 21st 2025 
# All rights reserved to Leelab.ai
# still need tested

# Brief User Introduction:
# This script is created to conduct data augmentation on folded ROSE-1 dataset



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