# Created by Kuan-Min Lee
# Created date: Oct, 21st 2025 
# All rights reserved to Leelab.ai
# passed

# Brief User Introduction:
# This script is created to conduct data augmentation on folded ROSE-2 dataset



# Created by Kuan-Min Lee
# Created date: Oct, 21st 2025 
# All rights reserved to Leelab.ai
# still need tested

# Brief User Introduction:
# This script is created to conduct data augmentation on folded ROSE-1 dataset



from .image_rose_data_augmentator import img_rotator, img_flipper, img_elastic_deformer, img_contrast_jitter # augmentation function
import numpy as np



def augmentate_rose_2_dataset(train_ROSE2_org,
                              train_ROSE2_orgGt):
    # conduct rotation augmentation
    num_aug = 3
    angles = np.array([-10, 10])
    augmentated_train_ROSE2_org, augmentated_train_ROSE2_orgGt = img_rotator(num_aug, angles, train_ROSE2_org, train_ROSE2_orgGt)


    # conduct flipping augmentation
    augmentated_train_ROSE2_org, augmentated_train_ROSE2_orgGt = img_flipper(augmentated_train_ROSE2_org, augmentated_train_ROSE2_orgGt)

    
    # conduct elastic deformation augmentation
    augmentated_train_ROSE2_org, augmentated_train_ROSE2_orgGt = img_elastic_deformer(augmentated_train_ROSE2_org, augmentated_train_ROSE2_orgGt)
    
    
    # conduct contrast jitter augmentation
    augmentated_train_ROSE2_org, augmentated_train_ROSE2_orgGt = img_contrast_jitter(augmentated_train_ROSE2_org, augmentated_train_ROSE2_orgGt)
    
    
    return {"augmentated_train_ROSE2_org": augmentated_train_ROSE2_org, 
            "augmentated_train_ROSE2_orgGt": augmentated_train_ROSE2_orgGt}