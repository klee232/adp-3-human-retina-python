# Created by Kuan-Min Lee
# Created date: Oct, 21st 2025 
# All rights reserved to Leelab.ai
# passed

# Brief User Introduction:
# This script is created to conduct data augmentation on folded ROSE-1 dataset



from .image_rose_1_data_augmentator import img_rotator_ROSE_SVC, img_flipper_ROSE_SVC, img_elastic_deformer_ROSE_SVC, img_contrast_jitter_ROSE_SVC # augmentation function for SVC
from .image_rose_data_augmentator import img_rotator, img_flipper, img_elastic_deformer, img_contrast_jitter # augmentation function
import numpy as np


def augmentate_rose_1_dataset(train_ROSE_SVC_org,
                              train_ROSE_SVC_orgGt,
                              train_ROSE_SVC_thickGt,
                              train_ROSE_SVC_thinGt,
                              train_ROSE_DVC_org,
                              train_ROSE_DVC_orgGt,
                              train_ROSE_SDVC_org,
                              train_ROSE_SDVC_orgGt):
    # conduct rotation augmentation
    num_aug = 3
    angles = np.array([-10, 10])
    augmentated_train_ROSE_SVC_org, augmentated_train_ROSE_SVC_orgGt, augmentated_train_ROSE_SVC_thickGt, augmentated_train_ROSE_SVC_thinGt = img_rotator_ROSE_SVC(num_aug, angles, 
                                                                                                                                                                   train_ROSE_SVC_org, 
                                                                                                                                                                   train_ROSE_SVC_orgGt, 
                                                                                                                                                                   train_ROSE_SVC_thickGt,
                                                                                                                                                                   train_ROSE_SVC_thinGt)
    augmentated_train_ROSE_DVC_org, augmentated_train_ROSE_DVC_orgGt = img_rotator(num_aug, angles, train_ROSE_DVC_org, train_ROSE_DVC_orgGt)
    augmentated_train_ROSE_SDVC_org, augmentated_train_ROSE_SDVC_orgGt = img_rotator(num_aug, angles, train_ROSE_SDVC_org, train_ROSE_SDVC_orgGt)
    


    # conduct flipping augmentation
    augmentated_train_ROSE_SVC_org, augmentated_train_ROSE_SVC_orgGt, augmentated_train_ROSE_SVC_thickGt, augmentated_train_ROSE_SVC_thinGt = img_flipper_ROSE_SVC(augmentated_train_ROSE_SVC_org, 
                                                                                                                                                                   augmentated_train_ROSE_SVC_orgGt, 
                                                                                                                                                                   augmentated_train_ROSE_SVC_thickGt,
                                                                                                                                                                   augmentated_train_ROSE_SVC_thinGt)
    augmentated_train_ROSE_DVC_org, augmentated_train_ROSE_DVC_orgGt = img_flipper(augmentated_train_ROSE_DVC_org, augmentated_train_ROSE_DVC_orgGt)
    augmentated_train_ROSE_SDVC_org, augmentated_train_ROSE_SDVC_orgGt = img_flipper(augmentated_train_ROSE_SDVC_org, augmentated_train_ROSE_SDVC_orgGt)
    

    
    # conduct elastic deformation augmentation
    augmentated_train_ROSE_SVC_org, augmentated_train_ROSE_SVC_orgGt, augmentated_train_ROSE_SVC_thickGt, augmentated_train_ROSE_SVC_thinGt = img_elastic_deformer_ROSE_SVC(augmentated_train_ROSE_SVC_org, 
                                                                                                                                                                            augmentated_train_ROSE_SVC_orgGt, 
                                                                                                                                                                            augmentated_train_ROSE_SVC_thickGt,
                                                                                                                                                                            augmentated_train_ROSE_SVC_thinGt)
    augmentated_train_ROSE_DVC_org, augmentated_train_ROSE_DVC_orgGt = img_elastic_deformer(augmentated_train_ROSE_DVC_org, augmentated_train_ROSE_DVC_orgGt)
    augmentated_train_ROSE_SDVC_org, augmentated_train_ROSE_SDVC_orgGt = img_elastic_deformer(augmentated_train_ROSE_SDVC_org, augmentated_train_ROSE_SDVC_orgGt)
    
    
    # conduct contrast jitter augmentation
    augmentated_train_ROSE_SVC_org, augmentated_train_ROSE_SVC_orgGt, augmentated_train_ROSE_SVC_thickGt, augmentated_train_ROSE_SVC_thinGt = img_contrast_jitter_ROSE_SVC(augmentated_train_ROSE_SVC_org, 
                                                                                                                                                                           augmentated_train_ROSE_SVC_orgGt, 
                                                                                                                                                                           augmentated_train_ROSE_SVC_thickGt,
                                                                                                                                                                           augmentated_train_ROSE_SVC_thinGt)
    augmentated_train_ROSE_DVC_org, augmentated_train_ROSE_DVC_orgGt = img_contrast_jitter(augmentated_train_ROSE_DVC_org, augmentated_train_ROSE_DVC_orgGt)
    augmentated_train_ROSE_SDVC_org, augmentated_train_ROSE_SDVC_orgGt = img_contrast_jitter(augmentated_train_ROSE_SDVC_org, augmentated_train_ROSE_SDVC_orgGt)
    
    
    
    return {"augmentated_train_ROSE_SVC_org": augmentated_train_ROSE_SVC_org, 
            "augmentated_train_ROSE_SVC_orgGt": augmentated_train_ROSE_SVC_orgGt, 
            "augmentated_train_ROSE_SVC_thickGt": augmentated_train_ROSE_SVC_thickGt, 
            "augmentated_train_ROSE_SVC_thinGt": augmentated_train_ROSE_SVC_thinGt, 
            "augmentated_train_ROSE_DVC_org": augmentated_train_ROSE_DVC_org,
            "augmentated_train_ROSE_DVC_orgGt": augmentated_train_ROSE_DVC_orgGt,
            "augmentated_train_ROSE_SDVC_org": augmentated_train_ROSE_SDVC_org,
            "augmentated_train_ROSE_SDVC_orgGt": augmentated_train_ROSE_SDVC_orgGt}