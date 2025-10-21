# Created by Kuan-Min Lee
# Createed date: Oct. 16th, 2025
# (still needed testing)

# Brief User Introduction:
# This function is built to load the octa image data for our URI data.
# Since the data padding and unification are already done in matlab, 
# python function will only load the processed data without any additional process

# Output of this script: 
# octa_img: octa image from URI patient (np array with size of [x,y,z,num_object])
# octa_gt_img: octa groundtruth from URI patient (np array with size of [x,y,z,num_object]



import h5py # for loading .mat files
import numpy as np # for array storage


# setup dataset directory
dataset_octa_dir = "~/data/klee232/processed_data/octa arrays"
dataset_octa_gt_dir = "~/data/klee232/processed_data/octa gt arrays"


# setup file directory
octa_img_file_dir = f"{dataset_octa_dir}/pad_octa_data.mat"
octa_gt_file_dir = f"{dataset_octa_gt_dir}/pad_octa_gt_data.mat"


# load all octa image and groundtruth files 
octa_img_file = h5py.File(octa_img_file_dir,'r')
octa_img_file_cell = octa_img_file['octa_storage']
octa_gt_file = h5py.File(octa_gt_file_dir,'r')
octa_gt_file_cell = octa_gt_file['octa_gt_storage']


# convert the cell variable into arrays
octa_img = []
octa_gt = []
num_obj = octa_img_file.shape[1]
for i_file in range(num_obj):
    # grab out data from cell 
    ref_octa = octa_img_file_cell[0,i_file]
    ref_octa_gt = octa_gt_file_cell[0,i_file]
    current_octa = octa_img_file[ref_octa]
    current_octa_gt = octa_gt_file[ref_octa_gt]
    # convert them into np.array with shape of [x,y,z]
    current_octa_reshaped = np.transpose(current_octa, (1,2,0))
    current_octa_gt_reshaped = np.transpose(current_octa_gt, (1,2,0))
    # store them into concatenated array
    octa_img = octa_img.append(current_octa_reshaped)
    octa_gt = octa_gt.append(current_octa_gt_reshaped)
# convert the final outcome into np.array
octa_img = np.stack(octa_img, axis=-1)
octa_gt = np.stack(octa_gt,axis=-1)

