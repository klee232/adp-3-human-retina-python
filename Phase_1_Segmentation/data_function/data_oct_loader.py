# Created by Kuan-Min Lee
# Createed date: Oct. 15th, 2025
# (still needed testing)

# Brief User Introduction:
# This function is built to load and oct image data for our URI data.
# Since the data padding and unification are already done in matlab, 
# python function will only load the processed data without any additional process

# Output of this script: 
# oct_img: oct image from URI patient (np array with size of [x,y,z,num_object])
# oct_gt_img: oct groundtruth from URI patient (np array with size of [x,y,z,num_object]



from pathlib import Path # for directory establishment
import h5py # for loading .mat files


# setup dataset directory
dataset_oct_dir = "~/data/klee232/processed_data/oct arrays"
dataset_octgt_dir = "~/data/klee232/processed_data/oct gt arrays"


# setup file directory
oct_img_file_dir = f"{dataset_oct_dir}/pad_octa_data.mat"
oct_gt_file_dir = f"{dataset_octgt_dir}/pad_octa_gt_data.mat"


# load all oct image and groundtruth files and stored the cell variable into arrays
oct_img_file = h5py.File(oct_img_file_dir,'r')
oct_img_file_cell = oct_img_file['oct_storage']
oct_gt_file = h5py.File(oct_gt_file_dir,'r')
oct_gt_file_cell = octa_gt_file['oct_gt_storage']


# convert the cell variable into arrays
oct_img = []
oct_gt = []
num_obj = oct_img_file.shape[1]
for i_file in range(num_obj):
    # grab out data from cell 
    ref_oct = oct_img_file_cell[0,i_file]
    ref_oct_gt = oct_gt_file_cell[0,i_file]
    current_oct = oct_img_file[ref_octa]
    current_oct_gt = oct_gt_file[ref_octa_gt]
    # convert them into np.array with shape of [x,y,z]
    current_oct_reshaped = np.transpose(current_oct, (1,2,0))
    current_oct_gt_reshaped = np.transpose(current_oct_gt, (1,2,0))
    # store them into concatenated array
    oct_img = oct_img.append(current_oct_reshaped)
    oct_gt = octa_gt.append(current_oct_gt_reshaped)
# convert the final outcome into np.array
oct_img = np.stack(oct_img, axis=-1)
oct_gt = np.stack(oct_gt,axis=-1)