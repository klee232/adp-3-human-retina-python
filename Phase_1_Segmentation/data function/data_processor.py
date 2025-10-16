# Created by Kuan-Min Lee
# Createed date: Oct. 15th, 2025
# (still needed testing)

# Brief User Introduction:
# This function is built to load and store processed octa image.
# At the end of this execution, you will receive:
# 1. octa_storage: storage for original OCTA image
# 2. octa_gt_storage: groundturth for OCTA image



from pathlib import Path # for directory establishment
import h5py # for loading .mat files


# create processed data storage directory if necessary
dataset_dir = "~/data/klee232/processed_data"
dataset_dir=Path(dataset_dir)
if not dataset_dir.exists():
    dataset_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created directory: {dataset_dir}")
else:
    print(f"Directory {dataset_dir} already existed")

    
# load all octa and oct image files
stored_image_dir="~/data/klee232/dataset"
stored_image_dir=Path(stored_image_dir)
octa_files=stored_image_dir.glob("*OCTA.mat") # retrieve all octa files
oct_files=stored_image_dir.glob("*OCT.mat") # retrieve all octa files
# check if the octa and oct files have identical amount
if len(octa_files)!=len(oct_files):
    raise ValueError("Unequal amount of octa and oct files found. Please double check")
else:
    octa_images_cell=[]
    octa_obj_cell=[]
    for current_file in octa_files:
        with h5py.File(current_file,"r") as f:
            current_image=f["DD1"][:]
            current_obj=f["r1"][:]
            octa_images_cell.append(current_image)
            octa_obj_cell.append(current_obj)
    oct_images_cell=[]
    for current_file in oct_files:
        with h5py.File(current_file,"r") as f:
            current_image=f["II1"][:]
            oct_image_cell.append(current_image)



    
    
 

    

