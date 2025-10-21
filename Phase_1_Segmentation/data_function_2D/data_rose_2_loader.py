# Created by Kuan-Min Lee
# Created date: Oct. 16th, 2025
# All rights reserved to Leelab.ai
# still need tested

# Brief User Introduction:
# This script is created to load ROSE-2 dataset into array

# Outputs of this script:
# train_ROSE2_org: Training image (np array: [x,y,num_image])
# train_ROSE2_orgGt: Ground truth for training image (np array: [x,y,num_image])
# test_ROSE2_org: Testing image (np array: [x,y,num_image])
# test_ROSE2_orgGt: Testing Ground Truth image (np array: [x,y,num_image])



def load_rose_2_dataset(base_dir=None):
    from pathlib import Path # for setting up directory for reading
    import imageio.v3 as iio # for image reading
    import numpy as np # for creating array 

    # setup parent directory for reading
    current_code_path = Path(__file__).resolve().parent
    if base_dir is None:
        base_dir = current_code_path.parent / 'rose_dataset' / 'ROSE-2'
        image_parent_dir = base_dir

    # dataset loading
    # setup train and test image directory
    train_img_dir = image_parent_dir / 'train' / 'original'
    test_img_dir = image_parent_dir / 'test' / 'original'
    # setup train and test groundtruth directory
    train_gt_dir = image_parent_dir / 'train' / 'gt' # overall groundtruth
    test_gt_dir = image_parent_dir / 'test' / 'gt' # overall groundtruth
    # train image
    train_img_files = sorted(Path(train_img_dir).glob("*.png"))
    train_data = [iio.imread(f) for f in train_img_files]
    train_ROSE2_org = np.stack(train_data, axis=-1)
    # test image
    test_img_files = sorted(Path(test_img_dir).glob("*.png"))
    test_data = [iio.imread(f) for f in test_img_files]
    test_ROSE2_org = np.stack(test_data, axis=-1)
    # train overall groundtruth
    train_gt_files = sorted(Path(train_gt_dir).glob("*.png"))
    train_gt = [iio.imread(f) for f in train_gt_files]
    train_ROSE2_orgGt = np.stack(train_gt, axis=-1)
    # test overall groundtruth
    test_gt_files = sorted(Path(test_gt_dir).glob("*.png"))
    test_gt = [iio.imread(f) for f in test_gt_files]
    test_ROSE2_orgGt = np.stack(test_gt, axis=-1)
    
    return {"train_ROSE2_org": train_ROSE2_org,
            "train_ROSE2_orgGt": train_ROSE2_orgGt,
            "test_ROSE2_org": test_ROSE2_org,
            "test_ROSE2_orgGt": test_ROSE2_orgGt}
            