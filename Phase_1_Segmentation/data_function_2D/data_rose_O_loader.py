# Created by Kuan-Min Lee
# Created date: Oct. 16th, 2025
# All rights reserved to Leelab.ai
# passed

# Brief User Introduction:
# This script is created to load ROSE-O dataset into array

# Outputs of this script:
# train_ROSEO_SVC_org: Training image for SVC layer (np array: [num_image,x,y])
# test_ROSEO_SVC_org: Testing image for SVC layer (np array: [num_image,x,y])
# train_ROSEO_DVC_org: Training image for DVC layer (np array: [num_image,x,y])
# test_ROSEO_DVC_org: Testing image for DVC layer (np array: [num_image,x,y])
# train_ROSEO_IVC_org: Training image for IVC layer (np array: [num_image,x,y])
# test_ROSEO_IVC_org: Testing image for IVC layer (np array: [num_image,x,y])
# train_ROSEO_FAZ_gt: Groundtruth image for FAZ zone (np array: [num_image,x,y])
# test_ROSEO_FAZ_gt: Groundtruth image for FAZ zone (np array: [num_image,x,y])
# train_ROSEO_junct_gt: Groundtruth image for junction node location of vessels (np array: [num_image,x,y])
# test_ROSEO_junct_gt: Groundtruth image for junction node location of vessels (np array: [num_image,x,y])
# train_ROSEO_vessel_gt: Groundtruth image for vessel segmentation (np array: [num_image,x,y])
# test_ROSEO_vessel_gt: Groundtruth image for vessel segmentation (np array: [num_image,x,y])



from .image_processors import image_SingleChannel_converter, image_safe_read_tif # read tif



def load_rose_O_dataset(base_dir=None):
    from pathlib import Path # for setting up directory for reading
    import imageio.v3 as iio # for image reading
    import numpy as np # for creating array 


    # setup parent directory for reading
    current_code_path = Path(__file__).resolve().parent
    if base_dir is None:
        base_dir = current_code_path.parent / 'rose_dataset' / 'ROSE-O'
        image_parent_dir = base_dir


    # image dataset loading
    # setup SVC train and test image directory
    train_img_SVC_dir = image_parent_dir / 'train' / 'img' / 'SVC'
    test_img_SVC_dir = image_parent_dir / 'test' / 'img' / 'SVC'
    # setup DVC train and test image directory
    train_img_DVC_dir = image_parent_dir / 'train' / 'img' / 'DVC'
    test_img_DVC_dir = image_parent_dir / 'test' / 'img' / 'DVC'
    # setup IVC train and test image directory
    train_img_IVC_dir = image_parent_dir / 'train' / 'img' / 'IVC'
    test_img_IVC_dir = image_parent_dir / 'test' / 'img' / 'IVC'

    # read all images stored inside
    # train SVC image
    train_img_SVC_files = sorted(Path(train_img_SVC_dir).glob("*.tif"))
    train_SVC_data = [image_safe_read_tif(f) for f in train_img_SVC_files]
    train_ROSEO_SVC_org = np.stack(train_SVC_data, axis=0)
    train_ROSEO_SVC_org_dim = train_ROSEO_SVC_org.ndim
    if train_ROSEO_SVC_org_dim > 3:
        train_ROSEO_SVC_org = image_SingleChannel_converter(train_ROSEO_SVC_org)
        
    # test SVC image
    test_img_SVC_files = sorted(Path(test_img_SVC_dir).glob("*.tif"))
    test_SVC_data = [image_safe_read_tif(f) for f in test_img_SVC_files]
    test_ROSEO_SVC_org = np.stack(test_SVC_data, axis=0)
    test_ROSEO_SVC_org_dim = test_ROSEO_SVC_org.ndim
    if test_ROSEO_SVC_org_dim > 3:
        test_ROSEO_SVC_org = image_SingleChannel_converter(test_ROSEO_SVC_org)
        
    # train DVC image
    train_img_DVC_files = sorted(Path(train_img_DVC_dir).glob("*.tif"))
    train_DVC_data = [image_safe_read_tif(f) for f in train_img_DVC_files]
    train_ROSEO_DVC_org = np.stack(train_DVC_data, axis=0)
    train_ROSEO_DVC_org_dim = train_ROSEO_DVC_org.ndim
    if train_ROSEO_DVC_org_dim > 3:
        train_ROSEO_DVC_org = image_SingleChannel_converter(train_ROSEO_DVC_org)

    # test DVC image
    test_img_DVC_files = sorted(Path(test_img_DVC_dir).glob("*.tif"))
    test_DVC_data = [image_safe_read_tif(f) for f in test_img_DVC_files]
    test_ROSEO_DVC_org = np.stack(test_DVC_data, axis=0)
    test_ROSEO_DVC_org_dim = test_ROSEO_DVC_org.ndim
    if test_ROSEO_DVC_org_dim > 3:
        test_ROSEO_DVC_org = image_SingleChannel_converter(test_ROSEO_DVC_org)

    # train IVC image
    train_img_IVC_files = sorted(Path(train_img_IVC_dir).glob("*.png"))
    train_IVC_data = [iio.imread(f) for f in train_img_IVC_files]
    train_ROSEO_IVC_org = np.stack(train_IVC_data, axis=0)
    train_ROSEO_IVC_org_dim = train_ROSEO_IVC_org.ndim
    if train_ROSEO_IVC_org_dim > 3:
        train_ROSEO_IVC_org = image_SingleChannel_converter(train_ROSEO_IVC_org)

    # test IVC image
    test_img_IVC_files = sorted(Path(test_img_IVC_dir).glob("*.png"))
    test_IVC_data = [iio.imread(f) for f in test_img_IVC_files]
    test_ROSEO_IVC_org = np.stack(test_IVC_data, axis=0)
    test_ROSEO_IVC_org_dim = test_ROSEO_IVC_org.ndim
    if test_ROSEO_IVC_org_dim > 3:
        test_ROSEO_IVC_org = image_SingleChannel_converter(test_ROSEO_IVC_org)

    # groundtruth dataset loading
    # FAZ segmentation groundtruth
    # setup FAZ train and test groundtruth directory
    train_gt_FAZ_dir = image_parent_dir / 'train' / 'gt' / 'FAZ'
    test_gt_FAZ_dir = image_parent_dir / 'test' / 'gt' / 'FAZ'
    # junctions groundtruth
    # setup junction train and test groundtruth directory
    train_gt_junct_dir = image_parent_dir / 'train' / 'gt' / 'junctions'
    test_gt_junct_dir = image_parent_dir / 'test' / 'gt' / 'junctions'
    # vessel segmentation groundtruth
    train_gt_vessel_dir = image_parent_dir / 'train' / 'gt' / 'vessel'
    test_gt_vessel_dir = image_parent_dir / 'test' / 'gt' / 'vessel'
    
    # read all images stored inside
    # train FAZ groundtruth
    train_gt_FAZ_files = sorted(Path(train_gt_FAZ_dir).glob("*.png"))
    train_FAZ_gt = [iio.imread(f) for f in train_gt_FAZ_files]
    train_ROSEO_FAZ_gt = np.stack(train_FAZ_gt, axis=0)
    train_ROSEO_FAZ_gt_dim = train_ROSEO_FAZ_gt.ndim
    if train_ROSEO_FAZ_gt_dim > 3:
        train_ROSEO_FAZ_gt = image_SingleChannel_converter(train_ROSEO_FAZ_gt)
    
    # test FAZ groundtruth
    test_gt_FAZ_files = sorted(Path(test_gt_FAZ_dir).glob("*.png"))
    test_FAZ_gt = [iio.imread(f) for f in test_gt_FAZ_files]
    test_ROSEO_FAZ_gt = np.stack(test_FAZ_gt, axis=0)
    test_ROSEO_FAZ_gt_dim = test_ROSEO_FAZ_gt.ndim
    if test_ROSEO_FAZ_gt_dim > 3:
        test_ROSEO_FAZ_gt = image_SingleChannel_converter(test_ROSEO_FAZ_gt)
        
    # train junction groundtruth
    train_gt_junct_files = sorted(Path(train_gt_junct_dir).glob("*.tif"))
    train_junct_gt = [image_safe_read_tif(f) for f in train_gt_junct_files]
    train_ROSEO_junct_gt = np.stack(train_junct_gt, axis=0)
    train_ROSEO_junct_gt_dim = train_ROSEO_junct_gt.ndim
    if train_ROSEO_junct_gt_dim > 3:
        train_ROSEO_junct_gt = image_SingleChannel_converter(train_ROSEO_junct_gt)
    
    # test junction groundtruth
    test_gt_junct_files = sorted(Path(test_gt_junct_dir).glob("*.tif"))
    test_junct_gt = [image_safe_read_tif(f) for f in test_gt_junct_files]
    test_ROSEO_junct_gt = np.stack(test_junct_gt, axis=0)
    test_ROSEO_junct_gt_dim = test_ROSEO_junct_gt.ndim
    if test_ROSEO_junct_gt_dim > 3:
        test_ROSEO_junct_gt = image_SingleChannel_converter(test_ROSEO_junct_gt)
        
    # train vessel groundtruth
    train_gt_vessel_files = sorted(Path(train_gt_vessel_dir).glob("*.png"))
    train_vessel_gt = [iio.imread(f) for f in train_gt_vessel_files]
    train_ROSEO_vessel_gt = np.stack(train_vessel_gt, axis=0)
    train_ROSEO_vessel_gt_dim = train_ROSEO_vessel_gt.ndim
    if train_ROSEO_vessel_gt_dim > 3:
        train_ROSEO_vessel_gt = image_SingleChannel_converter(train_ROSEO_vessel_gt)
    
    # test vessel groundtruth
    test_gt_vessel_files = sorted(Path(test_gt_vessel_dir).glob("*.png"))
    test_vessel_gt = [iio.imread(f) for f in test_gt_vessel_files]
    test_ROSEO_vessel_gt = np.stack(test_vessel_gt, axis=0)
    test_ROSEO_vessel_gt_dim = test_ROSEO_vessel_gt.ndim
    if test_ROSEO_vessel_gt_dim > 3:
        test_ROSEO_vessel_gt = image_SingleChannel_converter(test_ROSEO_vessel_gt)
    
    
    return {"train_ROSEO_SVC_org": train_ROSEO_SVC_org,
            "train_ROSEO_DVC_org": train_ROSEO_DVC_org,
            "train_ROSEO_IVC_org": train_ROSEO_IVC_org,
            "test_ROSEO_SVC_org": test_ROSEO_SVC_org,
            "test_ROSEO_DVC_org": test_ROSEO_DVC_org,
            "test_ROSEO_IVC_org": test_ROSEO_IVC_org,
            "train_ROSEO_FAZ_gt": train_ROSEO_FAZ_gt,
            "train_ROSEO_junct_gt": train_ROSEO_junct_gt,
            "train_ROSEO_vessel_gt": train_ROSEO_vessel_gt,
            "test_ROSEO_FAZ_gt": test_ROSEO_FAZ_gt,
            "test_ROSEO_junct_gt": test_ROSEO_junct_gt,
            "test_ROSEO_vessel_gt": test_ROSEO_vessel_gt}