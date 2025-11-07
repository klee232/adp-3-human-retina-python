# Created by Kuan-Min Lee
# Created date: Oct, 16th 2025 
# All rights reserved to Leelab.ai
# passed

# Brief User Introduction:
# This script is created to load ROSE dataset into array

# Inputs of this script: None
# Outputs of this script:
# SVC dataset:
# train_ROSE_SVC_org: Training image for SVC (np array: [num_image,x,y])
# train_ROSE_SVC_orgGt: Ground truth for training image for SVC (np array: [num_image,x,y])
# train_ROSE_SVC_thinGt: Thin vessel Ground truth for training image for SVC (np array: [num_image,x,y])
# train_ROSE_SVC_thickGt: Thick vessel Ground truth for training image for SVC (np array: [num_image,x,y])
# test_ROSE_SVC_org: Testing image for SVC (np array: [num_image,x,y])
# test_ROSE_SVC_orgGt: Testing Ground Truth image for SVC (np array: [num_image,x,y])
# test_ROSE_SVC_thinGt: Thin vessel Ground Truth image for SVC (np array: [num_image,x,y])
# test_ROSE_SVC_thickGt: Thick vessel Ground Truth image for SVC (np array: [num_image,x,y])
#
# DVC dataset:
# train_ROSE_DVC_org: Training image for DVC (np array: [num_image,x,y])
# train_ROSE_DVC_orgGt: Ground truth for training image for DVC (np array: [num_image,x,y])
# test_ROSE_DVC_org: Testing image for DVC (np array: [num_image,x,y])
# test_ROSE_DVC_orgGt: Testing Ground Truth image for DVC (np array: [num_image,x,y])
#
# SVC_DVC dataset:
# train_ROSE_SDVC_org: Training image for SDVC (np array: [num_image,x,y])
# train_ROSE_SDVC_orgGt: Ground truth for training image for SDVC (np array: [num_image,x,y])
# test_ROSE_SDVC_org: Testing image for SDVC (np array: [num_image,x,y])
# test_ROSE_SDVC_orgGt: Testing Ground Truth image for SDVC (np array: [num_image,x,y])




from .image_processors import image_SingleChannel_converter, image_safe_read_tif # read tif



def load_rose_1_dataset(base_dir=None):
    from pathlib import Path # for setting up directory for reading
    import imageio.v3 as iio # for image reading
    import numpy as np # for creating array 


    # setup parent directory for reading
    current_code_path = Path(__file__).resolve().parent
    if base_dir is None:
        base_dir = current_code_path.parent / 'rose_dataset' / 'ROSE-1'
        image_parent_dir = base_dir


    # SVC dataset loading
    # setup SVC parent directory
    SVC_dir = image_parent_dir / 'SVC' 
    # setup SVC train and test image directory
    train_img_SVC_dir = SVC_dir / 'train' / 'img'
    test_img_SVC_dir = SVC_dir / 'test' / 'img'
    # setup SVC train and test groundtruth directory
    train_gt_SVC_dir = SVC_dir / 'train' / 'gt' # overall groundtruth
    train_thickgt_SVC_dir = SVC_dir / 'train' / 'thick_gt_converted' # thick vessel groundtruth
    train_thingt_SVC_dir = SVC_dir / 'train' / 'thin_gt' # thick vessel groundtruth
    test_gt_SVC_dir = SVC_dir / 'test' / 'gt' # overall groundtruth
    test_thickgt_SVC_dir = SVC_dir / 'test' / 'thick_gt_converted' # thick vessel groundtruth
    test_thingt_SVC_dir = SVC_dir / 'test' / 'thin_gt' # thick vessel groundtruth
    
    
    # read all images stored inside
    # train SVC image
    train_img_SVC_files = sorted(Path(train_img_SVC_dir).glob("*.tif"))
    train_SVC_data = [image_safe_read_tif(f) for f in train_img_SVC_files]
    train_ROSE_SVC_org = np.stack(train_SVC_data, axis=0)
    train_ROSE_SVC_org_dim = train_ROSE_SVC_org.ndim
    if train_ROSE_SVC_org_dim > 3:
        train_ROSE_SVC_org = image_SingleChannel_converter(train_ROSE_SVC_org)
        
    # test SVC image
    test_img_SVC_files = sorted(Path(test_img_SVC_dir).glob("*.tif"))
    test_SVC_data = [image_safe_read_tif(f) for f in test_img_SVC_files]
    test_ROSE_SVC_org = np.stack(test_SVC_data, axis=0)
    test_ROSE_SVC_org_dim = test_ROSE_SVC_org.ndim
    if test_ROSE_SVC_org_dim > 3:
        test_ROSE_SVC_org = image_SingleChannel_converter(test_ROSE_SVC_org)
        
    # train SVC overall groundtruth
    train_gt_SVC_files = sorted(Path(train_gt_SVC_dir).glob("*.tif"))
    train_SVC_gt = [image_safe_read_tif(f) for f in train_gt_SVC_files]
    train_ROSE_SVC_orgGt = np.stack(train_SVC_gt, axis=0)
    train_ROSE_SVC_orgGt_dim = train_ROSE_SVC_orgGt.ndim
    if train_ROSE_SVC_orgGt_dim > 3:
        train_ROSE_SVC_orgGt = image_SingleChannel_converter(train_ROSE_SVC_orgGt)
        
    # train SVC thick vessel groundtruth
    train_thickgt_SVC_files = sorted(Path(train_thickgt_SVC_dir).glob("*.tif"))
    train_SVC_thickgt = [image_safe_read_tif(f) for f in train_thickgt_SVC_files]
    train_ROSE_SVC_thickGt = np.stack(train_SVC_thickgt, axis=0)
    train_ROSE_SVC_thickGt_dim = train_ROSE_SVC_thickGt.ndim
    if train_ROSE_SVC_thickGt_dim > 3:
        train_ROSE_SVC_thickGt = image_SingleChannel_converter(train_ROSE_SVC_thickGt)
        
    # train SVC thin vessel groundtruth
    train_thingt_SVC_files = sorted(Path(train_thingt_SVC_dir).glob("*.tif"))
    train_SVC_thingt = [image_safe_read_tif(f) for f in train_thingt_SVC_files]
    train_ROSE_SVC_thinGt = np.stack(train_SVC_thingt, axis=0)
    train_ROSE_SVC_thinGt_dim = train_ROSE_SVC_thinGt.ndim
    if train_ROSE_SVC_thinGt_dim > 3:
        train_ROSE_SVC_thinGt = image_SingleChannel_converter(train_ROSE_SVC_thinGt)
        
    # test SVC overall groundtruth
    test_gt_SVC_files = sorted(Path(test_gt_SVC_dir).glob("*.tif"))
    test_SVC_gt = [image_safe_read_tif(f) for f in test_gt_SVC_files]
    test_ROSE_SVC_orgGt = np.stack(test_SVC_gt, axis=0)
    test_ROSE_SVC_orgGt_dim = test_ROSE_SVC_orgGt.ndim
    if test_ROSE_SVC_orgGt_dim > 3:
        test_ROSE_SVC_orgGt = image_SingleChannel_converter(test_ROSE_SVC_orgGt)
    
    # train SVC thick vessel groundtruth
    test_thickgt_SVC_files = sorted(Path(test_thickgt_SVC_dir).glob("*.tif"))
    test_SVC_thickgt = [image_safe_read_tif(f) for f in test_thickgt_SVC_files]
    test_ROSE_SVC_thickGt = np.stack(test_SVC_thickgt, axis=0)
    test_ROSE_SVC_thickGt_dim = test_ROSE_SVC_thickGt.ndim
    if test_ROSE_SVC_thickGt_dim > 3:
        test_ROSE_SVC_thickGt = image_SingleChannel_converter(test_ROSE_SVC_thickGt)
    
    # train SVC thin vessel groundtruth
    test_thingt_SVC_files = sorted(Path(test_thingt_SVC_dir).glob("*.tif"))
    test_SVC_thingt = [image_safe_read_tif(f) for f in test_thingt_SVC_files]
    test_ROSE_SVC_thinGt = np.stack(test_SVC_thingt, axis=0)
    test_ROSE_SVC_thinGt_dim = test_ROSE_SVC_thinGt.ndim
    if test_ROSE_SVC_thinGt_dim > 3:
        test_ROSE_SVC_thinGt = image_SingleChannel_converter(test_ROSE_SVC_thinGt)


    # DVC dataset loading
    # setup DVC parent directory
    DVC_dir = image_parent_dir / 'DVC' 
    # setup DVC train and test image directory
    train_img_DVC_dir = DVC_dir / 'train' / 'img'
    test_img_DVC_dir = DVC_dir / 'test' / 'img'
    # setup DVC train and test groundtruth directory
    train_gt_DVC_dir = DVC_dir / 'train' / 'gt' # overall groundtruth
    test_gt_DVC_dir = DVC_dir / 'test' / 'gt' # overall groundtruth
    
    # train DVC image
    train_img_DVC_files = sorted(Path(train_img_DVC_dir).glob("*.tif"))
    train_DVC_data = [image_safe_read_tif(f) for f in train_img_DVC_files]
    train_ROSE_DVC_org = np.stack(train_DVC_data, axis=0)
    train_ROSE_DVC_org_dim = train_ROSE_DVC_org.ndim
    if train_ROSE_DVC_org_dim > 3:
        train_ROSE_DVC_org = image_SingleChannel_converter(train_ROSE_DVC_org)
    
    # test DVC image
    test_img_DVC_files = sorted(Path(test_img_DVC_dir).glob("*.tif"))
    test_DVC_data = [image_safe_read_tif(f) for f in test_img_DVC_files]
    test_ROSE_DVC_org = np.stack(test_DVC_data, axis=0)
    test_ROSE_DVC_org_dim = test_ROSE_DVC_org.ndim
    if test_ROSE_DVC_org_dim > 3:
        test_ROSE_DVC_org = image_SingleChannel_converter(test_ROSE_DVC_org)
        
    # train DVC overall groundtruth
    train_gt_DVC_files = sorted(Path(train_gt_DVC_dir).glob("*.tif"))
    train_DVC_gt = [image_safe_read_tif(f) for f in train_gt_DVC_files]
    train_ROSE_DVC_orgGt = np.stack(train_DVC_gt, axis=0)
    train_ROSE_DVC_orgGt_dim = train_ROSE_DVC_orgGt.ndim
    if train_ROSE_DVC_orgGt_dim > 3:
        train_ROSE_DVC_orgGt = image_SingleChannel_converter(train_ROSE_DVC_orgGt)
    
    # test DVC overall groundtruth
    test_gt_DVC_files = sorted(Path(test_gt_DVC_dir).glob("*.tif"))
    test_DVC_gt = [image_safe_read_tif(f) for f in test_gt_DVC_files]
    test_ROSE_DVC_orgGt = np.stack(test_DVC_gt, axis=0)
    test_ROSE_DVC_orgGt_dim = test_ROSE_DVC_orgGt.ndim
    if test_ROSE_DVC_orgGt_dim > 3:
        test_ROSE_DVC_orgGt = image_SingleChannel_converter(test_ROSE_DVC_orgGt)


    # SDVC dataset loading
    # setup SDVC parent directory
    SDVC_dir = image_parent_dir / 'SVC_DVC' 
    # setup SDVC train and test image directory
    train_img_SDVC_dir = SDVC_dir / 'train' / 'img'
    test_img_SDVC_dir = SDVC_dir / 'test' / 'img'
    # setup SDVC train and test groundtruth directory
    train_gt_SDVC_dir = SDVC_dir / 'train' / 'gt' # overall groundtruth
    test_gt_SDVC_dir = SDVC_dir / 'test' / 'gt' # overall groundtruth
    
    # train SDVC image
    train_img_SDVC_files = sorted(Path(train_img_SDVC_dir).glob("*.png"))
    train_SDVC_data = [iio.imread(f) for f in train_img_SDVC_files]
    train_ROSE_SDVC_org = np.stack(train_SDVC_data, axis=0)
    train_ROSE_SDVC_org_dim = train_ROSE_SDVC_org.ndim
    if train_ROSE_SDVC_org_dim > 3:
        train_ROSE_SDVC_org = image_SingleChannel_converter(train_ROSE_SDVC_org)
    
    # test SDVC image
    test_img_SDVC_files = sorted(Path(test_img_SDVC_dir).glob("*.png"))
    test_SDVC_data = [iio.imread(f) for f in test_img_SDVC_files]
    test_ROSE_SDVC_org = np.stack(test_SDVC_data, axis=0)
    test_ROSE_SDVC_org_dim = test_ROSE_SDVC_org.ndim
    if test_ROSE_SDVC_org_dim > 3:
        test_ROSE_SDVC_org = image_SingleChannel_converter(test_ROSE_SDVC_org)
    
    # train SDVC overall groundtruth
    train_gt_SDVC_files = sorted(Path(train_gt_SDVC_dir).glob("*.tif"))
    train_SDVC_gt = [image_safe_read_tif(f) for f in train_gt_SDVC_files]
    train_ROSE_SDVC_orgGt = np.stack(train_SDVC_gt, axis=0)
    train_ROSE_SDVC_orgGt_dim = train_ROSE_SDVC_orgGt.ndim
    if train_ROSE_SDVC_orgGt_dim > 3:
        train_ROSE_SDVC_orgGt = image_SingleChannel_converter(train_ROSE_SDVC_orgGt)
    
    # test SDVC overall groundtruth
    test_gt_SDVC_files = sorted(Path(test_gt_SDVC_dir).glob("*.tif"))
    test_SDVC_gt = [image_safe_read_tif(f) for f in test_gt_SDVC_files]
    test_ROSE_SDVC_orgGt = np.stack(test_SDVC_gt, axis=0)
    test_ROSE_SDVC_orgGt_dim = test_ROSE_SDVC_orgGt.ndim
    if test_ROSE_SDVC_orgGt_dim > 3:
        test_ROSE_SDVC_orgGt = image_SingleChannel_converter(test_ROSE_SDVC_orgGt)
    
    
    return {"train_ROSE_SVC_org": train_ROSE_SVC_org,
            "train_ROSE_SVC_orgGt": train_ROSE_SVC_orgGt,
            "train_ROSE_SVC_thickGt": train_ROSE_SVC_thickGt,
            "train_ROSE_SVC_thinGt": train_ROSE_SVC_thinGt,
            "test_ROSE_SVC_org": test_ROSE_SVC_org,
            "test_ROSE_SVC_orgGt": test_ROSE_SVC_orgGt,
            "test_ROSE_SVC_thickGt": test_ROSE_SVC_thickGt,
            "test_ROSE_SVC_thinGt": test_ROSE_SVC_thinGt,
            "train_ROSE_DVC_org": train_ROSE_DVC_org,
            "train_ROSE_DVC_orgGt": train_ROSE_DVC_orgGt,
            "test_ROSE_DVC_org": test_ROSE_DVC_org,
            "test_ROSE_DVC_orgGt": test_ROSE_DVC_orgGt,
            "train_ROSE_SDVC_org": train_ROSE_SDVC_org,
            "train_ROSE_SDVC_orgGt": train_ROSE_SDVC_orgGt,
            "test_ROSE_SDVC_org": test_ROSE_SDVC_org,
            "test_ROSE_SDVC_orgGt": test_ROSE_SDVC_orgGt}
            
