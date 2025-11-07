# Created by Kuan-Min Lee
# Created date: Oct, 28th 2025 
# All rights reserved to Leelab.ai
# passed

# Brief User Introducttion:
# This script is constructed to conduct training for SVC coarse model

# ----- overall library -----
import os
import time
import torch # for model construction
from torch.utils.data import TensorDataset, DataLoader
import numpy as np # for variable storage construction
import matplotlib.pyplot as plt # for plotting

# ----- neural network source -----
from .Seg_enface_coarse_DVC_models import DVC_coarse_prototype_net_1

# ----- supplementary library -----
# import augmentation functions
from data_function_2D.image_rose_data_augmentator import img_flipper, img_elastic_deformer, img_contrast_jitter
# import random cropping
from .input_feature_blocks import random_crop_block
# import model checking function 
from .model_checking import register_nan_inf_hooks, must_be_finite
# import resume chacking function
from .resume_checking import CheckpointManager
from .losses import TverskyPlusMSE



# ----- building model function -----
def build_model():
    model = DVC_coarse_prototype_net_1()  # your network class
    return model



# ----- training function -----
def train_rose_1_DVC_coarse (overall_data, num_epoch, learning_rate, checkpoint_dir="./checkpoints", prefix="dvc_coarse", use_amp=True, use_aug=True, early_break_counter=10):
    # ----- setup device for running the training -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = None
    # scaler = torch.cuda.amp.GradScaler() if (use_amp and torch.cuda.is_available()) else None
    torch.backends.cudnn.benchmark = True
    
    # ----- setup loss function -----
    criterion = TverskyPlusMSE(alpha=0.7, beta=0.3, gamma=1.33, lam_tversky=1.0, lam_mse=0.1)
    
    # ----- extract out dataset -----
    augmentated_train_ROSE_DVC_org = overall_data["train_ROSE_DVC_org"]
    augmentated_train_ROSE_DVC_orgGt = overall_data["train_ROSE_DVC_orgGt"]
    
    # ----- grab out dimensional information -----
    num_files = augmentated_train_ROSE_DVC_org.shape[0]
    num_row = augmentated_train_ROSE_DVC_org.shape[1]
    num_col = augmentated_train_ROSE_DVC_org.shape[2]
    
    # ----- setupt the number of training and validation files -----
    num_folds = 4.0
    ratio = 1 / num_folds
    num_valid_files = int(num_files * ratio)
    num_train_files = num_files - num_valid_files
    start_valid_ind = 0
    end_valid_ind = num_valid_files
    
    # ----- check previously stored outcome -----
    net = build_model()
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=1e-4,fused=True)
    ckpt = CheckpointManager(net, optimizer, checkpoint_dir=checkpoint_dir, prefix=prefix)
    exist_file = ckpt.try_load()
    # Prepare defaults if nothing was loaded
    if exist_file is None:
        start_fold  = 0
        start_epoch = 0
        # make sure your loss buffers exist
        augmentated_train_loss = np.zeros((int(num_folds), int(num_epoch)), dtype=np.float32)
        augmentated_valid_loss = np.zeros((int(num_folds), int(num_epoch)), dtype=np.float32)
        best_val = float('inf')
    else:
        best_val, augmentated_train_loss, augmentated_valid_loss, start_fold, start_epoch = ckpt.load_data(net, optimizer)
        net = net.to(device).to(memory_format=torch.channels_last)

    # conduct training loop
    for i_fold in range(start_fold, int(num_folds)):
        # setupt index
        valid_index = np.arange(start_valid_ind, end_valid_ind)
        train_index = np.setdiff1d(np.arange(num_files), valid_index)
        
        # grab out current file
        current_valid_DVC_org = augmentated_train_ROSE_DVC_org[valid_index, :, :]
        current_valid_DVC_orgGt = augmentated_train_ROSE_DVC_orgGt[valid_index, :, :]
        current_train_DVC_org = augmentated_train_ROSE_DVC_org[train_index, :, :]
        current_train_DVC_orgGt = augmentated_train_ROSE_DVC_orgGt[train_index, :, :]
        
        # ----- Conduct data augmentation if wanted -----
        if use_aug == True:
            # conduct flipping augmentation
            current_augmentated_train_DVC_org, current_augmentated_train_DVC_orgGt = img_flipper(current_train_DVC_org, current_train_DVC_orgGt)    
            # conduct elastic deformation augmentation
            current_augmentated_train_DVC_org, current_augmentated_train_DVC_orgGt = img_elastic_deformer(current_augmentated_train_DVC_org, current_augmentated_train_DVC_orgGt)
            # conduct contrast jitter augmentation
            current_augmentated_train_DVC_org, current_augmentated_train_DVC_orgGt = img_contrast_jitter(current_augmentated_train_DVC_org, current_augmentated_train_DVC_orgGt)
        else:
            current_augmentated_train_DVC_org = current_train_DVC_org
            current_augmentated_train_DVC_orgGt = current_train_DVC_orgGt
        
        # ----- Datasets stay on CPU; move to GPU per-batch in the loop -----
        current_augmentated_train_DVC_org = torch.from_numpy(current_augmentated_train_DVC_org).float().unsqueeze(1)
        current_augmentated_train_DVC_orgGt = torch.from_numpy(current_augmentated_train_DVC_orgGt).float().unsqueeze(1)
        current_augmentated_train_DVC_data = TensorDataset(current_augmentated_train_DVC_org, current_augmentated_train_DVC_orgGt)
        current_augmentated_train_DVC_data = random_crop_block(current_augmentated_train_DVC_data, size=(192,192))   

        batchSize = 4
        current_augmentated_train_DVC_data = DataLoader(
            current_augmentated_train_DVC_data,
            batch_size=batchSize, shuffle=True, drop_last=True,
            pin_memory=True, num_workers=0, persistent_workers=False
        )        
        
        current_valid_DVC_org = torch.from_numpy(current_valid_DVC_org).float().unsqueeze(1)
        current_valid_DVC_orgGt = torch.from_numpy(current_valid_DVC_orgGt).float().unsqueeze(1)
        current_valid_DVC_data = TensorDataset(current_valid_DVC_org, current_valid_DVC_orgGt)

        batchSize = 4
        current_valid_DVC_data = DataLoader(
            current_valid_DVC_data,
            batch_size=batchSize, shuffle=False, drop_last=False,
            pin_memory=True, num_workers=0, persistent_workers=False
        )
        
        # ----- setup model to device (no extra .to('cuda')) -----
        if exist_file is None:
            net = build_model()
            net = net.to(device).to(memory_format=torch.channels_last)
        
            # ----- setup optimizer -----
            optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=1e-4, fused=True)
        
        # optional: gradient accumulation to simulate bigger batch
        acc_steps = 4  # set to 2–4 if you want an “effective” batch > 1
        break_counter = 0
        for i_epoch in range(start_epoch, (num_epoch)):
            if (i_epoch+1) == 1:
                start_time = time.time()
            #  --------------- training phase ---------------
            optimizer.zero_grad(set_to_none=True)
            current_epoch_loss = 0.0
            num_batches = 0
            net.train()
            for step, (current_file_DVC_org, current_file_DVC_orgGt) in enumerate(current_augmentated_train_DVC_data,1):
                # move dataset to device
                current_file_DVC_org = current_file_DVC_org.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
                current_file_DVC_orgGt = current_file_DVC_orgGt.to(device, non_blocking=True)
                
                # assure the groundtruth values are within range of [0,1]
                # map common encodings to [0,1]
                for tgt_name, tgt in [("Gt", current_file_DVC_orgGt)]:
                    # if it's 0/255, scale down:
                    mx = tgt.max().item()
                    if mx > 1.5 and mx <= 255.0:
                        tgt /= 255.0
                    # if there are ignore labels (e.g., 255), zero them out OR make a mask to exclude in loss
                    tgt.nan_to_num_(0.0)  # just in case
                    tgt.clamp_(0.0, 1.0)  # final safety
                    
                # ensure shape [B,1,H,W]
                if current_file_DVC_orgGt.dim() == 3:
                    current_file_DVC_orgGt = current_file_DVC_orgGt.unsqueeze(1)
                    
                # check if the current batch is empty
                with torch.no_grad():
                    gt = (current_file_DVC_orgGt > 0.5).float().mean().item()
                # if the current batch is almost empty skip
                if gt  < 0.01:
                    continue
                
                # ----- training phase -----
                # forward part
                amp_dtype = torch.bfloat16 if (use_amp and device.type=="cuda") else torch.float32
                with torch.autocast(device_type=device.type, dtype=amp_dtype if amp_dtype!=torch.float32 else torch.float32,
                                    enabled=(use_amp and device.type=="cuda")):
                    current_pred_DVC_orgGt = net(current_file_DVC_org)   
                    
                # backward part
                with torch.autocast(device_type=device.type, enabled=False):
                    current_loss = (criterion(current_pred_DVC_orgGt.float(), current_file_DVC_orgGt.float()))
                    must_be_finite("loss", current_loss)
                k = max(1, int(0.5 * current_loss.numel()))
                current_loss_scalar = current_loss.topk(k, largest=True).values.mean()
                (current_loss_scalar / acc_steps).backward()  
                
                # Gradient accumulation update
                if step % acc_steps == 0:
                    optimizer.step()         # normal full precision step
                    optimizer.zero_grad(set_to_none=True)  # reset accumulated gradients
                       
                # accumulate (multiply by batch size only; acc_steps already divided above)
                current_epoch_loss += current_loss_scalar.item() 
                num_batches += 1
                
            # after the batch loop:
            if (step % acc_steps) == 0: 
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                
            # store training loss
            current_epoch_loss /= max(1, num_batches)
            
            if (i_epoch+1) % 10 == 0:
                print(f"Current Epoch: {i_epoch+1} | Loss value {current_epoch_loss}")
            
            # ----- validation phase -----
            net.eval()
            current_epoch_valid_loss = 0.0
            num_val_batches = 0
            with torch.inference_mode():
                for current_valid_DVC_org, current_valid_DVC_orgGt in current_valid_DVC_data:
                    # move dataset to device
                    current_valid_DVC_org = current_valid_DVC_org.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
                    current_valid_DVC_orgGt = current_valid_DVC_orgGt.to(device, non_blocking=True)
                    
                    # assure the groundtruth values are within range of [0,1]
                    # map common encodings to [0,1]
                    for tgt_name, tgt in [("Gt", current_valid_DVC_orgGt)]:
                        # if it's 0/255, scale down:
                        mx = torch.as_tensor(tgt.max(), device=tgt.device)
                        if mx > 1.5 and mx <= 255.0:
                            tgt /= 255.0
                        # if there are ignore labels (e.g., 255), zero them out OR make a mask to exclude in loss
                        tgt.nan_to_num_(0.0)  # just in case
                        tgt.clamp_(0.0, 1.0)  # final safety
                    
                    # ensure shape [B,1,H,W]
                    if current_valid_DVC_orgGt.dim() == 3:
                        current_valid_DVC_orgGt = current_valid_DVC_orgGt.unsqueeze(1)
                        
                    # run validation
                    with torch.autocast(device_type=device.type, dtype=amp_dtype if amp_dtype!=torch.float32 else torch.float32,
                                        enabled=(use_amp and device.type=="cuda")):
                        current_pred_valid_DVC_orgGt = net(current_valid_DVC_org)                           
                    with torch.autocast(device_type=device.type, enabled=False):
                        current_valid_loss = criterion(current_pred_valid_DVC_orgGt.float(), current_valid_DVC_orgGt.float())
                    current_epoch_valid_loss += current_valid_loss.sum().item()  
                    
                    num_val_batches += 1
                        
            # store validation loss
            current_epoch_valid_loss /= max(1, num_val_batches)
            
            if (i_epoch+1) % 10 == 0:
                print(f"Current Epoch: {i_epoch+1} | Validation Loss value {current_epoch_valid_loss}")
                    
            # print out epoch time
            if (i_epoch+1) % 10 == 0:
                end_time = time.time()
                epoch_time = end_time - start_time
                print(f"Epoch [{i_epoch+1}/{num_epoch}] finished in {epoch_time:.2f} seconds ({epoch_time/60:.2f} minutes)")
                start_time = time.time()
            
            # store all losses
            augmentated_train_loss[i_fold, i_epoch] = current_epoch_loss
            augmentated_valid_loss[i_fold, i_epoch] = current_epoch_valid_loss
            
            # ---- save "best" when val improves ----
            if (current_epoch_valid_loss < best_val) or (i_epoch == 1):
                best_val = current_epoch_valid_loss
                ckpt.save_model(
                    net=net, optimizer=optimizer,
                    learning_rate=optimizer.param_groups[0]["lr"],
                    scaler=scaler, best_val=best_val, tag="best"
                )
                break_counter = 0
            # implement an early break feature
            else:
                if break_counter >= early_break_counter:
                    print("Early Break Activated")
                    print(f"Fold [{i_fold+1}/{num_folds}] | Epoch [{i_epoch+1}/{num_epoch}] "
                          f"| Train Loss: {current_epoch_loss:.4f} | "
                          f"Val Loss: {current_epoch_valid_loss:.4f}")
                    break
                else:
                    break_counter+=1
                    
                    
            # Print progress every 10 epochs (and on first)
            if (i_epoch + 1) % 10 == 0:
                print(f"Fold [{i_fold+1}/{num_folds}] | Epoch [{i_epoch+1}/{num_epoch}] "
                      f"| Train Loss: {current_epoch_loss:.4f} | "
                      f"Val Loss: {current_epoch_valid_loss:.4f}")
            
            # save all the training parameters
            ckpt.save_train_param(
                i_fold=i_fold, i_epoch=i_epoch+1,
                fold_train_loss=augmentated_train_loss,
                fold_valid_loss=augmentated_valid_loss,
                num_epoch=num_epoch,
            )
        
        # update validation start and end indices
        start_valid_ind = start_valid_ind + num_valid_files
        end_valid_ind = end_valid_ind + num_valid_files 
        
        # update the best model
        if i_fold == 0:
            best_model = net
            current_fold_valid_loss = augmentated_valid_loss[i_fold,:]
            current_fold_valid_loss = current_fold_valid_loss[current_fold_valid_loss != 0]
            best_val = np.min(current_fold_valid_loss)
        else:
            current_fold_valid_loss = augmentated_valid_loss[i_fold,:]
            current_fold_valid_loss = current_fold_valid_loss[current_fold_valid_loss != 0]
            current_best_valid_loss = np.min(current_fold_valid_loss)
            if i_fold == 1:
                prev_fold_valid_loss = augmentated_valid_loss[0,:]
                prev_fold_valid_loss = prev_fold_valid_loss[prev_fold_valid_loss != 0]
                prev_best_valid_loss = np.min(prev_fold_valid_loss)
            else:
                prev_fold_valid_loss = augmentated_valid_loss[0:(i_fold-1),:]
                prev_fold_valid_loss = prev_fold_valid_loss[prev_fold_valid_loss != 0]
                prev_best_valid_loss = np.min(prev_fold_valid_loss)
            if current_best_valid_loss < prev_best_valid_loss:
                best_model = net
        
        # update exist_file
        exist_file = None
        start_epoch = 0

    # plot the final traing loss plot
    plt.figure(figsize=(9, 6))
    ax = plt.gca()
    for i in range(int(num_folds)):
       # grab out data
        train_loss = augmentated_train_loss[i]
        train_loss = train_loss[train_loss != 0]
        valid_loss = augmentated_valid_loss[i]
        valid_loss = valid_loss[valid_loss != 0]
        
        # plot the data
        if train_loss.any():
            ax.plot(
                np.flatnonzero(train_loss) + 1,
                train_loss,
                label=f"Train Fold {i+1}",
                linestyle='-'
            )
        if valid_loss.any():
            ax.plot(
                np.flatnonzero(valid_loss) + 1,
                valid_loss,
                label=f"Valid Fold{i+1}",
                linestyle='--'
            )
        
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), ncol=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Across Folds')
    plt.grid(True)
    plt.tight_layout()
    
    # save figure
    model_name = type(net).__name__
    augmentated = "flip_elastic_contrast"
    epoch_num = "_e200"
    fold_num = "_f4"
    figure_type = ".png"
    full_figure_name = model_name + augmentated + epoch_num + fold_num +figure_type
    plt.savefig(full_figure_name)

    # show figure
    plt.show(full_figure_name)
    
    # save final outcome
    net_name = best_model.__class__.__name__
    torch.save(best_model.state_dict(), f"trained_{net_name}.pth")
    
    return best_model


