# Created by Kuan-Min Lee
# Created date: Nov, 6th 2025 
# All rights reserved to Leelab.ai

# Brief User Introducttion:
# This script is constructed to conduct testing for SVC coarse model

# ----- overall library -----
import os
import torch # for model construction
from torch.utils.data import TensorDataset, DataLoader
import numpy as np # for variable storage construction
import matplotlib.pyplot as plt # for plotting
import pandas as pd


# ----- supplementary library -----
# import evaluation functions
from .evaluations import (evaluate_area_under_curves, 
                          evaluate_plot_roc,
                          evaluate_confusion_matrix, 
                          evaluate_plot_confusion,
                          evaluate_sensitivity, 
                          evaluate_specificity, 
                          evaluate_accuracy, 
                          evaluate_precision, 
                          evaluate_f1_score, 
                          evaluate_dice_score, 
                          evaluate_iou, 
                          evaluate_g_mean, 
                          evaluate_mcc)



def test_rose_1_DVC_coarse (model, overall_data, threshold=0.5, use_amp=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- grab out testing data -----
    test_ROSE_DVC_org = overall_data["test_ROSE_DVC_org"]
    test_ROSE_DVC_orgGt = overall_data["test_ROSE_DVC_orgGt"]
    
    # ----- form data builder -----
    test_DVC_org = torch.from_numpy(test_ROSE_DVC_org).float().unsqueeze(1)
    test_DVC_orgGt = torch.from_numpy(test_ROSE_DVC_orgGt).float().unsqueeze(1)
    test_DVC_data = TensorDataset(test_DVC_org, test_DVC_orgGt)
    test_DVC_data = DataLoader(test_DVC_data, batch_size=1, shuffle=False, drop_last=False,
                               pin_memory=True, num_workers=0, persistent_workers=False)

    # ----- initiate model to testing mode -----
    model.to(device).eval()

    # ----- conduct prediction -----
    num_batch = 0
    rows = []
    example = 1
    with torch.no_grad():
        for _, (test_DVC_org, test_DVC_orgGt) in enumerate(test_DVC_data):
            test_DVC_org = test_DVC_org.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            test_DVC_orgGt = test_DVC_orgGt.to(device, non_blocking=True)
            name = "test_image_" + str(example)
            
            # assure the groundtruth values are within range of [0,1]
            # map common encodings to [0,1]
            for tgt_name, tgt in [("Gt", test_DVC_orgGt)]:
                # if it's 0/255, scale down:
                mx = torch.as_tensor(tgt.max(), device=tgt.device)
                if mx > 1.5 and mx <= 255.0:
                    tgt /= 255.0
                # if there are ignore labels (e.g., 255), zero them out OR make a mask to exclude in loss
                tgt.nan_to_num_(0.0)  # just in case
                tgt.clamp_(0.0, 1.0)  # final safety
            
            # ensure shape [B,1,H,W]
            if test_DVC_orgGt.dim() == 3:
                test_DVC_orgGt = test_DVC_orgGt.unsqueeze(1)
            
            # run forward
            amp_dtype = torch.bfloat16 if (use_amp and device.type=="cuda") else torch.float32
            with torch.autocast(device_type=device.type, dtype=amp_dtype if amp_dtype!=torch.float32 else torch.float32,
                                enabled=(use_amp and device.type=="cuda")):
                test_DVC_pred = model(test_DVC_org)           

            # compute binary results
            if threshold is not None:
                test_DVC_pred = (test_DVC_pred >= threshold).float()
            else:
                test_DVC_pred = test_DVC_pred
                
            # update batch number
            num_batch += 1
            
            # ----- conduct evaluation -----
            # AUC
            test_DVC_orgGt = test_DVC_orgGt.detach().float().cpu().numpy().ravel()
            test_DVC_pred = test_DVC_pred.detach().float().cpu().numpy().ravel()
            fpr, tpr, auc = evaluate_area_under_curves(test_DVC_orgGt,test_DVC_pred)
            savePath = "coarse_results_AUC"
            evaluate_plot_roc(savePath, model, fpr, tpr, auc)

            # Confusion matrix
            TP, FP, TN, FN = evaluate_confusion_matrix(test_DVC_orgGt,test_DVC_pred)
            savePath = "coarse_results_confusion_matrix"
            evaluate_plot_confusion(savePath, model, TP, FP, TN, FN)
            
            # Sensitivity
            sensitivity = evaluate_sensitivity(TP, FP, TN, FN, eps=1e-8)

            # Specificity
            specificity = evaluate_specificity(TP, FP, TN, FN, eps=1e-8)

            # Accuracy
            acc = evaluate_accuracy(TP, FP, TN, FN, eps=1e-8)

            # Precision
            precision = evaluate_precision(TP, FP, TN, FN, eps=1e-8)

            # F1 score
            f1 = evaluate_f1_score(TP, FP, TN, FN, eps=1e-8)

            # Dice score
            dice = evaluate_dice_score(TP, FP, TN, FN, eps=1e-8)
   
            # IoU
            IoU = evaluate_iou(TP, FP, TN, FN, eps=1e-8)

            # G-mean
            gmean = evaluate_g_mean(TP, FP, TN, FN, eps=1e-8)

            # MCC
            mcc = evaluate_mcc(TP, FP, TN, FN, eps=1e-8)
            
            # ----- create a table for storage -----
            # ----- append a row for CSV -----
            rows.append({
                "image": name,
                # AUCs
                "thick_auc": auc, 
                # thick metrics
                "Sensitivity": sensitivity, "Specificity": specificity, "accuracy": acc,
                "Precision": precision, "F1": f1, "Dice": dice,
                "IoU": IoU, "GMean": gmean, "MCC": mcc,
                # confusion counts (optional)
                "TP": TP, "FP": FP, "TN": TN, "FN": FN,
            })
            
            example += 1
            
    # ----- save table when finished -----
    output_folder = "coarse_results_table"
    os.makedirs(output_folder, exist_ok=True)
    table_path = os.path.join(output_folder, model.__class__.__name__+"results.csv")
    df = pd.DataFrame(rows)
    df.to_csv(table_path, index=False)
    print(f"[OK] Saved {len(rows)} rows to {table_path} and figures")
    return df
    