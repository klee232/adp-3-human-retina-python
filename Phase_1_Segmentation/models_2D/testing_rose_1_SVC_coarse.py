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
                          evaluate_plot_roc_pair,
                          evaluate_confusion_matrix, 
                          evaluate_plot_confusion_pair,
                          evaluate_sensitivity, 
                          evaluate_specificity, 
                          evaluate_accuracy, 
                          evaluate_precision, 
                          evaluate_f1_score, 
                          evaluate_dice_score, 
                          evaluate_iou, 
                          evaluate_g_mean, 
                          evaluate_mcc)



def test_rose_1_SVC_coarse (model, overall_data, threshold=0.5, use_amp=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- grab out testing data -----
    test_ROSE_SVC_org = overall_data["test_ROSE_SVC_org"]
    test_ROSE_SVC_thickGt = overall_data["test_ROSE_SVC_thickGt"]
    test_ROSE_SVC_thinGt = overall_data["test_ROSE_SVC_thinGt"]
    
    # ----- form data builder -----
    test_SVC_org = torch.from_numpy(test_ROSE_SVC_org).float().unsqueeze(1)
    test_SVC_thickGt = torch.from_numpy(test_ROSE_SVC_thickGt).float().unsqueeze(1)
    test_SVC_thinGt = torch.from_numpy(test_ROSE_SVC_thinGt).float().unsqueeze(1)
    test_SVC_data = TensorDataset(test_SVC_org, test_SVC_thickGt, test_SVC_thinGt)
    test_SVC_data = DataLoader(test_SVC_data, batch_size=1, shuffle=False, drop_last=False,
                               pin_memory=True, num_workers=0, persistent_workers=False)

    # ----- initiate model to testing mode -----
    model.to(device).eval()

    # ----- conduct prediction -----
    num_batch = 0
    rows = []
    example = 1
    with torch.no_grad():
        for _, (test_SVC_org, test_SVC_thickGt, test_SVC_thinGt) in enumerate(test_SVC_data):
            test_SVC_org = test_SVC_org.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            test_SVC_thickGt = test_SVC_thickGt.to(device, non_blocking=True)
            test_SVC_thinGt = test_SVC_thinGt.to(device, non_blocking=True)
            name = "test_image_" + str(example)
            
            # assure the groundtruth values are within range of [0,1]
            # map common encodings to [0,1]
            for tgt_name, tgt in [("thick", test_SVC_thickGt), ("thin", test_SVC_thinGt)]:
                # if it's 0/255, scale down:
                mx = torch.as_tensor(tgt.max(), device=tgt.device)
                if mx > 1.5 and mx <= 255.0:
                    tgt /= 255.0
                # if there are ignore labels (e.g., 255), zero them out OR make a mask to exclude in loss
                tgt.nan_to_num_(0.0)  # just in case
                tgt.clamp_(0.0, 1.0)  # final safety
            
            # ensure shape [B,1,H,W]
            if test_SVC_thickGt.dim() == 3:
                test_SVC_thickGt = test_SVC_thickGt.unsqueeze(1)
            if test_SVC_thinGt.dim() == 3:
                test_SVC_thinGt = test_SVC_thinGt.unsqueeze(1)
            
            # run forward
            amp_dtype = torch.bfloat16 if (use_amp and device.type=="cuda") else torch.float32
            with torch.autocast(device_type=device.type, dtype=amp_dtype if amp_dtype!=torch.float32 else torch.float32,
                                enabled=(use_amp and device.type=="cuda")):
                test_SVC_pred_thick, test_SVC_pred_thin = model(test_SVC_org)           

            # compute binary results
            if threshold is not None:
                test_SVC_pred_thick = (test_SVC_pred_thick >= threshold).float()
                test_SVC_pred_thin = (test_SVC_pred_thin >= threshold).float()
            else:
                test_SVC_pred_thick = test_SVC_pred_thick
                test_SVC_pred_thin = test_SVC_pred_thin
                
            # update batch number
            num_batch += 1
            
            # ----- conduct evaluation -----
            # AUC
            test_SVC_thickGt = test_SVC_thickGt.detach().float().cpu().numpy().ravel()
            test_SVC_pred_thick = test_SVC_pred_thick.detach().float().cpu().numpy().ravel()
            test_SVC_thinGt = test_SVC_thinGt.detach().float().cpu().numpy().ravel()
            test_SVC_pred_thin = test_SVC_pred_thin.detach().float().cpu().numpy().ravel()
            thick_fpr, thick_tpr, thick_auc = evaluate_area_under_curves(test_SVC_thickGt,test_SVC_pred_thick)
            thin_fpr, thin_tpr, thin_auc = evaluate_area_under_curves(test_SVC_thinGt,test_SVC_pred_thin)
            savePath = "coarse_results_AUC"
            evaluate_plot_roc_pair(savePath, model, thick_fpr, thick_tpr, thick_auc, thin_fpr, thin_tpr, thin_auc)

            # Confusion matrix
            thick_TP, thick_FP, thick_TN, thick_FN = evaluate_confusion_matrix(test_SVC_thickGt,test_SVC_pred_thick)
            thin_TP, thin_FP, thin_TN, thin_FN = evaluate_confusion_matrix(test_SVC_thinGt,test_SVC_pred_thin)
            savePath = "coarse_results_confusion_matrix"
            evaluate_plot_confusion_pair(savePath, model, thick_TP, thick_FP, thick_TN, thick_FN, thin_TP, thin_FP, thin_TN, thin_FN)
            
            # Sensitivity
            thick_sensitivity = evaluate_sensitivity(thick_TP, thick_FP, thick_TN, thick_FN, eps=1e-8)
            thin_sensitivity = evaluate_sensitivity(thin_TP, thin_FP, thin_TN, thin_FN, eps=1e-8)

            # Specificity
            thick_specificity = evaluate_specificity(thick_TP, thick_FP, thick_TN, thick_FN, eps=1e-8)
            thin_specificity = evaluate_specificity(thin_TP, thin_FP, thin_TN, thin_FN, eps=1e-8)

            # Accuracy
            thick_acc = evaluate_accuracy(thick_TP, thick_FP, thick_TN, thick_FN, eps=1e-8)
            thin_acc = evaluate_accuracy(thin_TP, thin_FP, thin_TN, thin_FN, eps=1e-8)

            # Precision
            thick_precision = evaluate_precision(thick_TP, thick_FP, thick_TN, thick_FN, eps=1e-8)
            thin_precision = evaluate_precision(thin_TP, thin_FP, thin_TN, thin_FN, eps=1e-8)

            # F1 score
            thick_f1 = evaluate_f1_score(thick_TP, thick_FP, thick_TN, thick_FN, eps=1e-8)
            thin_f1 = evaluate_f1_score(thin_TP, thin_FP, thin_TN, thin_FN, eps=1e-8)

            # Dice score
            thick_dice = evaluate_dice_score(thick_TP, thick_FP, thick_TN, thick_FN, eps=1e-8)
            thin_dice = evaluate_dice_score(thin_TP, thin_FP, thin_TN, thin_FN, eps=1e-8)
   
            # IoU
            thick_IoU = evaluate_iou(thick_TP, thick_FP, thick_TN, thick_FN, eps=1e-8)
            thin_IoU = evaluate_iou(thin_TP, thin_FP, thin_TN, thin_FN, eps=1e-8)

            # G-mean
            thick_gmean = evaluate_g_mean(thick_TP, thick_FP, thick_TN, thick_FN, eps=1e-8)
            thin_gmean = evaluate_g_mean(thin_TP, thin_FP, thin_TN, thin_FN, eps=1e-8)

            # MCC
            thick_mcc = evaluate_mcc(thick_TP, thick_FP, thick_TN, thick_FN, eps=1e-8)
            thin_mcc = evaluate_mcc(thin_TP, thin_FP, thin_TN, thin_FN, eps=1e-8)
            
            # ----- create a table for storage -----
            # ----- append a row for CSV -----
            rows.append({
                "image": name,
                # AUCs
                "thick_auc": thick_auc, "thin_auc": thin_auc,
                # thick metrics
                "thick_Specificity": thick_sensitivity, "thick_SP": thick_specificity, "thick_ACC": thick_acc,
                "thick_Prec": thick_precision, "thick_F1": thick_f1, "thick_Dice": thick_dice,
                "thick_IoU": thick_IoU, "thick_GMean": thick_gmean, "thick_MCC": thick_mcc,
                # thin metrics
                "thin_SE": thin_sensitivity, "thin_SP": thin_specificity, "thin_ACC": thin_acc,
                "thin_Prec": thin_precision, "thin_F1": thin_f1, "thin_Dice": thin_dice,
                "thin_IoU": thin_IoU, "thin_GMean": thin_gmean, "thin_MCC": thin_mcc,
                # confusion counts (optional)
                "thick_TP": thick_TP, "thick_FP": thick_FP, "thick_TN": thick_TN, "thick_FN": thick_FN,
                "thin_TP": thin_TP,   "thin_FP": thin_FP,   "thin_TN": thin_TN,   "thin_FN": thin_FN,
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
    