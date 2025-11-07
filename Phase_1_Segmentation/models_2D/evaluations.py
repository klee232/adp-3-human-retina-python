# Created by Kuan-Min Lee
# Created date: Nov, 6th 2025 
# All rights reserved to Leelab.ai
# still need tested

# Brief User Introduction:
# This script is created for storing different functions used for testing trained neural networks


import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, average_precision_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
from tqdm import tqdm



# ----- evaluation functions -----
# area under curves function
# Brief User Introduction:
# Computes ROC curve and AUC for a binary classification / segmentation model.
# ----- Input parameters: -----
# targets: groundtruth
# preds: prediction from model
# save_path (str, optional): if you want to save the ROC figure
# ----- Output parameters: -----
# fpr, tpr, auc_value
def evaluate_area_under_curves(targets, preds, save_path=None):
    # ----- Compute ROC Curve -----
    fpr, tpr, _ = roc_curve(targets, preds)
    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, roc_auc
    
def evaluate_plot_roc_pair(save_path, model, thick_fpr, thick_tpr, thick_auc, thin_fpr, thin_tpr, thin_auc):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    # Thick
    ax = axes[0]
    ax.plot(thick_fpr, thick_tpr, label=f"AUC={thick_auc:.4f}")
    ax.plot([0,1], [0,1], linestyle="--", linewidth=1)
    ax.set_title("ROC — SVC Thick"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend(); ax.grid(True)
    # Thin
    ax = axes[1]
    ax.plot(thin_fpr, thin_tpr, label=f"AUC={thin_auc:.4f}")
    ax.plot([0,1], [0,1], linestyle="--", linewidth=1)
    ax.set_title("ROC — SVC Thin"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend(); ax.grid(True)

    os.makedirs(save_path, exist_ok=True)
    filename = model.__class__.__name__ + "_roc"
    full_save_path = os.path.join(save_path, filename)
    fig.savefig(full_save_path, dpi=200)
    plt.close(fig)

def evaluate_plot_roc(save_path, model, fpr, tpr, auc):
    import os
    import matplotlib.pyplot as plt

    # Create output directory if needed
    os.makedirs(save_path, exist_ok=True)

    # Create a single ROC plot for SVC Thick
    fig, ax = plt.subplots(figsize=(5, 4))

    ax.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_title("ROC — DVC")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend()
    ax.grid(True)

    # Save as <save_path>/<filename>.png
    os.makedirs(save_path, exist_ok=True)
    filename = model.__class__.__name__ + "_roc"
    full_save_path = os.path.join(save_path, filename + ".png")
    fig.savefig(full_save_path, dpi=200)
    plt.close(fig)

# confusion function
# Brief User Introduction:
# Computes confusion matrix.
# ----- Input parameters: -----
# targets: groundtruth
# preds: prediction from model
# save_path (str, optional): if you want to save the ROC figure
# ----- Output parameters: -----
# TP, FP, TN, FN
def evaluate_confusion_matrix(targets, preds):
    
    TP = np.sum((preds == 1) & (targets == 1))
    FP = np.sum((preds == 1) & (targets == 0))
    TN = np.sum((preds == 0) & (targets == 0))
    FN = np.sum((preds == 0) & (targets == 1))
    
    return TP, FP, TN, FN

def evaluate_plot_confusion_pair(save_path, model, thick_TP, thick_FP, thick_TN, thick_FN, thin_TP, thin_FP, thin_TN, thin_FN):
    import numpy as np
    thick_mat = np.array([[thick_TN, thick_FP],
                          [thick_FN, thick_TP]])
    thin_mat  = np.array([[thin_TN,  thin_FP],
                          [thin_FN,  thin_TP]])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, mat, title in zip(
        axes, (thick_mat, thin_mat), ("Confusion — SVC Thick", "Confusion — SVC Thin")
    ):
        im = ax.imshow(mat, cmap="Blues")
        for (i, j), v in np.ndenumerate(mat):
            ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=10)
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Pred 0","Pred 1"])
        ax.set_yticklabels(["True 0","True 1"])
        ax.set_title(title)
        ax.grid(False)
    os.makedirs(save_path, exist_ok=True)
    filename = model.__class__.__name__ + "_confusion_matrix"
    full_save_path = os.path.join(save_path, filename)
    fig.savefig(full_save_path, dpi=200)
    plt.close(fig)
    
def evaluate_plot_confusion(save_path, model, TP, FP, TN, FN):
    # Build confusion matrix for SVC Thick
    mat = np.array([[TN, FP],
                    [FN, TP]])

    # Plot only this confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(mat, cmap="Blues")

    # Annotate each cell with its number
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=10)

    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])
    ax.set_title("Confusion Matrix — DVC")
    ax.grid(False)

    # Save figure
    os.makedirs(save_path, exist_ok=True)
    filename = model.__class__.__name__ + "_confusion_matrix"
    full_save_path = os.path.join(save_path, filename + ".png")
    fig.savefig(full_save_path, dpi=200)
    plt.close(fig)

    
# Used after confustion matrix function  
# ----- sensitivity function ----- 
# ----- Input parameters: -----
# TP: True Positive
# FP: False Positive
# TN: True Negative
# FN: False Negative
# ----- Output parameters: -----
# sensitivity
def evaluate_sensitivity(TP, FP, TN, FN, eps=1e-8):
    return TP / (TP + FN + eps)
    
# ----- specificity function ----- 
# ----- Input parameters: -----
# TP: True Positive
# FP: False Positive
# TN: True Negative
# FN: False Negative
# ----- Output parameters: -----
# specificity
def evaluate_specificity(TP, FP, TN, FN, eps=1e-8):
    return TN / (TN + FP + eps)

# ----- Accuracy function ----- 
# ----- Input parameters: -----
# TP: True Positive
# FP: False Positive
# TN: True Negative
# FN: False Negative
# ----- Output parameters: -----
# accuracy
def evaluate_accuracy(TP, FP, TN, FN, eps=1e-8):
    return (TP + TN) / (TP + TN + FP + FN + eps)

# ----- precision function ----- 
# ----- Input parameters: -----
# TP: True Positive
# FP: False Positive
# TN: True Negative
# FN: False Negative
# ----- Output parameters: -----
# precision
def evaluate_precision(TP, FP, TN, FN, eps=1e-8):
    return TP / (TP + FP + eps)

# ----- f1 score function ----- 
# ----- Input parameters: -----
# TP: True Positive
# FP: False Positive
# TN: True Negative
# FN: False Negative
# ----- Output parameters: -----
# f1 score
def evaluate_f1_score(TP, FP, TN, FN, eps=1e-8):
    p = evaluate_precision(TP, FP, TN, FN, eps)
    r = evaluate_sensitivity(TP, FP, TN, FN, eps)
    return 2 * p * r / (p + r + eps)

# ----- dice score function ----- 
# ----- Input parameters: -----
# TP: True Positive
# FP: False Positive
# TN: True Negative
# FN: False Negative
# ----- Output parameters: -----
# dice score
def evaluate_dice_score(TP, FP, TN, FN, eps=1e-8):
    return 2 * TP / (2 * TP + FP + FN + eps)

# ----- IoU (Intersection over Union / Jaccard Index) function ----- 
# ----- Input parameters: -----
# TP: True Positive
# FP: False Positive
# TN: True Negative
# FN: False Negative
# ----- Output parameters: -----
# IoU score
def evaluate_iou(TP, FP, TN, FN, eps=1e-8):
    return TP / (TP + FP + FN + eps)

# ----- G-Mean (Geometric Mean of Sensitivity and Specificity) function ----- 
# ----- Input parameters: -----
# TP: True Positive
# FP: False Positive
# TN: True Negative
# FN: False Negative
# ----- Output parameters: -----
# G-Mean
def evaluate_g_mean(TP, FP, TN, FN, eps=1e-8):
    se = evaluate_sensitivity(TP, FP, TN, FN, eps)
    sp = evaluate_specificity(TP, FP, TN, FN, eps)
    return (se * sp) ** 0.5

# ----- MCC (Matthews Correlation Coefficient) function ----- 
# ----- Input parameters: -----
# TP: True Positive
# FP: False Positive
# TN: True Negative
# FN: False Negative
# ----- Output parameters: -----
# MCC (Matthews Correlation Coefficient)
def evaluate_mcc(TP, FP, TN, FN, eps=1e-8):
    num = (TP * TN) - (FP * FN)
    den = ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + eps) ** 0.5
    return num / (den + eps)