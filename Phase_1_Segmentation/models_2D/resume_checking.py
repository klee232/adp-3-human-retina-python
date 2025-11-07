# Created by Kuan-Min Lee
# Created date: Oct, 28th 2025 
# All rights reserved to Leelab.ai
# passed

# Brief User Introducttion:
# This script is constructed to retrieve all saved intermediate files for training



import os, json
from pathlib import Path
import torch
import numpy as np



class CheckpointManager:
    # setup directory for storing checkpoints
    def __init__(self, net, optimizer, checkpoint_dir="./checkpoints", prefix="svc_coarse"):
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.model_name = type(net).__name__
        self.optimizer_name = type(optimizer).__name__
        

    # retrieve checkpint directory
    def _model_path(self, tag="last"):
        return self.dir / f"{self.prefix}_{self.model_name}_{self.optimizer_name}_{tag}.pt"

    # retrieve checkpint directory
    def _param_path(self, tag="last"):
        return self.dir / f"{self.prefix}_{self.model_name}_training_and_valid_loss_{tag}.pt"

    # saving all intermediate files
    def save_model(self, *, net, optimizer, learning_rate, best_val, scaler=None, tag="last"):
        # create state variable to store all necessary variables
        state = {
            "model_state": net.state_dict(),
            "model_name": type(net).__name__,
            "optim_state": optimizer.state_dict(),
            "optimizer_name": type(optimizer).__name__,
            "learning_rate": learning_rate,
            "best_val": best_val
        }
        
        # save scaler if not none
        if scaler is not None:
            state["scaler_state"] = scaler.state_dict()
        torch.save(state, self._model_path(tag))
        
        # also write a small JSON stub (human-readable)
        # this is a text file we created for human-readable only
        with open(self.dir / f"{self.prefix}_{tag}.json", "w") as f:
            json.dump({"model_name": type(net).__name__, 
                       "optimizer_name": type(optimizer).__name__,
                       "learning_rate": learning_rate,
                       "best_val": best_val}, f, indent=2)
                       
    
    # saving all loss files
    def save_train_param(self, *, i_fold, i_epoch, fold_train_loss, fold_valid_loss, num_epoch):
        # create state variable to store all loss values
        state = {
            "i_fold": i_fold,
            "i_epoch": i_epoch,
            "fold_train_loss": fold_train_loss,
            "fold_valid_loss": fold_valid_loss,
            "num_epoch": num_epoch
        }
        
        # save the state variable
        tag='last'
        torch.save(state, self._param_path(tag))
        
        # also write a small JSON stub (human-readable)
        # this is a text file we created for human-readable only
        with open(self.dir / f"{self.prefix}_{tag}.json", "w") as f:
            json.dump({"i_fold": i_fold, 
                       "i_epoch": i_epoch, 
                       "num_epoch": num_epoch}, f, indent=2)

    # loading detector
    def try_load(self, model_tag='best', tag='last'):
        model_p = self._model_path(model_tag)
        # return none if the directory not found
        if not model_p.exists():
            return None
            
        param_p = self._param_path(tag) 
        # return non if the directory not found
        if not param_p.exists():
            return None
        
        return True


    # loading all intermediate files
    def load_data(self, net, optimizer, scaler=None, model_tag="best", tag="last"):            
        # load all stored state into net, optimizer, and scaler
        model_p = self._model_path(model_tag)
        map_loc = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(model_p, map_location=map_loc)
        net.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        best_val = ckpt["best_val"]
        if scaler is not None and "scaler_state" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state"])
        
        # load all stored loss training parameters
        param_p = self._param_path(tag) 
        ckpt_param = torch.load(param_p, map_location=map_loc)
        fold_train_loss = ckpt_param['fold_train_loss']
        fold_valid_loss = ckpt_param['fold_valid_loss']
        i_fold = ckpt_param['i_fold']
        i_epoch = ckpt_param['i_epoch']
                
        return best_val, fold_train_loss, fold_valid_loss, i_fold, i_epoch
        
