

import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import os
import sys
import pickle
from tqdm import tqdm
import math
import logging
import pdb

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
from trAISformer_testing import test_, replace_xy_with_grid_center

import models, trainers, datasets, utils
from config_trAISformer import Config
cf = Config()
TB_LOG = cf.tb_log
tb = None
if TB_LOG:
    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter()

# make deterministic
utils.set_seed(42)
torch.pi = torch.acos(torch.zeros(1)).item()*2


if __name__ == "__main__":

    device = cf.device

    
    ## Logging
    #===============================
    if not os.path.isdir(cf.savedir):
        os.makedirs(cf.savedir)
        print('======= Create directory to store trained models: '+cf.savedir)
    else:
        print('======= Directory to store trained models: '+cf.savedir)
    utils.new_log(cf.savedir,"log")
    pickle.dump(cf, open(cf.config_pkl, 'wb'))

    ## Data
    #===============================
    moving_threshold = cf.moving_threshold
    l_pkl_filenames = [cf.trainset_name, cf.validset_name, cf.testset_name]
    Data, aisdatasets, aisdls = {}, {}, {}
    for phase, filename in zip(("train","valid","test"),l_pkl_filenames):
        datapath = os.path.join(cf.datadir,filename)
        print(f"Loading {datapath}...")
        with open(datapath,"rb") as f:
            l_pred_errors = pickle.load(f)
        for V in l_pred_errors:
            if np.isnan(V['traj']).any():
                V['traj'] = utils.ffill_roll(V['traj'].copy())
            try:
                moving_idx = np.where(V["traj"][:,2]>=moving_threshold)[0][0]
            except:
                moving_idx = len(V["traj"]) - 1 # This track will be removed
            if moving_idx > 0:
                a = 10
            V["traj"] = V["traj"][moving_idx:,:]
        Data[phase] = [x for x in l_pred_errors if not np.isnan(x["traj"]).any() and len(x["traj"]) > cf.min_seqlen]
        print(len(l_pred_errors), len(Data[phase]))
        print(f"Length: {len(Data[phase])}")
        print("Creating pytorch dataset...")
        # Latter in this scipt, we will use inputs = x[:-1], targets = x[1:], hence
        # max_seqlen = cf.max_seqlen + 1.
        if cf.mode in ("pos_grad","grad"):
            aisdatasets[phase] = datasets.AISDataset_grad(Data[phase],
                                                     max_seqlen=cf.max_seqlen+1,
                                                     device=cf.device)    
        else:
            aisdatasets[phase] = datasets.AISDataset(Data[phase],
                                                     max_seqlen=cf.max_seqlen+1,
                                                     device=cf.device)
        if phase == "test":
            shuffle = False
        else:
            shuffle = True
        aisdls[phase] = DataLoader(aisdatasets[phase], 
                                   batch_size=cf.batch_size,
                                   shuffle=shuffle)
    cf.final_tokens=2*len(aisdatasets["train"])*cf.max_seqlen

    ## Model
    #===============================
    model = models.TrAISformer(cf, partition_model=None)

    ## Trainer
    #===============================
    trainer = trainers.Trainer(
        model, aisdatasets["train"], aisdatasets["valid"], cf, savedir=cf.savedir, device=cf.device, tb=tb , aisdls=aisdls)


    pickle.dump(aisdls, open(cf.aisdls_pkl, 'wb'))
    trainer.save_epoch_loss_plot()
    ## Training
    #===============================
    if cf.retrain:
        trainer.train()
        trainer.save_epoch_loss_plot()

    ## Evaluation
    #===============================
    # Load the best model
    model.load_state_dict(torch.load(cf.ckpt_path))


    torch.save(model, open(cf.model_pkl, 'wb'))



    test_(model, cf, aisdls, replace_xy_with_grid_center)


    # Yeah, done!!!
