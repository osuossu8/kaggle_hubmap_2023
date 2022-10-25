import os
import gc
import re
import sys
sys.path.append("/root/workspace/CompetitionBase")
import json
import time
import math
import string
import pickle
import random
import joblib
import itertools
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()

from contextlib import contextmanager
from pathlib import Path
from typing import List
from typing import Optional
from sklearn import metrics
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, GroupKFold, KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler

import transformers
import tokenizers
print(f"tokenizers.__version__: {tokenizers.__version__}")
print(f"transformers.__version__: {transformers.__version__}")
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup


from src.scheduler import get_scheduler
from src.split_data import split_kfold, split_stratified, split_group, split_stratified_group, split_multilabel_stratified
from src.machine_learning_util import set_seed, set_device, init_logger, AverageMeter, to_pickle, unpickle, asMinutes, timeSince


class CFG:
    # common
    EXP_ID = 'BASE'
    apex = True
    debug = False
    seed = 71
    num_fold = 5 
    trn_fold = [i for i in range(num_fold)]

    # competition specific
    id_col = ''
    group_col = ''
    target_cols = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
    target_size = len(target_cols)

    # training    
    num_epochs = 5
    batch_size = 16
    num_workers = 0
    scheduler = 'cosine'
    lr = 1e-3
    min_lr=1e-6
    weigth_decay = 0.01
    scheduler = 'cosine'
    num_warmup_steps = 0
    num_cycles=0.5
    n_accumulate=1
    print_freq = 100
    eval_freq = 200


set_seed(CFG.seed)
device = set_device()
LOGGER = init_logger(log_file='log/' + f"{CFG.EXP_ID}.log")

OUTPUT_DIR = f'output/{CFG.EXP_ID}/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def criterion(outputs, targets):
    #loss_fct = nn.MSELoss()
    loss_fct = nn.BCEWithLogitsLoss()
    loss = loss_fct(outputs, targets)
    return loss


def get_score(outputs, targets):
    mcrmse = []
    for i in range(CFG.target_size):
        mcrmse.append(
            metrics.mean_squared_error(
                targets[:, i],
                outputs[:, i],
                squared=False,
            ),
        )
    mcrmse = np.mean(mcrmse)
    return mcrmse


class ForTableDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df
        self.features = df[CFG.numerical_cols].values
        self.targets = df[CFG.target_cols].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        features = self.features[index]
        targets = self.targets[index]
            
        return {
            'features' : torch.FloatTensor(features),
            'targets' : torch.FloatTensor(targets),
        }


class ForTableModel(nn.Module):
    def __init__(self, CFG):
        super(ForTableModel, self).__init__()

        self.regressor = nn.Sequential(
            nn.Linear(len(CFG.numerical_cols), 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, CFG.target_size)
        )

    def forward(self, features):
        # bs, len_features
        output = self.regressor(features)
        return output


def train_one_epoch(model, dataloader, device, epoch, criterion, optimizer, scheduler):
    model.train()
    scaler = GradScaler(enabled=CFG.apex)

    dataset_size = 0
    running_loss = 0
    start = end = time.time()

    for step, data in enumerate(dataloader):
        features = data['features'].to(device, dtype=torch.float)
        targets = data['targets'].to(device, dtype=torch.float)

        batch_size = features.size(0)

        with autocast(enabled=CFG.apex):
            outputs = model(features)
            loss = criterion(outputs, targets)

        loss = loss / CFG.n_accumulate
        scaler.scale(loss).backward()
        if (step +1) % CFG.n_accumulate == 0:
            # torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        epoch_loss = running_loss / dataset_size
        end = time.time()

        if step % CFG.print_freq == 0 or step == (len(dataloader) - 1):
            print(
                "Epoch: [{0}][{1}/{2}] "
                "Loss: [{3}] "
                "Elapsed {remain:s} ".format(
                    epoch + 1,
                    step,
                    len(dataloader),
                    epoch_loss,
                    remain=timeSince(start, float(step + 1) / len(dataloader)),
                )
            )

    return epoch_loss


@torch.no_grad()
def valid_one_epoch(model, dataloader, device, criterion):
    model.eval()

    dataset_size = 0
    running_loss = 0

    start = end = time.time()
    pred = []

    for step, data in enumerate(dataloader):
        features = data['features'].to(device, dtype=torch.float)
        targets = data['targets'].to(device, dtype=torch.float)

        batch_size = features.size(0)
        outputs = model(features)
        loss = criterion(outputs, targets)
        pred.append(outputs.to('cpu').numpy())

        running_loss += (loss.item()* batch_size)
        dataset_size += batch_size
        epoch_loss = running_loss / dataset_size
        end = time.time()

        if step % CFG.print_freq == 0 or step == (len(dataloader) - 1):
            print(
                "EVAL: [{0}/{1}] "
                "Loss: [{2}] "
                "Elapsed {remain:s} ".format(
                    step, len(dataloader), epoch_loss, remain=timeSince(start, float(step + 1) / len(dataloader))
                )
            )

    pred = np.concatenate(pred)
    return epoch_loss, pred


def train_loop(fold):
    LOGGER.info(f'-------------fold:{fold} training-------------')

    train_data = train[train.kfold != fold].reset_index(drop=True)
    valid_data = train[train.kfold == fold].reset_index(drop=True)
    valid_labels = valid_data[CFG.target_cols].values

    trainDataset = ForTableDataset(train_data)
    validDataset = ForTableDataset(valid_data)

    train_loader = DataLoader(trainDataset,
                              batch_size = CFG.batch_size,
                              shuffle=True,
                              num_workers = CFG.num_workers,
                              pin_memory = True,
                              drop_last=True)

    valid_loader = DataLoader(validDataset,
                              batch_size = CFG.batch_size * 2,
                              shuffle=False,
                              num_workers = CFG.num_workers,
                              pin_memory = True,
                              drop_last=False)

    model = ForTableModel(CFG)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weigth_decay)
    num_train_steps = int(len(train_data) / CFG.batch_size * CFG.epochs)
    scheduler = get_scheduler(CFG, optimizer, num_train_steps)

    # loop
    best_score = 100

    for epoch in range(CFG.num_epochs):

        start_time = time.time()

        train_epoch_loss = train_one_epoch(model, train_loader, device, epoch, criterion, optimizer, scheduler)
        valid_epoch_loss, pred = valid_one_epoch(model, valid_loader, device, criterion)
    
        score = get_score(pred, valid_labels)

        elapsed = time.time() - start_time

        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {train_epoch_loss:.4f}  avg_val_loss: {valid_epoch_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch+1} - Score: {score:.4f}')

        if score < best_score:
            best_score = score
            LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save({'model': model.state_dict(),
                        'predictions': pred},
                        OUTPUT_DIR+f"{CFG.EXP_ID}_fold{fold}_best.pth")

    predictions = torch.load(OUTPUT_DIR+f"{CFG.EXP_ID}_fold{fold}_best.pth",
                             map_location=torch.device('cpu'))['predictions']
    valid_data['pred_0'] = predictions[:, 0]
    valid_data['pred_1'] = predictions[:, 1]
    valid_data['pred_2'] = predictions[:, 2]
    valid_data['pred_3'] = predictions[:, 3]
    valid_data['pred_4'] = predictions[:, 4]
    valid_data['pred_5'] = predictions[:, 5]

    torch.cuda.empty_cache()
    gc.collect()

    return valid_data


if __name__ == '__main__':

    def get_result(oof_df):
        labels = oof_df[CFG.target_cols].values
        preds = oof_df[['pred_0', 'pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_5']].values
        score = get_score(preds, labels)
        LOGGER.info(f'Score: {score:<.4f}')

    if CFG.train:
        oof_df = pd.DataFrame()
        for fold in range(CFG.num_fold):
            if fold in CFG.trn_fold:
                _oof_df = train_loop(fold)
                oof_df = pd.concat([oof_df, _oof_df])
                LOGGER.info(f"========== fold: {fold} result ==========")
                get_result(_oof_df)
        oof_df = oof_df.reset_index(drop=True)
        LOGGER.info(f"========== CV ==========")
        get_result(oof_df)
        oof_df.to_csv(OUTPUT_DIR+f'oof_df.csv', index=False)

