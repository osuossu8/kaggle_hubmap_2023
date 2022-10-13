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
    EXP_ID = 'TEXT_BASE'
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
    batch_size = 8
    num_workers = 0
    scheduler = 'cosine'
    lr = 5e-6
    min_lr=1e-6
    weigth_decay = 0.01
    scheduler = 'cosine'
    num_warmup_steps = 0
    num_cycles=0.5
    n_accumulate=1
    print_freq = 100
    eval_freq = 200

    # text competition
    model_name = 'microsoft/deberta-v3-large'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_len = 1024
    encode_type = 'head_and_tail' # or ''


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


class ForTextDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df = df
        self.max_len = CFG.max_len
        self.text = df['full_text'].values
        self.tokenizer = CFG.tokenizer
        self.targets = df[CFG.target_cols].values

    def __len__(self):
        return len(self.df)

    def cut_head_and_tail(self, text):
        input_ids = self.tokenizer.encode(text)
        n_token = len(input_ids)

        if n_token == self.max_len:
            input_ids = input_ids
            attention_mask = [1 for _ in range(self.max_len)]
            token_type_ids = [1 for _ in range(self.max_len)]
        elif n_token < self.max_len:
            pad = [1 for _ in range(self.max_len-n_token)]
            input_ids = input_ids + pad
            attention_mask = [1 if n_token > i else 0 for i in range(self.max_len)]
            token_type_ids = [1 if n_token > i else 0 for i in range(self.max_len)]
        else:
            harf_len = (self.max_len-2)//2
            _input_ids = input_ids[1:-1]
            input_ids = [0]+ _input_ids[:harf_len] + _input_ids[-harf_len:] + [2]
            attention_mask = [1 for _ in range(self.max_len)]
            token_type_ids = [1 for _ in range(self.max_len)]

        d = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
        return d


    def __getitem__(self, index):
        text = self.text[index]
        if CFG.encode_type == 'head_and_tail':
            inputs = self.cut_head_and_tail(text)
        else:
            inputs = self.tokenizer.encode_plus(
                text,
                truncation=True,
                add_special_tokens=True,
                max_length = self.max_len
            )
        return {
            'input_ids':inputs['input_ids'],
            'attention_mask':inputs['attention_mask'],
            'target':self.targets[index]
            }


class Collate:
    def __init__(self, tokenizer, isTrain=True):
        self.tokenizer = tokenizer
        self.isTrain = isTrain

    def __call__(self, batch):
        output = dict()
        output["input_ids"] = [sample["input_ids"] for sample in batch]
        output["attention_mask"] = [sample["attention_mask"] for sample in batch]
        if self.isTrain:
            output["target"] = [sample["target"] for sample in batch]

        # calculate max token length of this batch
        batch_max = max([len(ids) for ids in output["input_ids"]])

        # add padding
        if self.tokenizer.padding_side == "right":
            output["input_ids"] = [s + (batch_max - len(s)) * [self.tokenizer.pad_token_id] for s in output["input_ids"]]
            output["attention_mask"] = [s + (batch_max - len(s)) * [0] for s in output["attention_mask"]]
        else:
            output["input_ids"] = [(batch_max - len(s)) * [self.tokenizer.pad_token_id] + s for s in output["input_ids"]]
            output["attention_mask"] = [(batch_max - len(s)) * [0] + s for s in output["attention_mask"]]

        # convert to tensors
        output["input_ids"] = torch.tensor(output["input_ids"], dtype=torch.long)
        output["attention_mask"] = torch.tensor(output["attention_mask"], dtype=torch.long)
        if self.isTrain:
            output["target"] = torch.tensor(output["target"], dtype=torch.float)

        return output

collate_fn = Collate(CFG.tokenizer, isTrain=True)


def freeze(module):
    """
    Freezes module's parameters.
    """

    for parameter in module.parameters():
        parameter.requires_grad = False


class ForTextModel(nn.Module):
    def __init__(self, model_name):
        super(FeedBackModel, self).__init__()

        self.cfg = CFG
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.hidden_dropout_prob = 0
        self.config.attention_probs_dropout_prob = 0

        self.model = AutoModel.from_pretrained(model_name, config=self.config)

        self.output = nn.Sequential(
            nn.LayerNorm(self.config.hidden_size),
            nn.Linear(self.config.hidden_size, self.cfg.target_size)
        )


        # Freeze
        if self.cfg.freezing:
            freeze(self.model.embeddings)
            # freeze(self.model.encoder.layer[:2])

        # Gradient Checkpointing
        #if self.cfg.gradient_checkpoint:
        #    self.model.gradient_checkpointing_enable() 

        #if self.cfg.reinit_layers > 0:
        #    layers = self.model.encoder.layer[-self.cfg.reinit_layers:]
        #    for layer in layers:
        #        for module in layer.modules():
        #            self._init_weights(module)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, ids, mask, token_type_ids=None, targets=None):
        if token_type_ids:
            transformer_out = self.model(ids, mask, token_type_ids)
        else:
            transformer_out = self.model(ids, mask)

        # simple CLS
        sequence_output = transformer_out[0][:, 0, :]

        logits = self.output(sequence_output)

        return logits


def train_one_epoch(model, optimizer, scheduler, dataloader, valid_loader, device, epoch, best_score, valid_labels):
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

        if (step > 0) & (step % CFG.eval_freq == 0) :

            valid_epoch_loss, pred = valid_one_epoch(model, valid_loader, device, epoch)

            score = get_score(pred, valid_labels)

            LOGGER.info(f'Epoch {epoch+1} Step {step} - avg_train_loss: {epoch_loss:.4f}  avg_val_loss: {valid_epoch_loss:.4f}')
            LOGGER.info(f'Epoch {epoch+1} Step {step} - Score: {score:.4f}')

            if score < best_score:
                best_score = score
                LOGGER.info(f'Epoch {epoch+1} Step {step} - Save Best Score: {best_score:.4f} Model')
                torch.save({'model': model.state_dict(),
                            'predictions': pred},
                            OUTPUT_DIR+f"{CFG.model_name.replace('/', '-')}_fold{fold}_best.pth")

            # model.train()

    gc.collect()

    return epoch_loss, valid_epoch_loss, pred, best_score


@torch.no_grad()
def valid_one_epoch(model, dataloader, device, epoch):
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

    trainDataset = ForTextDataset(train_data, CFG.tokenizer, CFG.max_len)
    validDataset = ForTextDataset(valid_data, CFG.tokenizer, CFG.max_len)

    train_loader = DataLoader(trainDataset,
                              batch_size = CFG.batch_size,
                              shuffle=True,
                              collate_fn = collate_fn,
                              num_workers = CFG.num_workers,
                              pin_memory = True,
                              drop_last=True)

    valid_loader = DataLoader(validDataset,
                              batch_size = CFG.batch_size * 2,
                              shuffle=False,
                              collate_fn = collate_fn,
                              num_workers = CFG.num_workers,
                              pin_memory = True,
                              drop_last=False)

    model = ForTextModel(CFG.model_name)
    torch.save(model.config, OUTPUT_DIR+'config.pth')
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weigth_decay)
    num_train_steps = int(len(train_data) / CFG.batch_size * CFG.epochs)
    scheduler = get_scheduler(CFG, optimizer, num_train_steps)

    # loop
    best_score = 100

    for epoch in range(CFG.num_epochs):
        if epoch == (CFG.num_epochs - 1):
            break

        start_time = time.time()

        train_epoch_loss, valid_epoch_loss, pred, best_score = train_one_epoch(model, optimizer, scheduler, train_loader, valid_loader, device, epoch, best_score, valid_labels)

        score = get_score(pred, valid_labels)

        elapsed = time.time() - start_time

        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {train_epoch_loss:.4f}  avg_val_loss: {valid_epoch_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch+1} - Score: {score:.4f}')

        if score < best_score:
            best_score = score
            LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save({'model': model.state_dict(),
                        'predictions': pred},
                        OUTPUT_DIR+f"{CFG.model_name.replace('/', '-')}_fold{fold}_best.pth")

    predictions = torch.load(OUTPUT_DIR+f"{CFG.model_name.replace('/', '-')}_fold{fold}_best.pth",
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

