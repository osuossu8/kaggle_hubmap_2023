import pandas as pd
from typing import Dict
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, StratifiedGroupKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


class DataSplitter(object):
    @staticmethod
    def split_kfold(df : pd.DataFrame, cfg: Dict):
        df["kfold"] = -1
        kf = KFold(n_splits=cfg['num_fold'], shuffle=True, random_state=cfg['seed'])
        for n, (trn_index, val_index) in enumerate(kf.split(X=df, y=df[cfg['target_col']])):
            df.loc[val_index, "kfold"] = int(n)
        df["kfold"] = df["kfold"].astype(int)
        return df
    
    @staticmethod
    def split_stratified(df : pd.DataFrame, cfg: Dict):
        df["kfold"] = -1
        kf = StratifiedKFold(n_splits=cfg['num_fold'], shuffle=True, random_state=cfg['seed'])
        for n, (trn_index, val_index) in enumerate(kf.split(X=df, y=df[cfg['target_col']])):
            df.loc[val_index, "kfold"] = int(n)
        df["kfold"] = df["kfold"].astype(int)
        return df
    
    @staticmethod
    def split_group(df : pd.DataFrame, cfg: Dict):
        df["kfold"] = -1
        kf = GroupKFold(n_splits=cfg['num_fold'])
        for n, (trn_index, val_index) in enumerate(kf.split(X=df, groups=df[cfg['group_col']])):
            df.loc[val_index, "kfold"] = int(n)
        df["kfold"] = df["kfold"].astype(int)
        return df
    
    @staticmethod
    def split_stratified_group(df : pd.DataFrame, cfg: Dict):
        df["kfold"] = -1
        kf = StratifiedGroupKFold(n_splits=cfg['num_fold'], shuffle=True, random_state=cfg['seed'])
        for n, (trn_index, val_index) in enumerate(kf.split(X=df, y=df[cfg['target_col']], groups=df[cfg['group_col']])):
            df.loc[val_index, "kfold"] = int(n)
        df["kfold"] = df["kfold"].astype(int)
        return df
    
    @staticmethod
    def split_multilabel_stratified(df : pd.DataFrame, cfg: Dict):
        df["kfold"] = -1
        kf = MultilabelStratifiedKFold(n_splits=cfg['num_fold'], shuffle=True, random_state=cfg['seed'])
        for n, (trn_index, val_index) in enumerate(kf.split(X=df, y=df[cfg['target_multi_cols']])):
            df.loc[val_index, "kfold"] = int(n)
        df["kfold"] = df["kfold"].astype(int)
        return df
