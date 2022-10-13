from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, StratifiedGroupKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


def split_kfold(df, CFG):
    kf = KFold(n_splits=CFG.num_fold, shuffle=True, random_state=CFG.SEED)
    for n, (trn_index, val_index) in enumerate(kf.split(X=df, y=df[CFG.target_cols])):
        df.loc[val_index, "kfold"] = int(n)
    df["kfold"] = df["kfold"].astype(int)
    return df


def split_stratified(df, CFG):
    df["kfold"] = -1
    kf = StratifiedKFold(n_splits=CFG.num_fold, shuffle=True, random_state=CFG.SEED)
    for n, (trn_index, val_index) in enumerate(kf.split(X=df, y=df[CFG.target_cols])):
        df.loc[val_index, "kfold"] = int(n)
    df["kfold"] = df["kfold"].astype(int)
    return df


def split_group(df, CFG):
    df["kfold"] = -1
    kf = GroupKFold(n_splits=CFG.num_fold)
    for n, (trn_index, val_index) in enumerate(kf.split(X=df, groups=df[CFG.group_col])):
        df.loc[val_index, "kfold"] = int(n)
    df["kfold"] = df["kfold"].astype(int)
    return df


def split_stratified_group(df, CFG):
    df["kfold"] = -1
    kf = StratifiedGroupKFold(n_splits=CFG.num_fold, shuffle=True, random_state=CFG.SEED)
    for n, (trn_index, val_index) in enumerate(kf.split(X=df, y=df[CFG.target_cols], groups=df[CFG.group_col])):
        df.loc[val_index, "kfold"] = int(n)
    df["kfold"] = df["kfold"].astype(int)
    return df


def split_multilabel_stratified(df, CFG):
    df["kfold"] = -1
    kf = MultilabelStratifiedKFold(n_splits=CFG.num_fold, shuffle=True, random_state=CFG.SEED)
    for n, (trn_index, val_index) in enumerate(kf.split(X=df, y=df[CFG.target_cols])):
        df.loc[val_index, "kfold"] = int(n)
    df["kfold"] = df["kfold"].astype(int)
    return df
