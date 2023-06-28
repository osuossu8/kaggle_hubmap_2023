import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from dataclasses import dataclass
from typing import List, Union


@dataclass
class BaseConfig:
    seed: int
    num_fold: int
    group_col: str
    target_col: Union[str, List[str]]
        
class CFG(BaseConfig):
    seed = 42
    num_fold = 5
    target_col = 'id'
    

def split_kfold(df: pd.DataFrame, cfg: BaseConfig) -> pd.DataFrame:
    df["kfold"] = -1
    kf = KFold(n_splits=cfg.num_fold, shuffle=True, random_state=cfg.seed)
    for n, (trn_index, val_index) in enumerate(kf.split(X=df, y=df[cfg.target_col])):
        df.loc[val_index, "kfold"] = int(n)
    df["kfold"] = df["kfold"].astype(int)
    return df


def main() -> None:
    data_root = Path('../input')
    tile_meta_df = pd.read_csv(data_root / "tile_meta.csv")
    tile_meta_df = split_kfold(tile_meta_df, CFG)
    tile_meta_df.to_csv(data_root / "tile_meta_with_5fold.csv", index=False)


if __name__ == "__main__":
    main()