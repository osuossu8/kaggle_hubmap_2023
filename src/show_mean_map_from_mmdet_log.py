import glob
from typing import Tuple

import numpy as np


def show_mAP_from_log(log_path: str) -> Tuple[float, str]:
    log_file = list(open(log_path))
    last_best_map_str = [ll for ll in log_file if "best_coco_segm_mAP_epoch" in ll][-1]
    last_best_map = float(last_best_map_str.split()[11])
    return last_best_map, last_best_map_str


def show_mean_mAP_from_log(EXP_ID: str) -> None:
    exp_maps = []
    exp_log_paths = glob.glob(f"work_dirs/exp{EXP_ID}/fold*/*/*.log")
    for log_path in exp_log_paths:
        last_best_map, last_best_map_str = show_mAP_from_log(log_path)
        print(EXP_ID, log_path.split("/")[2])
        print(last_best_map_str)
        exp_maps.append(last_best_map)
    print(f"*** mean mAP is {np.mean(exp_maps):4f} at exp{EXP_ID} ***")
    print()


EXP_IDs = ["015", "021", "024", "027"]

for EXP_ID in EXP_IDs:
    show_mean_mAP_from_log(EXP_ID)
