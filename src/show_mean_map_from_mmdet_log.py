import argparse
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
    for fold in [0, 1, 2, 3, 4]:
        log_paths = glob.glob(f"work_dirs/exp{EXP_ID}/fold{fold}/*/*.log")
        for log_path in log_paths:
            last_best_map, last_best_map_str = show_mAP_from_log(log_path)
            print(f"exp{EXP_ID} fold{fold}")
            print(last_best_map_str)
            exp_maps.append(last_best_map)
    print(f"*** mean mAP is {np.mean(exp_maps):4f} at exp{EXP_ID} ***")
    print()


def main(args) -> None:
    EXP_IDs = args.exp_ids.split(" ")
    for EXP_ID in EXP_IDs:
        show_mean_mAP_from_log(EXP_ID)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-ids", "-e", type=str, required=True, help="exp id list string"
    )
    args = parser.parse_args()
    main(args)
