import argparse
import glob
from typing import Tuple

import numpy as np


def show_mAP_from_log(log_path: str) -> Tuple[float, str]:
    log_file = list(open(log_path))
    last_best_map_str = [ll for ll in log_file if "best_coco_segm_mAP_epoch" in ll][-1]
    last_best_map = float(last_best_map_str.split()[11])
    return last_best_map, last_best_map_str


def show_mean_mAP_from_log(EXP_ID: str, path_to_dir: str) -> None:
    exp_maps = []
    for fold in [0, 1, 2, 3, 4]:
        log_paths = glob.glob(f"{path_to_dir}/{EXP_ID}/fold{fold}/*/*.log")
        last_log_path = sorted(log_paths)[-1]
        last_best_map, last_best_map_str = show_mAP_from_log(last_log_path)
        print(f"exp{EXP_ID} fold{fold}")
        print(last_best_map_str)
        exp_maps.append(last_best_map)
    print(f"*** mean mAP is {np.mean(exp_maps):4f} at {EXP_ID} ***")
    print()


def main(args) -> None:
    EXP_IDs = args.exp_ids.split(" ")
    path_to_dir = args.path_to_dir
    for EXP_ID in EXP_IDs:
        show_mean_mAP_from_log(EXP_ID, path_to_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-ids", "-e", type=str, required=True, help="exp id list string"
    )
    parser.add_argument(
        "--path-to-dir", "-p", type=str, required=True, help="path to dir where exp log is saved"
    )
    args = parser.parse_args()
    main(args)
