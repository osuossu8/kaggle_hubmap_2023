import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import argparse
import os
import cv2
import glob
import tifffile as tiff
from PIL import Image
from tqdm import tqdm
from pathlib import Path, PosixPath
import os.path as osp

from sklearn.model_selection import KFold
from dataclasses import dataclass
from typing import List, Tuple, Union

from mmengine.fileio import dump, load
from mmdet.apis import init_detector, inference_detector


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


def binary_mask_to_coco_segmentation(binary_mask: np.ndarray) -> List[List[int]]:
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    coco_segmentation = []
    for contour in contours:
        segmentation = contour.flatten().tolist()
        coco_segmentation.append(segmentation)

    return coco_segmentation


def get_bounding_boxes(binary_mask: np.ndarray) -> List[List[int]]:
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bbox = [x, y, x + w, y + h]  # 左上隅と右下隅の座標
        bounding_boxes.append(bbox)

    return bounding_boxes


def add_pseudo_label_and_to_coco(
    data_root: PosixPath,
    df: pd.DataFrame,
    out_file: str,
    image_prefix: str,
    kfold: int,
    pseudo_threshold: float,
    exp_id: str,
) -> None:
    os.makedirs(image_prefix, exist_ok=True)
    annotations = []
    images = []
    obj_count = 0

    EXP_ID = exp_id # '015'
    work_dir_path = f'./work_dirs/exp{EXP_ID}/fold{kfold}'
    config_file = f'{work_dir_path}/exp{EXP_ID}.py'
    checkpoint_file = glob.glob(f'{work_dir_path}/best_coco_segm_mAP_epoch_*.pth')[-1]
    model = init_detector(config_file, checkpoint_file, device='cuda:0')      
        
    for idx, file_id in tqdm(enumerate(df['id'].values), total=len(df)):
        tiff_array = tiff.imread(data_root / f'train/{file_id}.tif')
        img_example = Image.fromarray(tiff_array).convert("RGB")
        img = np.array(img_example)
        filename = f'{file_id}.png'
        img_path = osp.join(image_prefix, filename)
        cv2.imwrite(img_path, img)

        height, width = 512, 512
        images.append(
            dict(id=idx, file_name=filename, height=height, width=width))
        
        res = inference_detector(model, img)
        pred = res.pred_instances.detach().cpu().numpy()
        pred_masks = pred.masks
        pred_scores = pred.scores.tolist()
        
        for segm_binary_mask, segm_score in zip(pred_masks, pred_scores):
            if segm_score < pseudo_threshold:
                continue
            segm_binary_mask = segm_binary_mask.astype(np.uint8)
            x_min, y_min, x_max, y_max = get_bounding_boxes(segm_binary_mask)[0]
            coco_segm = binary_mask_to_coco_segmentation(segm_binary_mask)

            data_anno = dict(
                image_id=idx,
                id=obj_count,
                category_id=0,
                bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                area=(x_max - x_min) * (y_max - y_min),
                segmentation=coco_segm,
                iscrowd=0)
            annotations.append(data_anno)
            obj_count += 1

        coco_format_json = dict(
            images=images,
            annotations=annotations,
            categories=[{
                'id': 0,
                'name': 'blood_vessel'
            }])
        dump(coco_format_json, out_file)


def main(args) -> None:
    exp_id = args.exp_id
    pseudo_threshold = float(args.pseudo_labeling_threshold) # 0.8
    data_root = Path('../input')

    with open(data_root / 'polygons.jsonl', 'r') as json_file:
        json_list = list(json_file)

    tile_meta_df = pd.read_csv(data_root / "tile_meta.csv")
    tile_meta_df = split_kfold(tile_meta_df, CFG)

    df = pd.DataFrame(json_list, columns=['annot_dict'])
    df['id'] = df['annot_dict'].map(lambda x: eval(x)['id'])
    df = pd.merge(df, tile_meta_df, on='id', how='inner')
    print(df.shape)

    for kfold in [0,1,2,3,4]:
        train_dataset_number = 2
        dataset_number = 1
        source_wsi_number = 1
        val_df = df.query(f"kfold == {kfold} & dataset == {dataset_number} & source_wsi == {source_wsi_number}").reset_index(drop=True)
        train_df = df[~df['id'].isin(val_df['id'])].query(f"dataset == {train_dataset_number}").query(f"kfold != {kfold}").reset_index(drop=True)

        print(train_df.shape)
        print(train_df['source_wsi'].value_counts())

        DATASET_NAME = f'hubmap-converted-to-coco-ds2-5fold-pseudo-labeled-{str(pseudo_threshold).replace(".", "-")}-by-exp{exp_id}'
        out_file = f'../input/{DATASET_NAME}/fold{kfold}/train/annotation_coco.json'
        image_prefix = f'../input/{DATASET_NAME}/fold{kfold}/train'

        add_pseudo_label_and_to_coco(
            data_root=data_root,    
            df=train_df,
            out_file=out_file,
            image_prefix=image_prefix,
            kfold=kfold,
            pseudo_threshold=pseudo_threshold,
            exp_id=exp_id,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-id", "-e", type=str, required=True, help="exp id")
    parser.add_argument("--pseudo-labeling-threshold", "-th", type=str, required=True, help="pseudo labeling threshold")
    args = parser.parse_args()
    main(args)
