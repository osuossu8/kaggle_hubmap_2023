import glob
import json
import os
import os.path as osp
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import tifffile as tiff
from ensemble_boxes import non_maximum_weighted
from mmdet.apis import inference_detector, init_detector
from mmengine.fileio import dump, load
from PIL import Image
from sklearn.model_selection import KFold
from tqdm import tqdm


@dataclass
class BaseConfig:
    seed: int
    num_fold: int
    group_col: str
    target_col: Union[str, List[str]]


class CFG(BaseConfig):
    seed = 42
    num_fold = 5
    target_col = "id"


def split_kfold(df: pd.DataFrame, cfg: BaseConfig) -> pd.DataFrame:
    df["kfold"] = -1
    kf = KFold(n_splits=cfg.num_fold, shuffle=True, random_state=cfg.seed)
    for n, (trn_index, val_index) in enumerate(kf.split(X=df, y=df[cfg.target_col])):
        df.loc[val_index, "kfold"] = int(n)
    df["kfold"] = df["kfold"].astype(int)
    return df


def remove_score_zero_result(
    pred_classes: List[int],
    pred_scores: List[float],
    pred_bboxes: List[List[float]],
    pred_masks: np.ndarray,  # (num_mask, 512, 512)
) -> Tuple[List[int], List[float], List[List[float]], np.ndarray]:
    classes, scores, bboxes, masks = [], [], [], []
    for c, s, b, m in zip(pred_classes, pred_scores, pred_bboxes, pred_masks):
        if s > 0:
            classes.append(c)
            scores.append(s)
            bboxes.append(b)
            masks.append(m)
    return classes, scores, bboxes, masks


def nms_predictions(classes, scores, bboxes, masks, iou_th=0.5, shape=(512, 512)):
    he, wd = shape[0], shape[1]
    boxes_list = [[[x[0] / wd, x[1] / he, x[2] / wd, x[3] / he] for x in bboxes]]
    scores_list = [[x for x in scores]]
    classes_list = [[x for x in classes]]
    nms_bboxes, nms_scores, nms_classes = non_maximum_weighted(
        boxes_list,
        scores_list,
        classes_list,
        weights=None,
        iou_thr=IOU_TH,
        skip_box_thr=0.0001,
    )
    nms_masks = []
    for s in nms_scores:
        nms_masks.append(masks[scores.index(s)])
    nms_scores, nms_classes, nms_masks = zip(
        *sorted(zip(nms_scores, nms_classes, nms_masks), reverse=True)
    )
    return nms_classes, nms_scores, nms_masks


def binary_mask_to_coco_segmentation(binary_mask):
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    coco_segmentation = []
    for contour in contours:
        segmentation = contour.flatten().tolist()
        coco_segmentation.append(segmentation)

    return coco_segmentation


def get_bounding_boxes(binary_mask):
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bbox = [x, y, x + w, y + h]  # 左上隅と右下隅の座標
        bounding_boxes.append(bbox)

    return bounding_boxes


pseudo_threshold = 0.8
data_root = Path("../input")
EXP_ID = "015"

models = []
for fold in [0, 1, 2, 3, 4]:
    work_dir_path = f"./work_dirs/exp{EXP_ID}/fold{fold}"
    config_file = f"{work_dir_path}/exp{EXP_ID}.py"
    checkpoint_file = glob.glob(f"{work_dir_path}/best_coco_segm_mAP_epoch_*.pth")[-1]
    model = init_detector(config_file, checkpoint_file, device="cuda:0")
    models.append(model)


with open(data_root / "polygons.jsonl", "r") as json_file:
    json_list = list(json_file)

tile_meta_df = pd.read_csv(data_root / "tile_meta.csv")
tile_meta_df = split_kfold(tile_meta_df, CFG)

df = pd.DataFrame(json_list, columns=["annot_dict"])
df["id"] = df["annot_dict"].map(lambda x: eval(x)["id"])
df = pd.merge(df, tile_meta_df, on="id", how="inner")
print(df.shape)

kfold = 0
train_dataset_number = 2
dataset_number = 1
source_wsi_number = 1
val_df = df.query(
    f"kfold == {kfold} & dataset == {dataset_number} & source_wsi == {source_wsi_number}"
).reset_index(drop=True)
train_df = (
    df[~df["id"].isin(val_df["id"])]
    .query(f"dataset == {train_dataset_number}")
    .reset_index(drop=True)
)

print(train_df.shape)
print(train_df["source_wsi"].value_counts())

DATASET_NAME = f'hubmap-converted-to-coco-ds2-pseudo-labeled-{str(pseudo_threshold).replace(".", "-")}'
out_file = f"../input/{DATASET_NAME}/fold{kfold}/train/annotation_coco.json"
image_prefix = f"../input/{DATASET_NAME}/fold{kfold}/train"


os.makedirs(image_prefix, exist_ok=True)
annotations = []
images = []
obj_count = 0


for idx, (file_id, annot_dict) in tqdm(
    enumerate(
        zip(
            train_df["id"].values,
            train_df["annot_dict"].values,
        )
    ),
    total=len(train_df),
):

    # tldc = eval(annot_dict)
    tiff_array = tiff.imread(data_root / f"train/{file_id}.tif")
    img_example = Image.fromarray(tiff_array).convert("RGB")
    img = np.array(img_example)
    filename = f"{file_id}.png"
    img_path = osp.join(image_prefix, filename)
    cv2.imwrite(img_path, img)

    height, width = 512, 512
    images.append(dict(id=idx, file_name=filename, height=height, width=width))

    ens_classes = []
    ens_scores = []
    ens_bboxes = []
    ens_masks = []
    for model_ in models:
        res = inference_detector(model_, img)
        pred = res.pred_instances.detach().cpu().numpy()
        ens_classes.extend(pred.labels.tolist())
        ens_scores.extend(pred.scores.tolist())
        ens_bboxes.extend(pred.bboxes.tolist())
        ens_masks.append(pred.masks)
    ens_masks = np.concatenate(ens_masks, 0)
    # print(len(ens_classes), len(ens_scores), len(ens_masks))

    ens_classes, ens_scores, ens_bboxes, ens_masks = remove_score_zero_result(
        ens_classes, ens_scores, ens_bboxes, ens_masks
    )

    IOU_TH = 0.6
    ens_classes, ens_scores, ens_masks = nms_predictions(
        ens_classes, ens_scores, ens_bboxes, ens_masks, iou_th=IOU_TH
    )
    # print(len(ens_classes), len(ens_scores), len(ens_masks))

    for segm_binary_mask, segm_score in zip(ens_masks, ens_scores):
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
            iscrowd=0,
        )
        annotations.append(data_anno)
        obj_count += 1

    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=[{"id": 0, "name": "blood_vessel"}],
    )
    dump(coco_format_json, out_file)
