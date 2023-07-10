import argparse
import glob
import os
import os.path as osp
from pathlib import Path, PosixPath
from typing import List

import cv2
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import tifffile as tiff
from mmdet.apis import inference_detector, init_detector
from mmengine.fileio import dump
from PIL import Image
from tqdm import tqdm


def binary_mask_to_coco_segmentation(binary_mask: np.ndarray) -> List[List[int]]:
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    coco_segmentation = []
    for contour in contours:
        segmentation = contour.flatten().tolist()
        coco_segmentation.append(segmentation)

    return coco_segmentation


def get_bounding_boxes(binary_mask: np.ndarray) -> List[List[int]]:
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bbox = [x, y, x + w, y + h]  # 左上隅と右下隅の座標
        bounding_boxes.append(bbox)

    return bounding_boxes


def generate_chunks(data_list, num_chunk):
    for i in range(0, len(data_list), num_chunk):
        yield data_list[i : i + num_chunk]


def add_pseudo_label_and_to_coco_batch(
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
    img_count = 0
    annotation_conut = 0
    height, width = 512, 512

    EXP_ID = exp_id  # '015'
    work_dir_path = f"./work_dirs/exp{EXP_ID}/fold{kfold}"
    config_file = f"{work_dir_path}/exp{EXP_ID}.py"
    checkpoint_file = glob.glob(f"{work_dir_path}/best_coco_segm_mAP_epoch_*.pth")[-1]
    model = init_detector(config_file, checkpoint_file, device="cuda:0")

    batch_size = 32
    batch_generator = list(generate_chunks(df, batch_size))
    for batch in tqdm(batch_generator, total=len(batch_generator)):
        image_np_array_list = []
        image_count_list = []
        for file_id in batch["id"].values:
            image_np_array = np.array(
                Image.fromarray(
                    tiff.imread(data_root / f"train/{file_id}.tif")
                ).convert("RGB")
            )
            filename = f"{file_id}.png"
            img_path = osp.join(image_prefix, filename)
            cv2.imwrite(img_path, image_np_array)
            images.append(
                dict(id=img_count, file_name=filename, height=height, width=width)
            )
            image_count_list.append(img_count)
            img_count += 1
            image_np_array_list.append(image_np_array)
        res = inference_detector(model, image_np_array_list)
        preds = [r.pred_instances.detach().cpu().numpy() for r in res]
        for image_count, pred in zip(image_count_list, preds):
            pred_masks = pred.masks
            pred_scores = pred.scores.tolist()
            pred_labels = pred.labels.tolist()

            for segm_binary_mask, segm_score, segm_label in zip(pred_masks, pred_scores, pred_labels):
                if segm_score < pseudo_threshold:
                    continue
                segm_binary_mask = segm_binary_mask.astype(np.uint8)
                x_min, y_min, x_max, y_max = get_bounding_boxes(segm_binary_mask)[0]
                coco_segm = binary_mask_to_coco_segmentation(segm_binary_mask)

                data_anno = dict(
                    image_id=image_count,
                    id=annotation_conut,
                    category_id=segm_label,
                    bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                    area=(x_max - x_min) * (y_max - y_min),
                    segmentation=coco_segm,
                    iscrowd=0,
                )
                annotations.append(data_anno)
                annotation_conut += 1

        coco_format_json = dict(
            images=images,
            annotations=annotations,
            categories=[
                {
                    'id': 0,
                    'name': 'blood_vessel'
                },
                {
                    'id': 1,
                    'name': 'glomerulus'
                },
                {
                    'id': 2,
                    'name': 'unsure'
                }
            ],
        )
        dump(coco_format_json, out_file)


def main(args) -> None:
    exp_id = args.exp_id
    pseudo_threshold = float(args.pseudo_labeling_threshold)  # 0.8
    data_root = Path("../input")

    train_dataset_number = 3
    df = pd.read_csv(data_root / "tile_meta_with_5fold.csv")
    df_ds3 = df.query(f"dataset == {train_dataset_number}").reset_index(drop=True)
    print(df_ds3.shape)
    print(df_ds3["kfold"].value_counts())

    for kfold in [0, 1, 2, 3, 4]:
        train_df = df_ds3.query(f"kfold != {kfold}").reset_index(drop=True)

        print(train_df.shape)
        print(train_df["dataset"].value_counts())
        print(train_df["source_wsi"].value_counts())

        DATASET_NAME = f'hubmap-coco-ds3-5fold-pseudo-labeled-{str(pseudo_threshold).replace(".", "-")}-by-exp{exp_id}'
        out_file = f"../input/{DATASET_NAME}/fold{kfold}/train/annotation_coco.json"
        image_prefix = f"../input/{DATASET_NAME}/fold{kfold}/train"

        add_pseudo_label_and_to_coco_batch(
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
    parser.add_argument(
        "--pseudo-labeling-threshold",
        "-th",
        type=str,
        required=True,
        help="pseudo labeling threshold",
    )
    args = parser.parse_args()
    main(args)
