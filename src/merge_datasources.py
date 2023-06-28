import glob
import os
import shutil
from pathlib import Path

from mmengine.fileio import dump, load

EXP_ID = "021"
pseudo_threshold = '0-8'
DATA_SOURCE_1 = "/workspace/kaggle_hubmap_2023/input/hubmap-converted-to-coco-ds1-5fold"
DATA_SOURCE_2 = f"/workspace/kaggle_hubmap_2023/input/hubmap-converted-to-coco-ds2-5fold-pseudo-labeled-{pseudo_threshold}"
DATA_SOURCE_3 = f"/workspace/kaggle_hubmap_2023/input/hubmap-converted-to-coco-ds3-5fold-pseudo-labeled-{pseudo_threshold}-by-exp{EXP_ID}"

MERGE_DATA_SOURCE_NAME = f"ds1_ds2_ds3_th_{pseudo_threshold.replace('-', '_')}"
for kfold in [0, 1, 2, 3, 4]:
    image_prefix = f"/workspace/kaggle_hubmap_2023/input/{MERGE_DATA_SOURCE_NAME}/fold{kfold}/train"
    out_file = f"/workspace/kaggle_hubmap_2023/input/{MERGE_DATA_SOURCE_NAME}/fold{kfold}/train/annotation_coco.json"
    os.makedirs(image_prefix, exist_ok=True)

    source_images1 = glob.glob(f"{DATA_SOURCE_1}/fold{kfold}/train/*.png")
    source_images2 = glob.glob(f"{DATA_SOURCE_2}/fold{kfold}/train/*.png")
    source_images3 = glob.glob(f"{DATA_SOURCE_3}/fold{kfold}/train/*.png")
    source_images = source_images1 + source_images2 + source_images3
    destination_directory = f"/workspace/kaggle_hubmap_2023/input/{MERGE_DATA_SOURCE_NAME}/fold{kfold}/train/"

    for png_path in source_images:
        shutil.copy(png_path, destination_directory)

    anno1 = load(Path(DATA_SOURCE_1) / f"fold{kfold}" / "train/annotation_coco.json")
    anno2 = load(Path(DATA_SOURCE_2) / f"fold{kfold}" / "train/annotation_coco.json")
    anno3 = load(Path(DATA_SOURCE_3) / f"fold{kfold}" / "train/annotation_coco.json")

    new_image = []
    new_image += anno1["images"]
    image_offset = len(anno1["images"])
    image_offset2 = image_offset + len(anno2["images"])
 
    for img_dict in anno2["images"]:
        img_dict["id"] += image_offset
        new_image.append(img_dict)

    print(len(new_image))

    for img_dict in anno3["images"]:
        img_dict["id"] += image_offset2
        new_image.append(img_dict)

    print(len(new_image))

    new_anno = []
    new_anno += anno1["annotations"]
    annotation_offset = len(anno1["annotations"])
    annotation_offset2 = annotation_offset + len(anno2["annotations"])

    for anno_dict in anno2["annotations"]:
        anno_dict["id"] += annotation_offset
        anno_dict["image_id"] += image_offset

        new_anno.append(anno_dict)

    print(len(new_anno))

    for anno_dict in anno3["annotations"]:
        anno_dict["id"] += annotation_offset2
        anno_dict["image_id"] += image_offset2

        new_anno.append(anno_dict)

    print(len(new_anno))

    coco_format_json = dict(
        images=new_image,
        annotations=new_anno,
        categories=[{"id": 0, "name": "blood_vessel"}],
    )

    dump(coco_format_json, out_file)
