import os
import json
from PIL import Image
from collections import Counter

import numpy as np
import pandas as pd
import tifffile as tiff
from tqdm import tqdm

import cv2
from pathlib import Path

# https://www.kaggle.com/code/itsuki9180/hubmap-making-dataset
# 5fold run version
def make_seg_mask(tiles_dict):
    mask = np.zeros((512, 512), dtype=np.float32)
    for annot in tiles_dict['annotations']:
        cords = annot['coordinates']
        if annot['type'] == "blood_vessel":
            for cd in cords:
                rr, cc = np.array([i[1] for i in cd]), np.asarray([i[0] for i in cd])
                mask[rr, cc] = 1
                
    contours,_ = cv2.findContours((mask*255).astype(np.uint8), 1, 2)
    zero_img = np.zeros([mask.shape[0], mask.shape[1], 3], dtype="uint8")

    for p in contours:
        cv2.fillPoly(zero_img, [p], (255, 255, 255))

    contours, hierarchy = cv2.findContours(mask.astype("uint8"), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    img_with_area = zero_img

    for i in range(len(contours)):
        cv2.fillPoly(img_with_area, [contours[i][:,0,:]], (255-4*(i+1),255-4*(i+1),255-4*(i+1)), lineType=cv2.LINE_8, shift=0)
            
    return img_with_area  


def convert_dataset(df, image_prefix):
    os.makedirs(f'{image_prefix}/image', exist_ok=True)
    os.makedirs(f'{image_prefix}/mask', exist_ok=True)

    tiles_dicts = df['annot_dict'].values
    for i, tldc in enumerate(tqdm(tiles_dicts)):
        tldc = eval(tldc)
        array = tiff.imread(f'/workspace/kaggle_hubmap_2023/input/train/{tldc["id"]}.tif')
        img_example = Image.fromarray(array)
        img = np.array(img_example)
        mask = make_seg_mask(tldc)
        
        if np.sum(mask)>0:

            cv2.imwrite(f'{image_prefix}/image/{tldc["id"]}.png', img)
            cv2.imwrite(f'{image_prefix}/mask/{tldc["id"]}_mask.png', mask)


data_root = Path("../input")
with open(data_root / 'polygons.jsonl', 'r') as json_file:
    json_list = list(json_file)

tile_meta_df = pd.read_csv(data_root / "tile_meta_with_5fold.csv")

df = pd.DataFrame(json_list, columns=['annot_dict'])
df['id'] = df['annot_dict'].map(lambda x: eval(x)['id'])
df = pd.merge(df, tile_meta_df, on='id', how='inner')
print(df.shape)
print(df.head())

ds1_wsi12 = df.query('dataset == 1 & source_wsi in [1, 2]')
ds2_wsi34 = df.query('dataset == 2 & source_wsi in [3, 4]')

df_use = pd.concat([ds1_wsi12, ds2_wsi34]).reset_index(drop=True)
print(df_use.shape)

DATASET_NAME = 'hubmap-itk-san-5fold'


for kfold in [0,1,2,3,4]:
    val_df = df_use.query(f"kfold == {kfold}").reset_index(drop=True)
    train_df = df_use.query(f"kfold != {kfold}").reset_index(drop=True)
    print(train_df.shape, val_df.shape)

    convert_dataset(df=train_df,
                            image_prefix=f'../input/{DATASET_NAME}/fold{kfold}/train')
    convert_dataset(df=val_df,
                            image_prefix=f'../input/{DATASET_NAME}/fold{kfold}/val')