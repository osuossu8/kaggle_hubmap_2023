import os
import subprocess
from pathlib import Path
from mmengine.fileio import dump, load

pseudo_threshold = '0-9' # '0-8'
DATA_SOURCE_1 = '/workspace/kaggle_hubmap_2023/input/hubmap-converted-to-coco-ds1-5fold'
DATA_SOURCE_2 = f'/workspace/kaggle_hubmap_2023/input/hubmap-converted-to-coco-ds2-5fold-pseudo-labeled-{pseudo_threshold}'

MERGE_DATA_SOURCE_NAME = f'ds1_ds2_th_{pseudo_threshold}'
for kfold in [0,1,2,3,4]:
    image_prefix = f'/workspace/kaggle_hubmap_2023/input/{MERGE_DATA_SOURCE_NAME}/fold{kfold}/train'
    out_file = f'/workspace/kaggle_hubmap_2023/input/{MERGE_DATA_SOURCE_NAME}/fold{kfold}/train/annotation_coco.json'
    os.makedirs(image_prefix, exist_ok=True)

    data_copy_cmd1 = f'{DATA_SOURCE_1}/fold{kfold}/train/* /workspace/kaggle_hubmap_2023/input/{MERGE_DATA_SOURCE_NAME}/fold{kfold}/train/'
    data_copy_cmd2 = f'{DATA_SOURCE_2}/fold{kfold}/train/* /workspace/kaggle_hubmap_2023/input/{MERGE_DATA_SOURCE_NAME}/fold{kfold}/train/'

    subprocess.run(data_copy_cmd1.split())
    subprocess.run(data_copy_cmd2.split())
   
    anno1 = load(Path(DATA_SOURCE_1)/ f'fold{kfold}'/'train/annotation_coco.json')
    anno2 = load(Path(DATA_SOURCE_2)/ f'fold{kfold}'/'train/annotation_coco.json')

    new_image = []
    new_image += anno1['images']
    image_offset = len(anno1['images'])
    annotation_offset = len(anno1['annotations'])

    for img_dict in anno2['images']:
        img_dict['id'] += image_offset

        new_image.append(img_dict)

    print(len(new_image))
    
    new_anno = []
    new_anno += anno1['annotations']

    for anno_dict in anno2['annotations']:
        anno_dict['id'] += annotation_offset
        anno_dict['image_id'] += image_offset

        new_anno.append(anno_dict)

    print(len(new_anno))
    
    coco_format_json = dict(
        images=new_image,
        annotations=new_anno,
        categories=[{
            'id': 0,
            'name': 'blood_vessel'
        }])

    dump(coco_format_json, out_file)
    