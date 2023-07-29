from ensemble_boxes import non_maximum_weighted

from PIL import Image
import cv2
import numpy as np
import pandas as pd
import torch
import tifffile as tiff
from pathlib import Path
from tqdm.auto import tqdm

import glob
import pprint
from mmengine import Config
from mmdet.apis import init_detector, inference_detector

import pycocotools
from pycocotools.coco import COCO
from hubmap3_src.utils_coco import coco2mask
from hubmap3_src.comp_metric import get_score

from typing import List, Tuple, Union

import warnings 
warnings.filterwarnings('ignore')


from skimage.morphology import binary_dilation


# https://www.kaggle.com/datasets/markunys/ensemble-boxes
# https://www.kaggle.com/competitions/sartorius-cell-instance-segmentation/discussion/297998
# ensemble_masks_wbf_float16.py
import warnings
import numpy as np
from numba import jit

#@jit(nopython=True)
def bb_intersection_over_union(A, B): #-> float:
    xA = max(A[0], B[0])
    yA = max(A[1], B[1])
    xB = min(A[2], B[2])
    yB = min(A[3], B[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    if interArea == 0:
        return 0.0

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (A[2] - A[0]) * (A[3] - A[1])
    boxBArea = (B[2] - B[0]) * (B[3] - B[1])

    unionArea = boxAArea + boxBArea - interArea
    if unionArea == 0:
        return 0.0
    
    iou = interArea / unionArea
    return iou

#@jit(nopython=True)
def get_weighted_mask(masks, scores):
    mask = np.zeros(masks[0].shape, dtype=np.float16)
    conf = 0
    conf_list = []
    for m, s in zip(masks, scores):
        mask += s * m
        conf += s
        conf_list.append(s)
    score = np.max(conf_list)
    mask = mask / conf
    return mask, score, conf_list

def get_weighted_box(boxes, scores):
    box = np.zeros(4, dtype=np.float16)
    conf = 0
    conf_list = []
    for b, s in zip(boxes, scores):
        box += s * b
        conf += s
        conf_list.append(s)
    score = np.max(conf_list)
    box = box / conf
    return box, score


def get_weighted_box_and_mask(boxes, masks, scores):
    box = np.zeros(4, dtype=np.float16)
    mask = np.zeros(masks[0].shape, dtype=np.float16)
    conf = 0
    conf_list = []
    for b, m, s in zip(boxes, masks, scores):
        box += s * b
        mask += s * m
        conf += s
        conf_list.append(s)
    score = np.max(conf_list)
    box = box / conf
    mask = mask / conf
    return box, mask, score, conf_list


def find_matching_box(boxes_list, new_box, match_iou):
    best_iou = match_iou
    best_index = -1
    for i in range(len(boxes_list)):
        box = boxes_list[i]
        iou = bb_intersection_over_union(box, new_box)
        if iou > best_iou:
            best_index = i
            best_iou = iou

    return best_index, best_iou

def weighted_masks_fusion(masks, boxes, scores, iou_thr=0.7, skip_mask_thr=0.0, 
                        conf_type='max_weight', soft_weight=5, thresh_type=None, 
                        num_thresh=4, num_models=5):
    masks = masks[scores > skip_mask_thr]
    boxes = boxes[scores > skip_mask_thr]
    scores = scores[scores > skip_mask_thr]
    
    new_masks = []
    new_boxes = []
    new_scores = []
    weighted_boxes = []
    weighted_scores = []
    # Clusterize boxes
    for i in range(len(masks)):
            
        index, best_iou = find_matching_box(weighted_boxes, boxes[i], iou_thr)
        if index != -1:
            new_masks[index].append(masks[i])
            new_boxes[index].append(boxes[i])
            new_scores[index].append(scores[i])
            weighted_boxes[index], weighted_scores[index] = get_weighted_box(new_boxes[index], new_scores[index])
        else:
            new_masks.append([masks[i]])
            new_boxes.append([boxes[i].copy()])
            new_scores.append([scores[i].copy()])
            weighted_boxes.append(boxes[i].copy())
            weighted_scores.append(scores[i].copy())
            
    ens_masks = []
    ens_scores = []
    for nmasks, nscores in zip(new_masks, new_scores):
        mask, score, conf_list = get_weighted_mask(nmasks, nscores)
        if thresh_type == 'num_thresh':
            if len(conf_list) >= num_thresh:
                ens_masks.append(mask)
            else:
                continue
        else:
            ens_masks.append(mask)

        if conf_type =='max_weight':
            ens_scores.append(score * min(len(conf_list), num_models) / num_models)
        elif conf_type == 'max':
            ens_scores.append(score)
        elif conf_type == 'soft_weight':
            ens_scores.append(score * (min(len(conf_list), num_models) + soft_weight) / (soft_weight + num_models))

    return ens_masks, ens_scores, weighted_boxes


def filter_result(
    pred_classes: List[int],
    pred_scores: List[float], 
    pred_bboxes: List[List[float]], 
    pred_masks: np.ndarray, # (num_mask, 512, 512)
) -> Tuple[List[float], List[List[float]], np.ndarray]:
    scores, bboxes, masks = [], [], []
    for c,s,b,m in zip(pred_classes, pred_scores, pred_bboxes, pred_masks):
        if c != 0:
            continue
        if s > 0:
            scores.append(s)
            bboxes.append(b)
            masks.append(m)
    return scores, bboxes, masks


def nms_predictions_v3(scores, bboxes, masks, weights=None,
                    iou_th=.5, shape=(512, 512)):
    he, wd = shape[0], shape[1]
    boxes_list = [[[x[0] / wd, x[1] / he, x[2] / wd, x[3] / he] for x in bboxes]]
    scores_list = [[x for x in scores]]
    classes_list = [[0 for x in scores]]
    if weights is not None:
        weighted_scores_list = [x * weights[0] for x in scores]
    nms_bboxes, nms_scores, nms_classes = non_maximum_weighted(
        boxes_list, 
        scores_list, 
        classes_list,
        weights=weights, # weight with scores
        iou_thr=IOU_TH,
        skip_box_thr=0.0001,
    )
    nms_masks = []
    for s in nms_scores:
        if weights is not None:
            abs_diff = np.abs(np.array(scores) * weights[0] - s)
            index_of_closest_value = np.argmin(abs_diff)
            nms_masks.append(masks[index_of_closest_value])
        else:
            nms_masks.append(masks[scores.index(s)])
    nms_scores, nms_bboxes, nms_masks = zip(*sorted(zip(nms_scores, nms_bboxes, nms_masks), reverse=True))
    return nms_scores, nms_bboxes, nms_masks


def dilate_predict_mask(out_mask):
    # from https://www.kaggle.com/code/hengck23/lb4-09-baseline-yolov7
    for i in range(len(out_mask)):
        out_mask[i] = binary_dilation(out_mask[i])
    return out_mask


def cv2_dilate_predict_mask(out_mask, kernel_size, areas=None):
    iterations = 1
    # from https://www.kaggle.com/code/hengck23/lb4-09-baseline-yolov7
    for i in range(len(out_mask)):
        # kernel = np.ones(shape=(3, 3), dtype=np.uint8)
        # kernel = np.ones(shape=(2, 1), dtype=np.uint8)
        kernel = np.ones(shape=kernel_size, dtype=np.uint8)
        if areas is None:
            out_mask[i] = cv2.dilate(out_mask[i], kernel, iterations=iterations)
        else:
            mask_area = areas[i]
            if mask_area < 32*32:
                out_mask[i] = cv2.dilate(out_mask[i], kernel, iterations=iterations)
    return out_mask


dataDir = Path("../input/hubmap-converted-to-coco-5fold-v2-3class-val/fold0/val")
annFile = Path("../input/hubmap-converted-to-coco-5fold-v2-3class-val/fold0/val/annotation_coco.json")
coco = COCO(annFile)

imgIds = coco.getImgIds()
imgs = coco.loadImgs(imgIds)

# simple   dilate skimage     cv2.delite (2,1) cv2.delite (3,3)
# 0.6910558    0.680897          0.69270813        0.6716128

# 2model 
# cv2.delite (2,1)
# 0.67379326

# 3model 
# cv2.delite (2,1)
# 0.6860983

# 4model ['065', '078', '050', '055']
# cv2.delite (2,1) (3,3), dilate skimage, both (all image), cv2.delite (2,1) (only small mask)
# 0.6815282  0.6352656, 0.6543507, 0.6376714110374451, 0.6828042268753052

# 4model ['065', '078', '050', '079']
# cv2.delite (2,1)
# 0.6683653

# EXP_IDs = ['065', '078', '050', '055']
# EXP_IDs = ['065', '078', '050', '079']
EXP_IDs_list =[
    ['065', '078', '050', '079', '062'],
    ['065', '078', '050', '055', '079'],
    ['065', '079', '050', '055', '062'],
]
IOU_TH = 0.6
WMF_IOU_TH = 0.6 
WMF_SKIP_MASK_THR = 0.0
MASK_THRESHOLD = 0.1
CONF_TYPE = 'soft_weight'

# for dilation_mode in ['cv2_2_1', 'cv2_3_3', 'skimage', 'both]:
if 1:
    dilation_mode = 'cv2_2_1'
    for EXP_IDs in EXP_IDs_list:
        EXP_IDs_string = ' '.join(EXP_IDs)
        models = []
        mAP60_list = []
        for fold in [0, 1, 2, 3, 4]:
            if fold != 0:
                continue
            mAP60_list_fold = []
            for EXP_ID in EXP_IDs:
                work_dir_path = f'/external_disk/work_dirs/exp{EXP_ID}/fold{fold}'
                config_file = f'{work_dir_path}/exp{EXP_ID}.py'
                cfg = Config.fromfile(config_file)
                if EXP_ID in ['065', '079']:
                    cfg.model.test_cfg.rcnn.score_thr = 0.0001
                    cfg.model.test_cfg.rcnn.max_per_img = 300
                    cfg.model.test_cfg.rcnn.nms.type = 'soft_nms'

                checkpoint_file = glob.glob(f'{work_dir_path}/best_coco_segm_mAP_epoch_*.pth')[-1]
                model = init_detector(cfg, checkpoint_file, device='cuda:0')
                if EXP_ID in ['078']:
                    model.cfg.test_pipeline[1].scale = (1024, 1024)
                    model.cfg.test_dataloader.dataset.pipeline[1].scale = (1024, 1024)
                models.append(model) 

            dataDir = Path(f"../input/hubmap-converted-to-coco-5fold-v2-3class-val/fold{fold}/val")
            annFile = Path(f"../input/hubmap-converted-to-coco-5fold-v2-3class-val/fold{fold}/val/annotation_coco.json")
            coco = COCO(annFile)

            imgIds = coco.getImgIds()
            imgs = coco.loadImgs(imgIds)

            for img in tqdm(imgs, total=len(imgs)):
                # img : {'id': 0, 'file_name': '0067d5ad2250.png', 'height': 512, 'width': 512}
                annIds = coco.getAnnIds(imgIds=[img["id"]])
                if len(annIds) == 0:
                    continue
                anns = coco.loadAnns(annIds)
                gt_masks = [coco.annToMask(ann) for ann in anns]
                gt_bboxes = [ann["bbox"] for ann in anns]
                gt_labels = [ann["category_id"] for ann in anns]

                gt_dict = {
                    "boxes": torch.tensor(gt_bboxes),
                    "labels": torch.tensor(gt_labels),
                    "masks": torch.BoolTensor(np.array(gt_masks)),
                }
                
                ens_scores = []
                ens_bboxes = []
                ens_masks = []
                for model_ in models:
                    res = inference_detector(model_, dataDir / img["file_name"])
                    pred = res.pred_instances.numpy()
                    tmp_classes = pred.labels.tolist()
                    tmp_scores = pred.scores.tolist()
                    tmp_bboxes = pred.bboxes.tolist()
                    tmp_masks = pred.masks
                    tmp_scores, tmp_bboxes, tmp_masks = filter_result(tmp_classes, tmp_scores, tmp_bboxes, tmp_masks)
                    tmp_scores, tmp_bboxes, tmp_masks = nms_predictions_v3(
                                        tmp_scores, 
                                        tmp_bboxes,
                                        tmp_masks, 
                                        weights=None,
                                        iou_th=IOU_TH
                                    ) 
                    ens_scores.append(tmp_scores)
                    ens_bboxes.append(tmp_bboxes)
                    ens_masks.append(tmp_masks)

                ens_masks = np.concatenate(ens_masks, 0)
                ens_scores = np.concatenate(ens_scores, 0)
                ens_bboxes = np.concatenate(ens_bboxes, 0)
                # print(ens_masks.shape, ens_scores.shape, ens_bboxes.shape)

                ens_masks, ens_scores, ens_bboxes = weighted_masks_fusion(
                    ens_masks, ens_bboxes, ens_scores, iou_thr=WMF_IOU_TH, skip_mask_thr=WMF_SKIP_MASK_THR, 
                    conf_type=CONF_TYPE, soft_weight=5, thresh_type=None, 
                    num_thresh=4, 
                    num_models=len(EXP_IDs) # .555 (3 models) and .530 (5 models * 5folds)
                )
                ens_masks = np.array(ens_masks)
                ens_scores = np.array(ens_scores)

                ens_masks = ens_masks > MASK_THRESHOLD
                ens_masks = np.array(ens_masks, dtype=np.uint8)

                ens_mask_areas = [np.sum(m.flatten()) for m in ens_masks]
                ens_mask_areas = None
                if dilation_mode == 'cv2_2_1':   
                    ens_masks = cv2_dilate_predict_mask(ens_masks, kernel_size=(2, 1), areas=ens_mask_areas)
                elif dilation_mode == 'cv2_3_3':
                    ens_masks = cv2_dilate_predict_mask(ens_masks, kernel_size=(3, 3), areas=ens_mask_areas)
                elif dilation_mode == 'skimage':
                    ens_masks = dilate_predict_mask(ens_masks)
                elif dilation_mode == 'both':
                    ens_masks = cv2_dilate_predict_mask(ens_masks, kernel_size=(2, 1), areas=ens_mask_areas)
                    ens_masks = dilate_predict_mask(ens_masks)
                else:
                    pass
                ens_masks = np.array(ens_masks, dtype=bool)

                if len(ens_scores) > 0:

                    # pred dict
                    pred_dict = {
                        "boxes": torch.from_numpy(np.array(ens_bboxes)).clone(),
                        "scores": torch.from_numpy(np.array(ens_scores)).clone(),
                        "labels": torch.tensor(np.array([0 for _ in ens_bboxes])),
                        "masks": torch.BoolTensor(np.array(ens_masks)).clone(),
                    }

                    mAP60 = get_score([pred_dict], [gt_dict], is_bbox=False).numpy()
                    mAP60_list_fold.append(mAP60)
            mAP60_list_fold_avg = np.mean(mAP60_list_fold)
            mAP60_list.append(mAP60_list_fold_avg)
            print(f'fold{fold} mAP60: dilation {dilation_mode}, {mAP60_list_fold_avg}')

        mAP60_macro_avg = np.mean(mAP60_list)
        print(f'EXP_IDs_string : {EXP_IDs_string}, {mAP60_macro_avg}')

