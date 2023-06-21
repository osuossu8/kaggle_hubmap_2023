fold = 0
SEED = 42
EPOCHS = 30
NUM_CLASSES = 1
DATA_ROOT = f'/workspace/kaggle_hubmap_2023/input/hubmap-converted-to-coco-ds1-5fold/fold{fold}/'
IMG_SIZE_HW = (128, 128) # (768, 768) # (512, 512)
CLASSES = ('blood_vessel', )

# The new config inherits a base config to highlight the necessary modification
_base_ = '/workspace/kaggle_hubmap_2023/src/mmdetection/configs/cascade_rcnn/cascade_mask_rcnn_x101_32x4d_fpn_mstrain_3x_coco.py'

# Modify dataset related settings
dataset_type = 'CocoDataset'
data_root = DATA_ROOT
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
model = dict(
        type='CascadeRCNN',
roi_head = dict(
    type='CascadeRoIHead',
    bbox_head=[
        dict(
            type='Shared2FCBBoxHead',
            num_classes=1,
        ),
        dict(
            type='Shared2FCBBoxHead',
            num_classes=1,
        ),
        dict(
            type='Shared2FCBBoxHead',
            num_classes=1,
        ),
    ],
    mask_head=dict(
        type='FCNMaskHead',
        num_classes=1,
        )
    )
)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Resize',
        img_scale=IMG_SIZE_HW,
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=IMG_SIZE_HW,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# Use RepeatDataset to speed up training
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'train/annotation_coco.json',
            img_prefix=data_root + 'train/',
            classes=CLASSES,
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val/annotation_coco.json',
        img_prefix=data_root + 'val/',
        classes=CLASSES,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'val/annotation_coco.json',
        img_prefix=data_root + 'val/',
        classes=CLASSES,
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric=['bbox', 'segm'])

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_x101_32x4d_fpn_mstrain_3x_coco/cascade_mask_rcnn_x101_32x4d_fpn_mstrain_3x_coco_20210706_225234-40773067.pth'
