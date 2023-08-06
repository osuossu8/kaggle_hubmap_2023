# Swin-T	-	ImageNet-1K	50e	15.3	-	47.7	44.7	config	model | log

DATASET_NAME = 'hubmap-converted-to-coco-5fold-ds1'
TRAIN_DATASET_NAME = 'merge_ds1_ds2_given_ds2_pseudolabel_th08_bylate001'
fold = None
EXP_ID = 'late005'
SEED = 42
EPOCHS = 20
BATCH_SIZE = 4 # 2
NUM_CLASSES = 3
IMG_SIZE_HW = (768, 768) # (1024, 1024)
CLASSES = ('blood_vessel', 'glomerulus', 'unsure')
data_root = f'/workspace/kaggle_hubmap_2023/input/{DATASET_NAME}/fold{fold}/'
work_dir = f'./work_dirs/{EXP_ID}/fold{fold}'

# The new config inherits a base config to highlight the necessary modification
# _base_ = '/workspace/kaggle_hubmap_2023/src/mmdetection/configs/cascade_rcnn/cascade-mask-rcnn_x101-64x4d_fpn_20e_coco.py'
_base_ = '/workspace/kaggle_hubmap_2023/src/mmdetection/configs/mask_rcnn/mask-rcnn_x101-64x4d_fpn_ms-poly_3x_coco.py'

image_size = IMG_SIZE_HW

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=NUM_CLASSES), mask_head=dict(num_classes=NUM_CLASSES)))

# https://mmdetection.readthedocs.io/en/latest/migration/config_migration.html#configuration-for-saving-checkpoints
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True,
        save_last=True,
        interval=1, 
        save_best='coco/segm_mAP',
        max_keep_ckpts=1,
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

# Modify dataset related settings
dataset_type = 'CocoDataset'
metainfo = {
    'classes': CLASSES,
    'palette': [
        (220, 20, 60), (230, 30, 70), (210, 10, 50),
    ]
}

# https://mmdetection.readthedocs.io/en/latest/migration/config_migration.html?highlight=checkpoint
img_scale = IMG_SIZE_HW
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        to_float32=True,
        backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', prob=0.5, direction=['horizontal', 'vertical']),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomRotateScaleCrop',
         img_scale=img_scale,
         angle_range=(-180, 180),
         scale_range=(0.1, 2.0),
         border_value=(114, 114, 114),
         rotate_prob=0.5,
         scale_prob=1.0,
         hflip_prob=0.5,
         rot90_prob=1.0,
         mask_dtype='u1',
    ),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-5, 1e-5), by_mask=True),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        to_float32=True,
        backend_args={{_base_.backend_args}}),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=BATCH_SIZE,
    dataset=dict(
        times=1,
        dataset=dict(
            data_root=data_root,
            metainfo=metainfo,
            pipeline=train_pipeline,
            ann_file='train/annotation_coco.json',
            data_prefix=dict(img='train/'),)
    )
)

val_dataloader = dict(
    batch_size=BATCH_SIZE,
    num_workers=1,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        pipeline=test_pipeline,
        ann_file='val/annotation_coco.json',
        data_prefix=dict(img='val/'),)
)
test_dataloader = val_dataloader

# Modify metric related settings
val_evaluator = dict(ann_file=data_root + 'val/annotation_coco.json')
test_evaluator = val_evaluator

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=EPOCHS, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

LR = 3e-6 # 3e-5 # 5e-4
param_scheduler = [
    # learning rate scheduler
    # During the first 10 epochs, learning rate increases from 0 to lr * 10
    # during the next 20 epochs, learning rate decreases from lr * 10 to lr * 1e-4
    dict(
        type='CosineAnnealingLR',
        T_max=8,
        eta_min=LR * 10,
        begin=0,
        end=8,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=16,
        eta_min=LR * 1e-4,
        begin=8,
        end=20,
        by_epoch=True,
        convert_to_iter_based=True),
    # momentum scheduler
    # During the first 10 epochs, momentum increases from 0 to 0.85 / 0.95
    # during the next 20 epochs, momentum increases from 0.85 / 0.95 to 1
    dict(
        type='CosineAnnealingMomentum',
        T_max=8,
        eta_min=0.85 / 0.95,
        begin=0,
        end=8,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=16,
        eta_min=1,
        begin=8,
        end=20,
        by_epoch=True,
        convert_to_iter_based=True)
]
 
optim_wrapper = dict(
    _delete_ = True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=LR,
        betas=(0.9, 0.999),
        weight_decay=0.0001))

randomness=dict(seed=SEED)

# We can use the pre-trained Mask RCNN model to obtain higher performance
# load_from = 'https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_x101_64x4d_fpn_20e_coco/cascade_mask_rcnn_x101_64x4d_fpn_20e_coco_20200512_161033-bdb5126a.pth'
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco/mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco_20210526_120447-c376f129.pth'

# https://github.com/tascj/kaggle-hubmap-hacking-the-human-vasculature/blob/eb35dc7c629a7af7ba711aeefa680dd4c7389e3f/configs/r0.py#L231
custom_imports = dict(imports=['custom_modules'], allow_failed_imports=False)