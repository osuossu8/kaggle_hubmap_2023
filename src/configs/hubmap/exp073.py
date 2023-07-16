# Swin-T	-	ImageNet-1K	50e	15.3	-	47.7	44.7	config	model | log

DATASET_NAME = 'hubmap-converted-to-coco-5fold-v2-3class'
fold = None
EXP_ID = '073'
SEED = 42
EPOCHS = 20
BATCH_SIZE = 2
NUM_CLASSES = 3
IMG_SIZE_HW = (1024, 1024) # (768, 768) # (640, 640)
CLASSES = ('blood_vessel', 'glomerulus', 'unsure')
data_root = f'/workspace/kaggle_hubmap_2023/input/{DATASET_NAME}/fold{fold}/'
work_dir = f'./work_dirs/exp{EXP_ID}/fold{fold}'

# The new config inherits a base config to highlight the necessary modification
_base_ = '/workspace/kaggle_hubmap_2023/src/mmdetection/configs/mask2former/mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_coco.py'

num_things_classes = NUM_CLASSES
num_stuff_classes = 0
num_classes = num_things_classes + num_stuff_classes
image_size = IMG_SIZE_HW
batch_augments = [
    dict(
        type='BatchFixedSizePad',
        size=image_size,
        img_pad_value=0,
        pad_mask=True,
        mask_pad_value=0,
        pad_seg=False)
]
data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32,
    pad_mask=True,
    mask_pad_value=0,
    pad_seg=False,
    batch_augments=batch_augments)

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    data_preprocessor=data_preprocessor,
    panoptic_head=dict(
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        loss_cls=dict(class_weight=[1.0] * num_classes + [0.1])),
    panoptic_fusion_head=dict(
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes),
    test_cfg=dict(panoptic_on=False))

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
        # save_best='coco/bbox_mAP', 
        max_keep_ckpts=2,
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

log_processor = dict(
        type='LogProcessor', 
        window_size=50, 
        by_epoch=True
)


# Modify dataset related settings
dataset_type = 'CocoDataset'
metainfo = {
    'classes': CLASSES,
    'palette': [
        (220, 20, 60), (230, 30, 70), (210, 10, 50),
    ]
}

# https://mmdetection.readthedocs.io/en/latest/migration/config_migration.html?highlight=checkpoint

# dataset settings
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        to_float32=True,
        backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    # large scale jittering
    dict(
        type='RandomResize',
        scale=image_size,
        ratio_range=(0.1, 2.0),
        resize_type='Resize',
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=image_size,
        crop_type='absolute',
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5, direction=['horizontal', 'vertical']),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-5, 1e-5), by_mask=True),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        to_float32=True,
        backend_args={{_base_.backend_args}}),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=BATCH_SIZE,
    num_workers=1,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        pipeline=train_pipeline,
        ann_file='train/annotation_coco.json',
        data_prefix=dict(img='train/'))
)

val_dataloader = dict(
    batch_size=BATCH_SIZE,
    num_workers=1,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        pipeline=test_pipeline,
        ann_file='val/annotation_coco.json',
        data_prefix=dict(img='val/'))
)
test_dataloader = val_dataloader

# Modify metric related settings
val_evaluator = dict(ann_file=data_root + 'val/annotation_coco.json')
test_evaluator = val_evaluator

train_cfg = dict(
        _delete_ = True,
        type='EpochBasedTrainLoop', 
        max_epochs=EPOCHS, 
        val_interval=1)
#train_cfg = dict(type='IterBasedTrainLoop', val_interval=1)
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
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/mask2former/mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_coco/mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_coco_20220508_091649-01b0f990.pth'
# checkpoint_file = glob.glob(f'/workspace/kaggle_hubmap_2023/src/work_dirs/exp066/fold{fold}/best_coco_segm_mAP_epoch_*.pth')[-1]