fold = None
EXP_ID = '013'
SEED = 42
EPOCHS = 20
NUM_CLASSES = 1
IMG_SIZE_HW = (768, 768)
CLASSES = ('blood_vessel')
data_root = f'/workspace/kaggle_hubmap_2023/input/hubmap-converted-to-coco-ds1-5fold/fold{fold}/'
work_dir = f'./work_dirs/exp{EXP_ID}/fold{fold}'

# The new config inherits a base config to highlight the necessary modification
_base_ = '/workspace/kaggle_hubmap_2023/src/mmdetection/configs/convnext/mask-rcnn_convnext-t-p4-w7_fpn_amp-ms-crop-3x_coco.py'

checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-tiny_3rdparty_32xb128-noema_in1k_20220301-795e9634.pth'  # noqa


# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    backbone=dict(
        _delete_=True,
        type='mmcls.ConvNeXt',
        arch='tiny',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
    neck=dict(in_channels=[96, 192, 384, 768]),
    roi_head=dict(
        bbox_head=dict(num_classes=NUM_CLASSES,),
        mask_head=dict(num_classes=NUM_CLASSES,),
    )
)


# https://mmdetection.readthedocs.io/en/latest/migration/config_migration.html#configuration-for-saving-checkpoints
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', 
        interval=1, 
        save_best='coco/segm_mAP', 
        max_keep_ckpts=3,
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

# Modify dataset related settings
dataset_type = 'CocoDataset'
metainfo = {
    'classes': CLASSES,
    'palette': [
        (220, 20, 60),
    ]
}

# https://mmdetection.readthedocs.io/en/latest/migration/config_migration.html?highlight=checkpoint

# augmentation setting
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(type='Resize', scale=IMG_SIZE_HW, keep_ratio=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(type='RandomFlip', prob=0.5, direction=['horizontal', 'vertical']),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=IMG_SIZE_HW, keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        pipeline=train_pipeline,
        ann_file='train/annotation_coco.json',
        data_prefix=dict(img='train/'))
)

val_dataloader = dict(
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

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=EPOCHS, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

LR = 3e-5 # 5e-4
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
        T_max=15,
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
        T_max=15,
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
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/convnext/mask_rcnn_convnext-t_p4_w7_fpn_fp16_ms-crop_3x_coco/mask_rcnn_convnext-t_p4_w7_fpn_fp16_ms-crop_3x_coco_20220426_154953-050731f4.pth'
