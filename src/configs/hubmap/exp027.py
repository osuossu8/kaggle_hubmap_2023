# Swin-T	-	ImageNet-1K	50e	15.3	-	47.7	44.7	config	model | log

fold = None
EXP_ID = '027'
SEED = 42
EPOCHS = 20
NUM_CLASSES = 1
IMG_SIZE_HW = (1280, 1600) # (768, 768)
CLASSES = ('blood_vessel')
data_root = f'/workspace/kaggle_hubmap_2023/input/hubmap-converted-to-coco-ds1-5fold/fold{fold}/'
work_dir = f'./work_dirs/exp{EXP_ID}/fold{fold}'

# The new config inherits a base config to highlight the necessary modification
# _base_ = '/workspace/kaggle_hubmap_2023/src/mmdetection/configs/mask2former/mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_coco.py'
_base_ = '/workspace/kaggle_hubmap_2023/src/mmdetection/configs/cascade_rcnn/cascade-mask-rcnn_x101-64x4d_fpn_20e_coco.py'

model = dict(
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=NUM_CLASSES,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=NUM_CLASSES,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=NUM_CLASSES,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ],
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=NUM_CLASSES,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))
    ))

# https://mmdetection.readthedocs.io/en/latest/migration/config_migration.html#configuration-for-saving-checkpoints
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', 
        interval=1, 
        # save_best='coco/segm_mAP', 
        save_best='coco/bbox_mAP', 
        max_keep_ckpts=2,
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
        times=1,
        dataset=dict(
            data_root=data_root,
            metainfo=metainfo,
            pipeline=train_pipeline,
            ann_file='train/annotation_coco.json',
            data_prefix=dict(img='train/'))
    )
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

LR = 5e-4
param_scheduler = [
    # learning rate scheduler
    # During the first 10 epochs, learning rate increases from 0 to lr * 10
    # during the next 20 epochs, learning rate decreases from lr * 10 to lr * 1e-4
    dict(
        type='CosineAnnealingLR',
        T_max=10,
        eta_min=LR * 10,
        begin=0,
        end=10,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=20,
        eta_min=LR * 1e-4,
        begin=10,
        end=30,
        by_epoch=True,
        convert_to_iter_based=True),
    # momentum scheduler
    # During the first 10 epochs, momentum increases from 0 to 0.85 / 0.95
    # during the next 20 epochs, momentum increases from 0.85 / 0.95 to 1
    dict(
        type='CosineAnnealingMomentum',
        T_max=10,
        eta_min=0.85 / 0.95,
        begin=0,
        end=10,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=20,
        eta_min=1,
        begin=10,
        end=30,
        by_epoch=True,
        convert_to_iter_based=True)
]
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=LR, momentum=0.9, weight_decay=0.0001))

randomness=dict(seed=SEED)

# We can use the pre-trained Mask RCNN model to obtain higher performance
# load_from = 'https://download.openmmlab.com/mmdetection/v3.0/mask2former/mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_coco/mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_coco_20220508_091649-01b0f990.pth'
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_x101_64x4d_fpn_20e_coco/cascade_mask_rcnn_x101_64x4d_fpn_20e_coco_20200512_161033-bdb5126a.pth'
