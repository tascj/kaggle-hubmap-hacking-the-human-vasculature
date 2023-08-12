# model settings
norm_cfg = dict(type='GN', num_groups=32)
checkpoint = 'https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_large_patch4_window12_384_22kto1k-0a40944b.pth'
model = dict(
    type='RTMDet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='mmpretrain.SwinTransformer',
        arch='large',
        img_size=384,
        drop_path_rate=0.2,
        stage_cfgs=dict(block_cfgs=dict(window_size=12)),
        out_indices=(1, 2, 3),
        with_cp=False,
        init_cfg=dict(
            type='Pretrained', prefix='backbone', checkpoint=checkpoint)),
    neck=dict(
        type='CSPNeXtPAFPN',
        in_channels=[384, 768, 1536],
        out_channels=256,
        num_csp_blocks=4,
        expand_ratio=0.5,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='RTMDetHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=2,
        feat_channels=256,
        anchor_generator=dict(
            type='MlvlPointGenerator', offset=0, strides=[8, 16, 32]),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        with_objectness=False,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True)),
    train_cfg=dict(
        assigner=dict(type='DynamicSoftLabelAssigner', topk=13),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=30000,
        min_bbox_size=0,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.65),
        max_per_img=300),
)

# dataset settings
dataset_type = 'CocoDataset'
data_root = '../data/coco/'
backend_args = None


img_scale = (640, 640)
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='CachedMosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomResize',
        scale=img_scale,
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=img_scale),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=img_scale, pad_val=dict(img=(114., 114., 114.))),
    dict(
        type='CachedMixUp',
        img_scale=img_scale,
        ratio_range=(1.0, 1.0),
        max_cached_images=20,
        pad_val=(114., 114., 114.)),
    dict(type='PackDetInputs')
]

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=img_scale,
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=img_scale),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=img_scale, pad_val=dict(img=(114., 114., 114.))),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='Pad', size=img_scale, pad_val=dict(img=(114., 114., 114.))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        backend_args=backend_args,
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric='bbox',
    backend_args=backend_args)
test_evaluator = val_evaluator

# training schedule for 1x
base_lr = 0.0001
max_epochs = 30
num_last_epochs = 3
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        # use cosine lr from 5 to 285 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.01,
        begin=0,
        T_max=max_epochs - num_last_epochs,
        end=max_epochs - num_last_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        # use fixed lr during last 15 epochs
        type='ConstantLR',
        by_epoch=True,
        factor=1,
        begin=max_epochs - num_last_epochs,
        end=max_epochs,
    )
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.01),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        flat_decay_mult=0.0,
        custom_keys={
            '.absolute_pos_embed': dict(decay_mult=0.0),
            '.relative_position_bias_table': dict(decay_mult=0.0)
        }),
)

auto_scale_lr = dict(enable=True, base_batch_size=16)

default_scope = 'mmdet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

custom_hooks = [
    dict(
        type='PipelineSwitchHook',
        switch_epoch=max_epochs - num_last_epochs,
        switch_pipeline=train_pipeline_stage2)
]

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False

custom_imports = dict(
    imports=['custom_modules'],
    allow_failed_imports=False)
