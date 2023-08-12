# model settings
norm_cfg = dict(type='GN', num_groups=32)
checkpoint = 'https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_base_patch4_window12_384_22kto1k-d59b0d1d.pth'
model = dict(
    type='RTMDetWithMaskHead',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    mask_head=dict(type='FCNMaskHead',
        num_convs=7,
        in_channels=256,
        conv_out_channels=256,
        num_classes=1),
    backbone=dict(
        type='mmpretrain.SwinTransformer',
        arch=dict(
            embed_dims=128,
            depths=[2, 2, 18, 2, 1],
            num_heads=[4, 8, 16, 32, 64],
        ),
        img_size=384,
        drop_path_rate=0.2,
        stage_cfgs=dict(block_cfgs=dict(window_size=12)),
        out_indices=(1, 2, 3, 4),
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
        num_classes=3,
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
        mask_pos_mode='weighted_sum',
        mask_roi_size=28,
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
    init_cfg=dict(
        type='Pretrained',
        checkpoint='work_dirs/s/epoch_30.pth'
    )
)

model = dict(
    type='MultiEMADetector',
    momentums=[0.001, 0.0005, 0.00025],
    detector=model,
)

# dataset settings
dataset_type = 'CocoDataset'
data_root = '../data/'
img_prefix = '../data/train'
metainfo = dict(classes=('blood_vessel', 'glomerulus', 'unsure'))
backend_args = None

img_scale = (768, 768)
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
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
    dict(type='CropGtMasks', roi_size=56),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataset1 = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='dtrain0i.json',
    data_prefix=dict(img=img_prefix),
    metainfo=metainfo,
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline,
    backend_args=backend_args)
train_dataset2 = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='dtrain_dataset2_dropdup.json',
    data_prefix=dict(img=img_prefix),
    metainfo=metainfo,
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline,
    backend_args=backend_args)
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='GroupMultiSourceSampler',
        batch_size=8,
        source_ratio=[3, 5]),
    dataset=dict(
        type='ConcatDataset',
        datasets=[train_dataset1, train_dataset2]))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='dval0i.json',
        data_prefix=dict(img=img_prefix),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = [
    dict(
        type='FastCocoMetric',
        ann_file=data_root + val_dataloader['dataset']['ann_file'],
        metric=['bbox', 'segm'],
        classwise=True,
        format_only=False,
        backend_args=backend_args),
]
test_evaluator = val_evaluator

# training schedule for 1x
imgs_per_epoch = 338  # dataset 1
iters_per_epoch = imgs_per_epoch // 3
train_cfg = dict(type='IterBasedTrainLoop',
                 max_iters=200 * iters_per_epoch,
                 val_interval=iters_per_epoch * 9)
val_cfg = dict(type='MultiEMAValLoop')
test_cfg = dict(type='TestLoop')

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }),
    optimizer=dict(
        type='AdamW',
        lr=0.0005,
        betas=(0.9, 0.999),
        weight_decay=0.05))

auto_scale_lr = dict(enable=True, base_batch_size=16)

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=50),
]

default_scope = 'mmdet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False,
                    interval=train_cfg['val_interval'],
                    save_optimizer=False),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))


custom_hooks = [
    dict(type='MultiEMAHook',
         skip_buffers=False,
         interval=1)
]

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=False)

log_level = 'INFO'
resume = False

custom_imports = dict(imports=['custom_modules'], allow_failed_imports=False)
