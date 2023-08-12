# model settings
model = dict(
    type='YOLOXWithMaskHead',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        pad_size_divisor=32),
    mask_head=dict(type='FCNMaskHead',
        num_convs=7,
        in_channels=320,
        conv_out_channels=256,
        num_classes=1),
    backbone=dict(
        type='CSPDarknet',
        deepen_factor=1.33,
        widen_factor=1.25,
        out_indices=(2, 3, 4),
        use_depthwise=False,
        spp_kernal_sizes=(5, 9, 13),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish'),
    ),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[320, 640, 1280],
        out_channels=320,
        num_csp_blocks=4,
        use_depthwise=False,
        upsample_cfg=dict(scale_factor=2, mode='nearest'),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish')),
    bbox_head=dict(
        type='YOLOXHead',
        num_classes=3,
        in_channels=320,
        feat_channels=320,
        stacked_convs=2,
        strides=(8, 16, 32),
        use_depthwise=False,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish'),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        loss_bbox=dict(
            type='IoULoss',
            mode='square',
            eps=1e-16,
            reduction='sum',
            loss_weight=5.0),
        loss_obj=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        loss_l1=dict(type='L1Loss', reduction='sum', loss_weight=1.0)),
    train_cfg=dict(
        mask_pos_mode='weighted_sum',
        mask_roi_size=28,
        assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)),
    init_cfg=dict(
        type='Pretrained',
        checkpoint=('https://download.openmmlab.com/mmdetection/v2.0/yolox/'
             'yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth')
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
        metric=['bbox'],
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
    optimizer=dict(type='SGD', lr=0.01 / 2, momentum=0.9, weight_decay=0.0001))

auto_scale_lr = dict(enable=True, base_batch_size=64)

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
