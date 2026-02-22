_base_ = [
    '../../_base_/models/faster-rcnn_r50-dc5.py',
    '../../_base_/datasets/ceus_vid_selsa_hcc_style.py',
    '../../_base_/default_runtime.py'
]
model = dict(
    type='SELSA',
    detector=dict(
        roi_head=dict(
            type='mmtrack.SelsaRoIHead',
            bbox_head=dict(
                type='mmtrack.SelsaBBoxHead',
                num_classes=1,
                num_shared_fcs=2,
                aggregator=dict(
                    type='mmtrack.SelsaAggregator',
                    in_channels=1024,
                    num_attention_blocks=16)),
            bbox_roi_extractor=dict(
                type='mmtrack.SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=2),
                out_channels=512,
                featmap_strides=[16])
        ),
        test_cfg=dict(
            rcnn=dict(
                score_thr=0.0,
                max_per_img=1
            )
        )
    )
)

ann_path = 'annotations/video/hcc/ceus_video_hcc_fold0_'

data_root = 'data/CEUS/'
train_file = ann_path + 'train.json'
val_file = ann_path + 'val.json'
test_file = ann_path + 'test.json'

train_dataloader = dict(
    dataset=dict(
        metainfo=dict(classes=('HCC',)),
        ann_file=train_file))
val_dataloader = dict(
    dataset=dict(
        metainfo=dict(classes=('HCC',)),
        ann_file=val_file))
test_dataloader = dict(
    dataset=dict(
        metainfo=dict(classes=('HCC',)),
        ann_file=test_file))

val_evaluator = dict(
    type='CocoVideoMetric',
    ann_file=data_root + val_file,
    metric='bbox',
    format_only=False)
test_evaluator = dict(
    type='CocoVideoMetric',
    ann_file=data_root + test_file,
    metric='bbox',
    format_only=False)

# training schedule
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

# seed
randomness = dict(seed=2025)

# checkpoint
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=999999,
        save_best='coco/bbox_mAP_50',
        rule='greater',
        max_keep_ckpts=1,
        save_last=True
    )
)

# visualizer
visualizer = dict(type='mmdet.OverlayLocalVisualizer')
