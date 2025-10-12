_base_ = [
    '../../_base_/models/faster-rcnn_r50-dc5.py',
    '../../_base_/datasets/ceus_c1_vid_fgfa_style.py',
    '../../_base_/default_runtime.py'
]
model = dict(
    type='SELSA',
    detector=dict(
        roi_head=dict(
            type='mmtrack.SelsaRoIHead',
            bbox_roi_extractor=dict(
                type='mmtrack.TemporalRoIAlign',
                num_most_similar_points=2,
                num_temporal_attention_blocks=4,
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=2),
                out_channels=512,
                featmap_strides=[16]),
            bbox_head=dict(
                type='mmtrack.SelsaBBoxHead',
                num_shared_fcs=3,
                aggregator=dict(
                    type='mmtrack.SelsaAggregator',
                    in_channels=1024,
                    num_attention_blocks=16))
        ),
        test_cfg=dict(
            rcnn=dict(
                score_thr=1e-4,
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=1
            )
        )
    )
)

# training schedule
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=30, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

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
        end=30,
        by_epoch=True,
        milestones=[12, 20, 26, 28],
        gamma=0.2)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2),
    accumulative_counts=8
)

visualizer = dict(type='DetLocalVisualizer')
