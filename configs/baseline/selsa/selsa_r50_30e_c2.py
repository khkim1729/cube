_base_ = [
    '../../_base_/models/faster-rcnn_r50-dc5.py',
    '../../_base_/datasets/ceus_vid_selsa_c2_style.py',
    '../../_base_/default_runtime.py'
]
model = dict(
    type='SELSA',
    detector=dict(
        roi_head=dict(
            type='mmtrack.SelsaRoIHead',
            bbox_head=dict(
                type='mmtrack.SelsaBBoxHead',
                in_channels=512,
                num_shared_fcs=2,
                num_classes=2,
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
            rpn=dict(
                nms_pre=1000,
                max_per_img=300,
                nms=dict(type='nms', iou_threshold=0.5)
            ),
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

# vis_backends: where to save
vis_backends = [dict(type='LocalVisBackend')]

# visualizer: how to draw
visualizer = dict(
    type='DetLocalVisualizerOverlay',
    name='visualizer',
    vis_backends=vis_backends,
    # save_dir=None
)

# custom_hooks: when to draw
custom_hooks = [
    dict(
        type='TrackVisualizationHook',
        draw=False,
        interval=1,
        score_thr=0.0,
        show=False,        # 창 띄우지 않고 파일 저장
        test_out_dir='vis'
    )
]
