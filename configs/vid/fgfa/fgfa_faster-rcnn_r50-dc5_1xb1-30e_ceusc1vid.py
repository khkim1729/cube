_base_ = [
    '../../_base_/models/faster-rcnn_r50-dc5.py',
    '../../_base_/datasets/ceus_c1_vid_fgfa_style.py',
    '../../_base_/default_runtime.py'
]
model = dict(
    type='FGFA',
    detector=dict(
        test_cfg=dict(
            rcnn=dict(
                score_thr=1e-4,
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=1
            )
        )
    ),
    motion=dict(
        type='FlowNetSimple',
        img_scale_factor=1.0,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmtracking/pretrained_weights/flownet_simple.pth'  # noqa: E501
        )),
    aggregator=dict(
        type='EmbedAggregator', num_convs=1, channels=512, kernel_size=3),
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
    clip_grad=dict(max_norm=35, norm_type=2))

visualizer = dict(type='DetLocalVisualizer')
