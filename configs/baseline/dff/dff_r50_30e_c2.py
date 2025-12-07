_base_ = [
    '../../_base_/models/faster-rcnn_r50-dc5.py',
    '../../_base_/datasets/ceus_vid_dff_c2_style.py',
    '../../_base_/default_runtime.py'
]
model = dict(
    type='DFF',
    detector=dict(
        train_cfg=dict(
            rpn_proposal=dict(max_per_img=1000),
            rcnn=dict(sampler=dict(num=512))),
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
    ),
    motion=dict(
        type='FlowNetSimple',
        img_scale_factor=1.0,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmtracking/pretrained_weights/flownet_simple.pth'  # noqa: E501
        )),
    train_cfg=None,
    test_cfg=dict(key_frame_interval=1))

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
