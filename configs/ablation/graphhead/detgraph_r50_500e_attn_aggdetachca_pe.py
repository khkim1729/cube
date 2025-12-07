_base_ = [
    '../../_base_/models/faster-rcnn_r50-dc5.py',
    '../../_base_/datasets/ceus_vid_detgraph_style.py',
    '../../_base_/detgraph_runtime.py'
]
model = dict(
    type='mmtrack.DetGraph',
    detector=dict(
        roi_head=dict(
            type='mmtrack.DetGraphRoIHead',
            bbox_head=dict(
                type='mmtrack.DetGraphBBoxHead',
                in_channels=512,
                num_shared_fcs=2,
                num_classes=1,   # lesion foreground 1개

                # ★ 여기서 phase embedding on/off 및 세부 설정
                use_phase_embed=True,      # ← 여기서 켜고 끄면 됨
                num_phases=3,              # AP, PP/LP, KP
                phase_embed_dim=32,        # embedding 차원
                phase_fusion_mode='concat',  # 'concat' 또는 'add'
                unknown_phase_id=-1,       # dataset에서 invalid phase id

                # aggregator 설정
                aggregator=dict(
                    type='mmtrack.DetGraphAggregatorDetachCA',
                    in_channels=1024,  # shared FC output dim (fc_out_channels)
                    num_attention_blocks=16
                )
                # aggregator=None
            ),
            bbox_roi_extractor=dict(
                type='mmtrack.SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=2),
                out_channels=512,
                featmap_strides=[16])
        ),
        # ★ test 시 RPN + RCNN 둘 다 명시 (RPN proposal 300개, RCNN 최종 box 1개)
        test_cfg=dict(
            rpn=dict(
                nms_pre=1000,
                max_per_img=300,
                nms=dict(type='nms', iou_threshold=0.5)
            ),
            rcnn=dict(
                score_thr=1e-4,
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=1   # 최종 detection 1개
            )
        )
    ),

    # graph_head
    graph_head=None,          # 그래프 헤드 사용 안 함
    graph_loss_weight=0.0
)

# training schedule: 500 epochs, val every 10
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=500, val_interval=10)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# LR schedule
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        by_epoch=False,
        begin=0,
        end=500),  # warmup 그대로 유지
    dict(
        type='MultiStepLR',
        begin=0,
        end=500,
        by_epoch=True,
        milestones=[120, 200, 260, 280],
        gamma=0.2)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2),
    accumulative_counts=8
)

# vis
vis_backends = [dict(type='LocalVisBackend')]

visualizer = dict(
    type='DetGraphLocalVisualizerOverlay',
    name='visualizer',
    vis_backends=vis_backends,
    # save_dir=None
)

custom_hooks = [
    dict(
        type='DetGraphVisualizationHook',
        draw=True,
        interval=1,
        score_thr=0.0,
        show=False,
        test_out_dir='vis'
    )
]