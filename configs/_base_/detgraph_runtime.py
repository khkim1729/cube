default_scope = 'mmtrack'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(  # 최근 5개만 유지
        type='CheckpointHook',
        interval=10,
        max_keep_ckpts=5,
        save_best='coco/bbox_mAP_50',
        rule='greater',
        save_last=True,
        save_optimizer=True
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='TrackVisualizationHook', draw=False),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='TrackLocalVisualizer', vis_backends=vis_backends, name='visualizer')

log_level = 'INFO'
load_from = None
resume = False
