# dataset settings
dataset_type = 'CeusC2VIDDataset'
data_root = 'data/CEUS/'

# data pipeline
train_pipeline = [
    dict(
        type='TransformBroadcaster',
        share_random_params=True,
        transforms=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadTrackAnnotations', with_instance_id=True),
            dict(type='mmdet.Resize', scale=(512, 512), keep_ratio=False),
        ]),
    dict(type='PackTrackInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadTrackAnnotations', with_instance_id=True),
    dict(type='mmdet.Resize', scale=(512, 512), keep_ratio=False),
    dict(type='PackTrackInputs',
         pack_single_img=True,
         meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor',  # visualization 결과 저장을 위해 metadata 추가
             'video_id','frame_id','pid','fold','category','phase','is_aug')
         )
]

# dataloader
train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    # CEUS는 고정 크기이므로 AspectRatioBatchSampler 불필요
    dataset=dict(
        type=dataset_type,                  
        data_root=data_root,
        ann_file='annotations/ceus_c2_train.json',
        data_prefix=dict(img_path='Data'),     # file_name이 fold_k/... 시작
        filter_cfg=dict(filter_empty_gt=False, min_size=1),  # blank 프레임 유지
        pipeline=train_pipeline,
        load_as_video=True,
        key_img_sampler=dict(interval=1),       # CEUS: 모든 프레임을 key로 사용
        ref_img_sampler=dict(
            num_ref_imgs=1,
            frame_range=[-1, 0],
            filter_key_img=True,
            method='uniform'
        ),
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='VideoSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/ceus_c2_val.json',
        data_prefix=dict(img_path='Data'),
        pipeline=test_pipeline,
        load_as_video=True,
        ref_img_sampler=None,
        test_mode=True,
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='VideoSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/ceus_c2_test.json',
        data_prefix=dict(img_path='Data'),
        pipeline=test_pipeline,
        load_as_video=True,
        ref_img_sampler=None,
        test_mode=True,
    ),
)

# evaluator
val_evaluator = dict(
    type='CocoVideoMetric',
    ann_file=data_root + 'annotations/ceus_c2_val.json',
    metric='bbox')

test_evaluator = dict(
    type='CocoVideoMetric',
    ann_file=data_root + 'annotations/ceus_c2_test.json',
    metric='bbox'
)
