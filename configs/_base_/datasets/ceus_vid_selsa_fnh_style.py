# dataset settings
dataset_type = 'CeusFNHVIDDataset'
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
    dict(type='PackTrackInputs', meta_keys=('phase', 'phase_id'),)
]
test_pipeline = [
    dict(
        type='TransformBroadcaster',
        share_random_params=True,
        transforms=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadTrackAnnotations', with_instance_id=True),
            dict(type='mmdet.Resize', scale=(512, 512), keep_ratio=False),
        ]),
    dict(type='PackTrackInputs',
         pack_single_img=False,
         meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor',  # visualization 결과 저장을 위해 metadata 추가
                    'video_id','frame_id','pid','fold','category','phase','is_aug', 'phase_id')
         )
]

# dataloader
train_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,                  
        data_root=data_root,
        ann_file='annotations/myjson.json',
        data_prefix=dict(img_path='Data'),
        filter_cfg=dict(filter_empty_gt=False, min_size=1),
        pipeline=train_pipeline,
        load_as_video=True,
        key_img_sampler=dict(interval=1),
        ref_img_sampler=dict(
            num_ref_imgs=15,
            frame_range=[-15, 15],
            filter_key_img=True,
            method='all_inclusive_valid'
        ),
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='VideoSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/myjson.json',
        data_prefix=dict(img_path='Data'),
        pipeline=test_pipeline,
        load_as_video=True,
        ref_img_sampler=dict(
            num_ref_imgs=15,
            frame_range=[-15, 15],
            filter_key_img=True,
            method='all_inclusive_valid'
        ),
        test_mode=True
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='VideoSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/myjson.json',
        data_prefix=dict(img_path='Data'),
        pipeline=test_pipeline,
        load_as_video=True,
        ref_img_sampler=dict(
            num_ref_imgs=15,
            frame_range=[-15, 15],
            filter_key_img=True,
            method='all_inclusive_valid'
        ),
        test_mode=True
    ),
)
