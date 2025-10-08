_base_ = ['./fgfa_faster-rcnn_r50-dc5_1xb1-10e_ceusvid.py']
model = dict(
    detector=dict(
        backbone=dict(
            depth=101,
            init_cfg=dict(
                type='Pretrained', checkpoint='torchvision://resnet101'))))
