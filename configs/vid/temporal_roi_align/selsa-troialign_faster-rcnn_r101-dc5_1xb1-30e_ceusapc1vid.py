_base_ = ['./selsa-troialign_faster-rcnn_r50-dc5_1xb1-30e_ceusapc1vid.py']
model = dict(
    detector=dict(
        backbone=dict(
            depth=101,
            init_cfg=dict(
                type='Pretrained', checkpoint='torchvision://resnet101'))))
