_base_ = ['./detgraph_faster-rcnn_r50-dc5_1xb8-500e_ceusdgvid_attn.py']
model = dict(
    detector=dict(
        backbone=dict(
            depth=101,
            init_cfg=dict(
                type='Pretrained', checkpoint='torchvision://resnet101'))))
