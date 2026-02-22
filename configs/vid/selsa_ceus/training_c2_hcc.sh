for fold in {0..4}
do
  python tools/train.py \
    configs/vid/selsa_ceus/selsa_faster-rcnn_r50-dc5_1xb1-12e_c2_fold${fold}.py \
    --work-dir results/selsa_ceus/selsa_faster-rcnn_r50-dc5_1xb1-12e_c2_fold${fold}
  python tools/train.py \
    configs/vid/selsa_ceus/selsa_faster-rcnn_r50-dc5_1xb1-12e_hcc_fold${fold}.py \
    --work-dir results/selsa_ceus/selsa_faster-rcnn_r50-dc5_1xb1-12e_hcc_fold${fold}
done
