for fold in {0..4}
do
  for exp in c1 c2 fnh hcc
  do
    WORKDIR=results/selsa_ceus/selsa_faster-rcnn_r50-dc5_1xb1-12e_${exp}_fold${fold}
    CONFIG=configs/vid/selsa_ceus/selsa_faster-rcnn_r50-dc5_1xb1-12e_${exp}_fold${fold}.py
  
    CKPT=$(ls ${WORKDIR}/best*.pth 2>/dev/null | sort | head -n 1)

    if [ -f "$CKPT" ]; then
      echo "========================================="
      echo "Testing ${exp} fold ${fold}"
      echo "Checkpoint: $CKPT"
      echo "========================================="

      python tools/test.py \
        $CONFIG \
        --checkpoint $CKPT \
        --work-dir $WORKDIR \
        | tee ${WORKDIR}/test_log.txt
    else
      echo "⚠ No checkpoint found for ${exp} fold ${fold}"
    fi

  done
done