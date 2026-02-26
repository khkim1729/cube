EMBS=(8)
EXPS=(fnh hcc c1)

for emb in "${EMBS[@]}"; do
  for fold in {0..4}; do
    for exp in "${EXPS[@]}"; do
      WORKDIR="results/selsa_ceus_film_neck/emb${emb}/film_neck-12e_${exp}_fold${fold}"
      CONFIG="configs/vid/selsa_ceus_film_neck_tmp/film_neck-12e_${exp}_fold${fold}_emb${emb}.py"

      CKPT="$(ls -1 ${WORKDIR}/best*.pth 2>/dev/null | sort -V | head -n 1)"

      echo "========================================="
      echo "emb=${emb} | exp=${exp} | fold=${fold}"
      echo "WORKDIR: $WORKDIR"
      echo "CONFIG : $CONFIG"
      echo "CKPT   : $CKPT"
      echo "========================================="

      if [ ! -f "$CONFIG" ]; then
        echo "⚠ Config not found: $CONFIG"
        continue
      fi

      if [ -f "$CKPT" ]; then
        python tools/test.py \
          "$CONFIG" \
          --checkpoint "$CKPT" \
          --work-dir "$WORKDIR" \
          | tee "${WORKDIR}/test_log.txt"
      else
        echo "⚠ No checkpoint found for ${exp} fold ${fold} (emb=${emb})"
      fi

    done
  done
done