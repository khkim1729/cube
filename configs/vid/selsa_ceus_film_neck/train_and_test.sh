#!/usr/bin/env bash
set -e

GPU=0

mkdir -p configs/vid/selsa_ceus_film_neck_tmp

EMBS=(8 16 32 64)
EXPS=(c2)

for emb in "${EMBS[@]}"; do
  for exp in "${EXPS[@]}"; do

    echo "#########################################"
    echo "START TRAIN: emb=${emb}, exp=${exp}, gpu=${GPU}"
    echo "#########################################"

    # -----------------------------
    # 1) fold 0~4 학습 병렬 실행
    # -----------------------------
    for fold in {0..4}; do
      SRC="configs/vid/selsa_ceus_film_neck/film_neck-12e_${exp}_fold${fold}.py"
      TMP="configs/vid/selsa_ceus_film_neck_tmp/film_neck-12e_${exp}_fold${fold}_emb${emb}.py"
      WORKDIR="results/selsa_ceus_film_neck/emb${emb}/film_neck-12e_${exp}_fold${fold}"

      if [ ! -f "$SRC" ]; then
        echo "⚠ Source config not found: $SRC"
        continue
      fi

      mkdir -p "$WORKDIR"

      sed "s/^FILM_EMB_DIM *= *.*/FILM_EMB_DIM = ${emb}/" "$SRC" > "$TMP"

      echo "========================================="
      echo "[TRAIN] emb=${emb} | exp=${exp} | fold=${fold} | gpu=${GPU}"
      echo "SRC    : $SRC"
      echo "TMP    : $TMP"
      echo "WORKDIR: $WORKDIR"
      echo "========================================="

      CUDA_VISIBLE_DEVICES=$GPU \
      python tools/train.py "$TMP" --work-dir "$WORKDIR" \
        > "${WORKDIR}/train.log" 2>&1 &
    done

    wait

    echo "#########################################"
    echo "ALL TRAIN DONE: emb=${emb}, exp=${exp}"
    echo "START TEST"
    echo "#########################################"

    # -----------------------------
    # 2) fold 0~4 test 순차 실행
    # -----------------------------
    for fold in {0..4}; do
      WORKDIR="results/selsa_ceus_film_neck/emb${emb}/film_neck-12e_${exp}_fold${fold}"
      CONFIG="configs/vid/selsa_ceus_film_neck_tmp/film_neck-12e_${exp}_fold${fold}_emb${emb}.py"

      CKPT="$(ls -1 ${WORKDIR}/best*.pth 2>/dev/null | sort -V | head -n 1)"

      echo "========================================="
      echo "[TEST] emb=${emb} | exp=${exp} | fold=${fold} | gpu=${GPU}"
      echo "WORKDIR: $WORKDIR"
      echo "CONFIG : $CONFIG"
      echo "CKPT   : $CKPT"
      echo "========================================="

      if [ ! -f "$CONFIG" ]; then
        echo "⚠ Config not found: $CONFIG"
        continue
      fi

      if [ -f "$CKPT" ]; then
        CUDA_VISIBLE_DEVICES=$GPU \
        python tools/test.py \
          "$CONFIG" \
          --checkpoint "$CKPT" \
          --work-dir "$WORKDIR" \
          > "${WORKDIR}/test_log.txt" 2>&1
      else
        echo "⚠ No checkpoint found for ${exp} fold ${fold} (emb=${emb})"
      fi
    done

    echo "#########################################"
    echo "DONE TRAIN+TEST: emb=${emb}, exp=${exp}"
    echo "#########################################"

  done
done