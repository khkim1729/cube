#!/usr/bin/env bash
set -e

mkdir -p configs/vid/selsa_ceus_film_neck_tmp

EMBS=(8 16 32 64)
EXPS=(c2)

for emb in "${EMBS[@]}"
do
  for exp in "${EXPS[@]}"
  do
    for fold in {0..4}
    do
      SRC=configs/vid/selsa_ceus_film_neck/film_neck-12e_${exp}_fold${fold}.py
      TMP=configs/vid/selsa_ceus_film_neck_tmp/film_neck-12e_${exp}_fold${fold}_emb${emb}.py
      WORKDIR=results/selsa_ceus_film_neck/emb${emb}/film_neck-12e_${exp}_fold${fold}

      sed "s/^FILM_EMB_DIM *= *.*/FILM_EMB_DIM = ${emb}/" "$SRC" > "$TMP"

      python tools/train.py "$TMP" --work-dir "$WORKDIR"
    done
  done
done