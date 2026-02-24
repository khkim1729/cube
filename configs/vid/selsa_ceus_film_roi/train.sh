#!/usr/bin/env bash
set -e

mkdir -p configs/vid/selsa_ceus_film_roi_tmp

EMBS=(8 16 32 64)
EXPS=(c1 c2 fnh hcc)

for emb in "${EMBS[@]}"
do
  for fold in {0..4}
  do
    for exp in "${EXPS[@]}"
    do
      SRC=configs/vid/selsa_ceus_film_roi/film_roi-12e_${exp}_fold${fold}.py
      TMP=configs/vid/selsa_ceus_film_roi_tmp/film_roi-12e_${exp}_fold${fold}_emb${emb}.py
      WORKDIR=results/selsa_ceus_film_roi/emb${emb}/film_roi-12e_${exp}_fold${fold}

      sed "s/^FILM_EMB_DIM *= *.*/FILM_EMB_DIM = ${emb}/" "$SRC" > "$TMP"

      python tools/train.py "$TMP" --work-dir "$WORKDIR"
    done
  done
done