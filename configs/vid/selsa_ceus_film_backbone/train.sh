#!/usr/bin/env bash
set -e

mkdir -p configs/vid/selsa_ceus_film_backbone_tmp

EMBS=(8 16 32 64)
EXPS=(fnh hcc c1)

for emb in "${EMBS[@]}"
do
  for exp in "${EXPS[@]}"
  do
    for fold in {0..4}
    do
      # skip already done case
      if [[ "$emb" -eq 8 && "$fold" -eq 0 && "$exp" == "c1" ]]; then
        echo "Skipping emb=8, fold=0, exp=c1 (already done)"
        continue
      fi

      SRC=configs/vid/selsa_ceus_film_backbone/film_backbone-12e_${exp}_fold${fold}.py
      TMP=configs/vid/selsa_ceus_film_backbone_tmp/film_backbone-12e_${exp}_fold${fold}_emb${emb}.py
      WORKDIR=results/selsa_ceus_film_backbone/emb${emb}/film_backbone-12e_${exp}_fold${fold}

      sed "s/^FILM_EMB_DIM *= *.*/FILM_EMB_DIM = ${emb}/" "$SRC" > "$TMP"

      python tools/train.py "$TMP" --work-dir "$WORKDIR"
    done
  done
done