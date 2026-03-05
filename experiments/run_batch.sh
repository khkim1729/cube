#!/bin/bash
# Run N experiments sequentially on a specific GPU.
# Usage: bash run_batch.sh <gpu_id> <combos> <outdir>
# combos: space-separated "baseline:budget" pairs

GPU_ID=$1
COMBOS=$2
OUTDIR=$3
PROJDIR="$(cd "$(dirname "$0")/.." && pwd)"

for combo in $COMBOS; do
    BL="${combo%%:*}"
    BG="${combo##*:}"
    TS=$(date +%Y%m%d_%H%M%S)
    RID="${TS}_${BL}_${BG}_gpu${GPU_ID}"
    echo "[GPU${GPU_ID}] ${BL} x ${BG}  run_id=${RID}"
    cd "$PROJDIR" && python3 experiments/run_pilot.py \
        --baseline "$BL" --budget "$BG" \
        --gpu_id "$GPU_ID" \
        --output_dir "$OUTDIR" \
        --run_id "$RID"
done
