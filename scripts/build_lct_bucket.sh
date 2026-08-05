#!/bin/bash
# Bucket one lct dataset into the medium (16k-131k) or long (131k-262k) pool.
# Usage: build_lct_bucket.sh <dataset> <medium|long>
set -euo pipefail
DS="$1"; BAND="$2"
VD=/capstor/store/cscs/swissai/infra01/vision-datasets
case "$BAND" in
  medium) MIN=16385;  MAX=131072; OUT="$VD/Apertus1p5_sft_medium_tokenized";;
  long)   MIN=131073; MAX=262144; OUT="$VD/Apertus1p5_sft_long_tokenized";;
  *) echo "ERROR: band must be medium|long"; exit 1;;
esac
IN="$VD/Apertus1p5_sft_lct_tokenized/$DS"
OUTP="$OUT/$DS"
[ -f "$IN.idx" ] || { echo "ERROR: missing input $IN.idx"; exit 1; }
[ -f "$OUTP.idx" ] && { echo "SKIP $DS/$BAND (output exists)"; exit 0; }
mkdir -p "$OUT"
export LD_LIBRARY_PATH=/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse:/usr/lib64
export PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/apertus/Megatron-LM:/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier
cd /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier
echo ">>> $DS / $BAND  [$MIN,$MAX] -> $OUTP"
python -m vision_tokenization.pipeline.output.bucket_by_length \
  --input "$IN" --output "$OUTP" \
  --min-token "$MIN" --max-token "$MAX" \
  2>&1 | grep -viE "futurewarning|pynvml|warnings.warn"
