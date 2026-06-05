#!/bin/bash
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --job-name=merge
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/merge_%x_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/merge_%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --exclusive
#SBATCH --no-requeue
#
# Generic cross-rank merge: combine per-rank shards in SRC into OUT.{bin,idx},
# optionally placing them under the canonical DEST dir.
#
# Most merges (<200 GB) run inline on the head node (export PYTHONPATH, run
# merge.py); use this only for ~500 GB+ jobs (nemotron-v3 scale). As a tokenize
# dependency (afterany, so it still runs if tokenize hit a benign SIGTERM at the
# time limit; the merge fails cleanly on incomplete shards):
#
#   JOBID=$(sbatch --parsable scripts/slurm/tokenize/image2text/docci.slurm)
#   SRC=/.../tokenized/image2text/docci RANKS=4 \
#       sbatch --dependency=afterany:$JOBID scripts/slurm/ops/merge.slurm
#
# Required env:
#   SRC              rank shard directory (a stage2/ or lct/ subdir)
# Optional env:
#   OUT              output prefix name (default: merged)
#   RANKS            wait for this many rank checkpoints before merging (rank gate)
#   DEST             move OUT.{bin,idx} (+ sidecars) here after merge
#                    (.../Apertus1p5_sft_tokenized or .../Apertus1p5_sft_lct_tokenized)
#   STRIP_THINKING   "true" to also emit OUT_no_cot.{bin,idx} (<think>..</think> removed);
#                    default "false" — SFT merges preserve CoT
#
set -euo pipefail

: "${SRC:?SRC is required}"
OUT="${OUT:-merged}"
STRIP_THINKING="${STRIP_THINKING:-false}"

export LD_LIBRARY_PATH=/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/apertus/Megatron-LM:/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier:${PYTHONPATH:-}
cd /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier

echo "=== merge $SRC -> $OUT  (ranks=${RANKS:-auto}, strip=$STRIP_THINKING, dest=${DEST:-in-place})  on $(hostname) ==="
[ -d "$SRC" ] || { echo "missing SRC: $SRC"; exit 1; }

FLAGS=(--output-name "$OUT")
[ -n "${RANKS:-}" ] && FLAGS+=(--expected-ranks "$RANKS")
[ "$STRIP_THINKING" = "true" ] && FLAGS+=(--strip-thinking --no-cot-output-name "${OUT}_no_cot")

srun -u python -m vision_tokenization.pipeline.output.merge "$SRC" "${FLAGS[@]}"

[ -f "$SRC/$OUT.bin" ] && [ -f "$SRC/$OUT.idx" ] || { echo "✗ merge did not produce $OUT.{bin,idx}"; exit 1; }

if [ -n "${DEST:-}" ]; then
    mkdir -p "$DEST"
    for f in "$SRC/$OUT".bin "$SRC/$OUT".idx "$SRC/$OUT"*.json \
             "$SRC/${OUT}_no_cot".bin "$SRC/${OUT}_no_cot".idx; do
        [ -e "$f" ] && mv "$f" "$DEST/"
    done
    echo "=== placed under $DEST ==="
fi
echo "=== done: ${DEST:-$SRC}/$OUT.{bin,idx} ==="
