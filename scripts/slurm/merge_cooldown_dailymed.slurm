#!/bin/bash
# Merge dailymed-spl into Apertus1p5_cooldown_tokenized after job 2063453
# finishes.  Submit with --dependency=afterany:2063453.  Stage2 only.
#
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5
#SBATCH --job-name=merge-dailymed-spl
#SBATCH --environment=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/scripts/envs/nemo_25_11.toml
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/merge_dailymed_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/merge_dailymed_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --partition=normal

set -euo pipefail

SRC=/capstor/store/cscs/swissai/infra01/vision-datasets/tokenized/interleave/dailymed_spl/stage2
DEST=/capstor/store/cscs/swissai/infra01/vision-datasets/Apertus1p5_cooldown_tokenized
OUT=interleave_dailymed_spl

mkdir -p "$DEST"
export PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/apertus/Megatron-LM:/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier:${PYTHONPATH:-}
cd /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier

NRANKS=$(ls "$SRC"/rank_*_chunk_*.bin 2>/dev/null | sed 's|.*/rank_||; s|_chunk.*||' | sort -u | wc -l)
echo "merging $OUT — $NRANKS ranks at $SRC"

python -m vision_tokenization.pipeline.output.merge \
    "$SRC" --expected-ranks "$NRANKS" --output-name "$OUT"

mv "$SRC/$OUT.bin" "$DEST/$OUT.bin"
mv "$SRC/$OUT.idx" "$DEST/$OUT.idx"
echo "wrote $DEST/$OUT.{bin,idx}"
ls -lh "$DEST/$OUT".*
