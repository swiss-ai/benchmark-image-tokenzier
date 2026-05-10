#!/bin/bash
# Merge latex-formulas-80M-inline pass into Apertus1p5_cooldown_tokenized
# after job 2063432 finishes.  Submit with --dependency=afterany:2063432.
# Single-bucket (no stage2/lct split for image2text mode).
#
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5
#SBATCH --job-name=merge-latex80m-inline
#SBATCH --environment=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/scripts/envs/nemo_25_11.toml
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/merge_latex_inline_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/merge_latex_inline_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --partition=normal

set -euo pipefail

SRC=/capstor/store/cscs/swissai/infra01/vision-datasets/tokenized/image2text/latex_formulas_80m_inline
DEST=/capstor/store/cscs/swissai/infra01/vision-datasets/Apertus1p5_cooldown_tokenized
OUT=img2txt_latex_formulas_80m_inline

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
