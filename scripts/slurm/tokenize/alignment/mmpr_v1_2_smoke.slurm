#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=mmpr-smoke
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/mmpr_v1_2_smoke_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/mmpr_v1_2_smoke_%j.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --partition=debug
#SBATCH --exclusive
#SBATCH --no-requeue

# Multi-rank alignment SMOKE: 4-rank ENCODE of a pre-scanned 1-part subset, then
# inline MERGE + check_store — validates the scan -> encode -> merge wiring end
# to end (disjoint media across 4 ranks, cross-rank offset assembly).
#
# Prereq (already run inline in the pa-24 venv):
#   python -m vision_tokenization.alignment_scan mode=alignment dataset=alignment/mmpr_v1_2 \
#       dataset.input_pattern=<.../part-00000.parquet> \
#       dataset.output_dir=/iopsstor/scratch/cscs/xyixuan/apertus/mmpr_smoke_test \
#       dataset.output_name=mmpr_smoke
#
# Submit from a clean env:
#   env -i HOME="$HOME" USER="$USER" /usr/bin/sbatch \
#       /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/scripts/slurm/tokenize/alignment/mmpr_v1_2_smoke.slurm

set -e
export LD_LIBRARY_PATH=/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse:${LD_LIBRARY_PATH:-}
SMOKE=/iopsstor/scratch/cscs/xyixuan/apertus/mmpr_smoke_test
STORE=$SMOKE/alignment/mmpr_smoke
export NUM_GPUS=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))
mkdir -p /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs

echo "=== ${NUM_GPUS}-rank ENCODE (reads pre-built scan, spills disjoint media) ==="
srun -u --mpi=pmix --environment=nemo_26_02 bash -c '
    set -e
    unset LD_PRELOAD || true
    export LD_LIBRARY_PATH=/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse:${LD_LIBRARY_PATH:-}
    python -c "import orjson" 2>/dev/null || uv pip install --python "$(command -v python)" --break-system-packages orjson
    TOKENIZERS_0222=/iopsstor/scratch/cscs/xyixuan/.cache/uv/archive-v0/1CtSrhkL_p104lhucBDaz
    export PYTHONPATH=${TOKENIZERS_0222}:/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier:${PYTHONPATH:-}
    cd /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier
    python -m vision_tokenization.tokenize \
        mode=alignment dataset=alignment/mmpr_v1_2 num_gpus='"${NUM_GPUS}"' \
        dataset.output_dir='"$SMOKE"' dataset.output_name=mmpr_smoke \
        wandb.enabled=false
'

echo "=== inline MERGE + check_store (single task) ==="
srun -u --mpi=pmix --ntasks=1 --environment=nemo_26_02 bash -c '
    set -e
    unset LD_PRELOAD || true
    python -c "import orjson" 2>/dev/null || uv pip install --python "$(command -v python)" --break-system-packages orjson
    export PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier:${PYTHONPATH:-}
    cd /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier
    python -m vision_tokenization.pipeline.output.alignment_merge '"$STORE"' --keep-intermediates
    python vision_tokenization/tests/alignment/check_store.py '"$STORE"'
'
echo "SMOKE DONE: $STORE"
