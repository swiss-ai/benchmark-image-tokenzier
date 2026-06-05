#!/bin/bash
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --job-name=subsample
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/subsample_%x_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/subsample_%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --exclusive
#SBATCH --no-requeue
# Subsample a pool to RATIO of its tokens into DEST, writing only the kept subset (no discard).
# Pass via --export=ALL,SRC=,DEST=,RATIO=[,WORKERS=]
set -e
: "${SRC:?}"; : "${DEST:?}"; : "${RATIO:?}"
export LD_LIBRARY_PATH=/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/apertus/Megatron-LM:/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier
SAMPLER=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/scripts/subsample.py
echo "=== subsample $SRC -> $DEST  ratio=$RATIO  on $(hostname) ==="
srun -u python "$SAMPLER" --src "$SRC" --dest "$DEST" --ratio "$RATIO" --seed 42 --workers "${WORKERS:-9}"
echo "=== done ==="
