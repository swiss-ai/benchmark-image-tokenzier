#!/bin/bash
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --job-name=cot-filter
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/cot_filter_%x_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/cot_filter_%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --exclusive
#SBATCH --no-requeue
# Pass via --export=ALL,IN=,OUT=[,TOK=]
set -e
: "${IN:?}"; : "${OUT:?}"
export LD_LIBRARY_PATH=/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/apertus/Megatron-LM:/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier
SC=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/scripts/cot_filter.py
echo "=== cot_filter $IN -> $OUT (token ${TOK:-32}) on $(hostname) ==="
srun -u python "$SC" --input "$IN" --output "$OUT" --token "${TOK:-32}"
echo "=== done ==="
