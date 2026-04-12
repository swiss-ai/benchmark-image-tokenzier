#!/bin/bash
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5-5
#SBATCH --job-name=vdr-cook-sft
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/vdr_cooking_recipes_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/vdr_cooking_recipes_%j.err
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --partition=normal
#SBATCH --exclusive

set -e

export NUM_GPUS=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))

echo "=========================================="
echo "VDR Cooking Recipes SFT Tokenization"
echo "  GPUs: ${NUM_GPUS}, Nodes: ${SLURM_NNODES}"
echo "=========================================="

mkdir -p /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs

srun -u --kill-on-bad-exit=0 bash -c "
    set -e
    uv pip install --python /opt/venv/bin/python3 orjson

    export PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier:\$PYTHONPATH

    cd /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier

    python -m vision_tokenization.tokenize \
        mode=sft dataset=sft/vdr_cooking_recipes \
        num_gpus=${NUM_GPUS} \
        wandb.enabled=true \
        resume=true
"
