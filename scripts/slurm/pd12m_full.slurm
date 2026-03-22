#!/bin/bash
#SBATCH --account=infra01
#SBATCH --reservation=PA-2338-RL
#SBATCH --job-name=pd12m-tokenize
#SBATCH --environment=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/scripts/envs/nemo_25_11.toml
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/pd12m_full_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/pd12m_full_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=20
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --partition=normal
#SBATCH --exclusive

set -e

echo "=========================================="
echo "PD12M-Full Image2Text Tokenization"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname -s)"
export NUM_GPUS=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))
echo "GPUs: ${NUM_GPUS}"
echo "=========================================="

mkdir -p /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs

srun -u bash -c "
    set -e
    export PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier:\$PYTHONPATH

    cd /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier

    python -m vision_tokenization.tokenize \
        mode=image2text dataset=pd12m_full \
        num_gpus=${NUM_GPUS} \
        wandb.enabled=true \
        resume=true
"

echo ""
echo "=========================================="
echo "Tokenization completed!"
echo "=========================================="
