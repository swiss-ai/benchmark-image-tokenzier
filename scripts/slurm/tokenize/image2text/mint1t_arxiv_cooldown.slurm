#!/bin/bash
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5
#SBATCH --job-name=mint1t-arxiv-cooldown-tokenize
#SBATCH --environment=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/scripts/envs/nemo_25_11.toml
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/mint1t_arxiv_cooldown_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/mint1t_arxiv_cooldown_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --partition=normal
#SBATCH --exclusive

set -e

export NUM_GPUS=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))

echo "=========================================="
echo "MINT-1T-ArXiv Cooldown Image2Text Tokenization"
echo "  Nodes: ${SLURM_NNODES}   GPUs: ${NUM_GPUS}"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "  Projected wall (at ~21 k tok/s/rank): ~1.6 h on 128 GPUs"
echo "=========================================="

mkdir -p /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs

srun -u --kill-on-bad-exit=0 bash -c "
    set -e
    uv pip install --python /opt/venv/bin/python3 orjson

    export PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/apertus/Megatron-LM:/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier:\$PYTHONPATH

    cd /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier

    python -m vision_tokenization.tokenize \
        mode=image2text dataset=image2text/mint1t_arxiv_cooldown \
        num_gpus=${NUM_GPUS} \
        wandb.enabled=true \
        resume=true
"

echo ""
echo "=========================================="
echo "Tokenization completed!"
echo "=========================================="
