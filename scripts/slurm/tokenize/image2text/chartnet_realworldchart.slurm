#!/bin/bash
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --job-name=chartnet-realworld-tokenize
#SBATCH --environment=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/scripts/envs/nemo_25_11.toml
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/chartnet_realworldchart_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/chartnet_realworldchart_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --partition=normal
#SBATCH --exclusive

# ChartNet RealWorldChart (2077AI) — image2text, ~30K charts -> caption.
# Tiny dataset (2,337 image batches, ~62.5M image tokens) — 1 node is ample;
# finishes in minutes. seqlen_threshold=8192 -> mostly stage2/ with a thin lct/ tail.
set -e

# srun's switch/hpe_slingshot plugin needs libjson-c.so.5, which the --environment
# container lacks. The wheelhouse provides it; put it on LD_LIBRARY_PATH BEFORE srun
# (the batch script runs inside the container, which resets a submit-time LD path).
export LD_LIBRARY_PATH=/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse:${LD_LIBRARY_PATH:-}

export NUM_GPUS=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))
echo "chartnet_realworldchart: ${SLURM_NNODES} node(s), ${NUM_GPUS} GPUs"

mkdir -p /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs

srun -u bash -c "
    set -e
    uv pip install --python /opt/venv/bin/python3 orjson
    export LD_LIBRARY_PATH=/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse:\$LD_LIBRARY_PATH
    export PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/apertus/Megatron-LM:/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier:\$PYTHONPATH
    cd /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier
    python -m vision_tokenization.tokenize \
        mode=image2text dataset=image2text/chartnet_realworldchart \
        num_gpus=${NUM_GPUS} \
        wandb.enabled=true \
        resume=true
"
