#!/bin/bash
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --job-name=chartnet-realworld-tokenize
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

# --environment is on the `srun` line (NOT #SBATCH), so this batch body runs on the
# BARE HOST. srun loads its switch/hpe_slingshot plugin host-side at launch, and that
# plugin needs libjson-c.so.5 which is off the host default path — the wheelhouse
# provides it. (CSCS discourages --environment as #SBATCH; on-srun is the canonical form.)
export LD_LIBRARY_PATH=/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse:${LD_LIBRARY_PATH:-}

export NUM_GPUS=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))
echo "chartnet_realworldchart: ${SLURM_NNODES} node(s), ${NUM_GPUS} GPUs"

mkdir -p /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs

srun --environment=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/scripts/envs/nemo_25_11.toml -u bash -c "
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
