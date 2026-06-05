#!/bin/bash
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --job-name=mllm-dpo-tokenize
#SBATCH --output=/iopsstor/scratch/cscs/rkreft/mllm-dpo/logs/mllm_dpo_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/rkreft/mllm-dpo/logs/mllm_dpo_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --partition=normal
#SBATCH --exclusive
#
# Tokenize one variant of the mllm-dpo pairs with provenance enabled.
#
#   VARIANT=accepted sbatch scripts/slurm/mllm_dpo.slurm
#   VARIANT=rejected sbatch scripts/slurm/mllm_dpo.slurm
#
# ──────────────────────────────────────────────────────────────────────────────
# EDIT THESE to match your cluster checkout (the #SBATCH --environment above
# must also point at a valid env TOML on a compute-visible filesystem):
REPO=/users/rkreft/apertus/benchmark-image-tokenizer
MEGATRON=/iopsstor/scratch/cscs/xyixuan/apertus/Megatron-LM   # <- confirm your Megatron-LM path
ENV_TOML=${REPO}/scripts/envs/nemo_25_11.toml                 # container; --environment goes on srun
# ──────────────────────────────────────────────────────────────────────────────

set -e
: "${VARIANT:?Set VARIANT=accepted or VARIANT=rejected}"

export NUM_GPUS=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))
echo "=========================================="
echo "mllm-dpo tokenize — variant=${VARIANT}"
echo "Job ID: ${SLURM_JOB_ID}   GPUs: ${NUM_GPUS}"
echo "=========================================="

mkdir -p /iopsstor/scratch/cscs/rkreft/mllm-dpo/logs

srun -u --environment=${ENV_TOML} --kill-on-bad-exit=0 bash -c "
    set -e
    uv pip install --python /opt/venv/bin/python3 orjson
    export PYTHONPATH=${MEGATRON}:${REPO}:\$PYTHONPATH
    cd ${REPO}

    python -m vision_tokenization.tokenize \
        mode=sft dataset=sft/mllm_dpo_${VARIANT} \
        num_gpus=${NUM_GPUS} \
        emit_provenance=true \
        wandb.enabled=false \
        resume=true
"

echo ""
echo "=========================================="
echo "Tokenization (${VARIANT}) completed!"
echo "Shards: /iopsstor/scratch/cscs/rkreft/mllm-dpo/tokenized/sft/mllm_dpo_${VARIANT}/"
echo "=========================================="
