#!/bin/bash
#SBATCH --account=a-infra01
#SBATCH --job-name=subset-tok
#SBATCH --environment=emu3
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=normal
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=285
#SBATCH --time=12:00:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/benchmark-image-tokenizer/vision_tokenization/logs/subset_tok_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/benchmark-image-tokenizer/vision_tokenization/logs/subset_tok_%j.err

###################### Single-Subset Tokenization (Multi-Node Ray) ######################
# Tokenizes a single dataset config/subset using Ray distributed processing.
# Designed to be called by run_multi_subset_tokenization.sh, but can also be
# submitted directly via sbatch.
#
# Required Environment Variables:
#   CONFIG_PATH            - Path to base JSON config file
#
# Optional Environment Variables:
#   CONFIG_NAME            - Dataset config/subset name override
#   NUM_SHARDS_OVERRIDE    - Override num_shards from config
#   RESUME_MODE            - "true" to resume from checkpoint
#   OFFLINE_MODE           - "true" for HF offline mode
#
# Direct Usage:
#   CONFIG_PATH=/path/to/config.json CONFIG_NAME=my_subset sbatch run_subset_tokenization.slurm
#   CONFIG_PATH=/path/to/config.json sbatch --nodes=4 run_subset_tokenization.slurm
#
# SBATCH directives (--nodes, --time, etc.) can be overridden on the sbatch command line.
##########################################################################################

set -euo pipefail

# Validate required environment
if [ -z "${CONFIG_PATH:-}" ]; then
    echo "ERROR: CONFIG_PATH environment variable is required"
    echo "Usage: CONFIG_PATH=/path/to/config.json sbatch run_subset_tokenization.slurm"
    exit 1
fi

if [ ! -f "${CONFIG_PATH}" ]; then
    echo "ERROR: Config file not found: ${CONFIG_PATH}"
    exit 1
fi

# Project root
export PROJECT_ROOT="${PROJECT_ROOT:-/iopsstor/scratch/cscs/${USER}/benchmark-image-tokenizer}"

# Hub cache location
export HF_HUB_CACHE="${HF_HUB_CACHE:-/capstor/store/cscs/swissai/infra01/vision-datasets/hf_hub_cache}"

# Read optional overrides (with defaults)
CONFIG_NAME="${CONFIG_NAME:-}"
NUM_SHARDS_OVERRIDE="${NUM_SHARDS_OVERRIDE:-}"
RESUME_MODE="${RESUME_MODE:-false}"
OFFLINE_MODE="${OFFLINE_MODE:-false}"

# Cluster configuration
export SLURM_JOB_NUM_NODES="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}"
export GPUS_PER_NODE=4
export TOTAL_GPUS=$((SLURM_JOB_NUM_NODES * GPUS_PER_NODE))

# Ray cluster configuration
export MASTER_NODE=$(hostname)
export MASTER_NODE_IP=$(hostname -i)
export RAY_PORT=6379
export RAY_ADDRESS="${MASTER_NODE_IP}:${RAY_PORT}"
export REDIS_PASSWORD="SUBSET_TOKEN_$(date +%s)"

echo "=================================================================================="
echo "Subset Tokenization - Ray Distributed Processing"
echo "=================================================================================="
echo "Config:        ${CONFIG_PATH}"
echo "Config name:   ${CONFIG_NAME:-<default>}"
echo "Num shards:    ${NUM_SHARDS_OVERRIDE:-<from config>}"
echo "Nodes:         ${SLURM_JOB_NUM_NODES}"
echo "GPUs/node:     ${GPUS_PER_NODE}"
echo "Total GPUs:    ${TOTAL_GPUS}"
echo "Master node:   ${MASTER_NODE} (${MASTER_NODE_IP})"
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Resume:        ${RESUME_MODE}"
echo "Offline:       ${OFFLINE_MODE}"
echo "=================================================================================="

# Build CLI override flags
OVERRIDE_FLAGS=""

if [ -n "${CONFIG_NAME}" ]; then
    OVERRIDE_FLAGS="${OVERRIDE_FLAGS} --config-name ${CONFIG_NAME}"
fi

if [ -n "${NUM_SHARDS_OVERRIDE}" ]; then
    OVERRIDE_FLAGS="${OVERRIDE_FLAGS} --num-shards ${NUM_SHARDS_OVERRIDE}"
fi

RESUME_FLAG=""
if [ "${RESUME_MODE}" = "true" ]; then
    RESUME_FLAG="--resume"
    echo "RESUME mode enabled - will skip completed shards"
fi

if [ "${OFFLINE_MODE}" = "true" ]; then
    export HF_DATASETS_OFFLINE=1
    echo "OFFLINE mode enabled - HF_DATASETS_OFFLINE=1"
fi

echo "Override flags: ${OVERRIDE_FLAGS:-<none>}"
echo "=================================================================================="

# Execute on all nodes via srun (handles both single-node and multi-node)
srun -N ${SLURM_JOB_NUM_NODES} --tasks-per-node=1 --cpus-per-task=${SLURM_CPUS_PER_TASK} -u --export=ALL bash -c '
    # Environment setup
    export PYTHONPATH='"${PROJECT_ROOT}"':$PYTHONPATH
    pip install -q omegaconf fastkmeans
    export HF_DATASETS_OFFLINE
    export HF_HUB_CACHE

    export RAY_ADDRESS='"${MASTER_NODE_IP}"':'"${RAY_PORT}"'
    export GPUS_PER_NODE='"${GPUS_PER_NODE}"'
    export TOTAL_GPUS='"${TOTAL_GPUS}"'
    export REDIS_PASSWORD='"${REDIS_PASSWORD}"'
    export MASTER_NODE_IP='"${MASTER_NODE_IP}"'
    export RAY_PORT='"${RAY_PORT}"'

    if [[ $SLURM_PROCID = 0 ]]; then
        echo "=================================================================================="
        echo "[Master] Starting Ray head on ${MASTER_NODE_IP}:${RAY_PORT}"
        echo "=================================================================================="
        ray enable-usage-stats
        ray start --head \
            --node-ip-address=$MASTER_NODE_IP \
            --port=$RAY_PORT \
            --num-cpus=${SLURM_CPUS_PER_TASK} \
            --num-gpus='"${GPUS_PER_NODE}"' \
            --redis-password=$REDIS_PASSWORD

        # Wait for workers to connect (only needed for multi-node)
        if [ '"${SLURM_JOB_NUM_NODES}"' -gt 1 ]; then
            echo "[Master] Waiting 30 seconds for workers to join..."
            sleep 30
            echo "[Master] Ray Cluster Resources:"
            ray status
        fi

        echo "[Master] Starting tokenization..."
        echo "=================================================================================="

        # Run tokenization with overrides
        python -m vision_tokenization.tokenize \
            --num-gpus '"${TOTAL_GPUS}"' \
            hf \
            --config '"${CONFIG_PATH}"' \
            '"${OVERRIDE_FLAGS}"' \
            '"${RESUME_FLAG}"'

        EXITCODE=$?

        # Shutdown Ray
        echo "[Master] Shutting down Ray cluster..."
        ray stop

        if [ $EXITCODE -eq 0 ]; then
            echo "=================================================================================="
            echo "TOKENIZATION COMPLETED SUCCESSFULLY"
            echo "=================================================================================="
        else
            echo "=================================================================================="
            echo "TOKENIZATION FAILED WITH EXIT CODE: $EXITCODE"
            echo "=================================================================================="
        fi

        exit $EXITCODE

    else
        # Worker nodes
        sleep 15  # Give master time to start

        NODE_IP=$(hostname -i)
        echo "[Worker-${SLURM_PROCID}] Starting Ray worker on ${NODE_IP}"

        ray start --address="${RAY_ADDRESS}" \
                  --num-cpus=${SLURM_CPUS_PER_TASK} \
                  --num-gpus='"${GPUS_PER_NODE}"' \
                  --redis-password=$REDIS_PASSWORD \
                  --block

        echo "[Worker-${SLURM_PROCID}] Ray worker stopped"
    fi
'

echo "=================================================================================="
echo "Job completed at $(date)"
echo "=================================================================================="