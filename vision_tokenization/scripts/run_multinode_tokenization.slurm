#!/bin/bash
#SBATCH --account=a-infra01
#SBATCH --job-name=multinode-tok
#SBATCH --environment=emu3
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=normal
#SBATCH --reservation=PA-2338-RL
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=285
#SBATCH --time=12:00:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/benchmark-image-tokenizer/vision_tokenization/logs/multinode_tok_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/benchmark-image-tokenizer/vision_tokenization/logs/multinode_tok_%j.err
#SBATCH --dependency=afterany:1513677

###################### Multi-Node Tokenization Wrapper ######################
# This script wraps existing single-node tokenization scripts to run on
# multiple nodes using Ray for distributed processing.
#
# Usage:
#   TOKENIZATION_MODE=image_only sbatch run_multinode_tokenization.slurm
#   TOKENIZATION_MODE=instruct sbatch run_multinode_tokenization.slurm --debug
#   TOKENIZATION_MODE=paired sbatch run_multinode_tokenization.slurm --resume
#   TOKENIZATION_MODE=paired sbatch run_multinode_tokenization.slurm --offline
#   TOKENIZATION_MODE=image_only sbatch run_multinode_tokenization.slurm --config /path/to/config.json
#
# Environment Variables:
#   TOKENIZATION_MODE - Required: image_only, instruct, or paired
#
# Optional Arguments (passed to wrapped script):
#   --config <path>  - Path to config JSON file (overrides default in wrapped script)
#   --debug          - Run in debug mode (small dataset)
#   --resume         - Resume from checkpoint
#   --offline        - Enable HF datasets offline mode (only use cached datasets)
#############################################################################

# Get the project root directory
export PROJECT_ROOT="/iopsstor/scratch/cscs/${USER}/benchmark-image-tokenizer"
export SCRIPT_DIR="${PROJECT_ROOT}/vision_tokenization/scripts"
# Hub cache location
export HF_HUB_CACHE="${HF_HUB_CACHE:-/capstor/store/cscs/swissai/infra01/vision-datasets/hf_hub_cache}"


# Configuration
export SLURM_JOB_NUM_NODES="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}"
export GPUS_PER_NODE=4
export TOTAL_GPUS=$((SLURM_JOB_NUM_NODES * GPUS_PER_NODE))

# Ray cluster configuration
export MASTER_NODE=$(hostname)
export MASTER_NODE_IP=$(hostname -i)
export RAY_PORT=6379
export RAY_ADDRESS="${MASTER_NODE_IP}:${RAY_PORT}"
export REDIS_PASSWORD="MULTINODE_TOKEN_$(date +%s)"

# Validate TOKENIZATION_MODE
if [ -z "${TOKENIZATION_MODE}" ]; then
    echo "ERROR: TOKENIZATION_MODE environment variable is required"
    echo "Usage: TOKENIZATION_MODE=image_only|instruct|paired sbatch run_multinode_tokenization.slurm"
    exit 1
fi

# Select wrapped script based on mode
case ${TOKENIZATION_MODE} in
    image_only)
        WRAPPED_SCRIPT="run_image_only_tokenization.slurm"
        ;;
    instruct)
        WRAPPED_SCRIPT="run_instruct_tokenization.slurm"
        ;;
    paired)
        WRAPPED_SCRIPT="run_paired_tokenization.slurm"
        ;;
    *)
        echo "ERROR: Invalid TOKENIZATION_MODE: ${TOKENIZATION_MODE}"
        echo "Must be one of: image_only, instruct, paired"
        exit 1
        ;;
esac

export ARGS="$@"

# Check for --offline and --config flags
OFFLINE_MODE=false
CONFIG_OVERRIDE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_OVERRIDE="$2"
            shift 2
            ;;
        --offline)
            OFFLINE_MODE=true
            export HF_DATASETS_OFFLINE=1
            shift
            ;;
        *)
            shift
            ;;
    esac
done

echo "=================================================================================="
echo "Multi-Node Tokenization - Ray Distributed Processing"
echo "=================================================================================="
echo "Mode:         ${TOKENIZATION_MODE}"
echo "Script:       ${WRAPPED_SCRIPT}"
echo "Nodes:        ${SLURM_JOB_NUM_NODES}"
echo "GPUs/node:    ${GPUS_PER_NODE}"
echo "Total GPUs:   ${TOTAL_GPUS}"
echo "Master node:  ${MASTER_NODE} (${MASTER_NODE_IP})"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Config:       ${CONFIG_OVERRIDE:-<default from wrapped script>}"
echo "Offline mode: ${OFFLINE_MODE}"
echo "Arguments:    ${ARGS}"
echo "=================================================================================="

if [ "$OFFLINE_MODE" = true ]; then
    echo "OFFLINE mode enabled - HF_DATASETS_OFFLINE=1 exported to all nodes"
fi

# Execute on all nodes
srun -N ${SLURM_JOB_NUM_NODES} --tasks-per-node=1 --cpus-per-task=${SLURM_CPUS_PER_TASK} -u --export=ALL bash -c '
    # Python path setup and critical variables
    export PYTHONPATH='"${PROJECT_ROOT}"':$PYTHONPATH
    pip install -q omegaconf fastkmeans
    export HF_DATASETS_OFFLINE
    export HF_HUB_CACHE

    export RAY_ADDRESS='"${MASTER_NODE_IP}"':'"${RAY_PORT}"'
    export PROJECT_ROOT='"${PROJECT_ROOT}"'
    export SCRIPT_DIR='"${SCRIPT_DIR}"'
    export WRAPPED_SCRIPT='"${WRAPPED_SCRIPT}"'
    # ARGS is already exported via --export=ALL and contains space-separated arguments
    # Don'\''t re-export it to avoid word-splitting issues with flags like --resume
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
        # Start Ray head with redis password
        ray start --head \
            --node-ip-address=$MASTER_NODE_IP \
            --port=$RAY_PORT \
            --num-cpus=${SLURM_CPUS_PER_TASK} \
            --num-gpus='"${GPUS_PER_NODE}"' \
            --redis-password=$REDIS_PASSWORD

        # Wait for workers to connect
        echo "[Master] Waiting 30 seconds for workers to join..."
        sleep 30  # Workers sleep 15s, then need time to connect

        echo "[Master] Proceeding with tokenization (workers should be connected)"

        # Show cluster resources
        echo "=================================================================================="
        echo "[Master] Ray Cluster Resources:"
        ray status
        echo "=================================================================================="

        # Export NUM_GPUS for the wrapped script to use
        export NUM_GPUS='"${TOTAL_GPUS}"'

        # Execute the wrapped script with all arguments passed through
        echo "[Master] Executing wrapped script: '"${WRAPPED_SCRIPT}"'"
        echo "[Master] With arguments: $ARGS"
        bash '"${SCRIPT_DIR}"'/'"${WRAPPED_SCRIPT}"' $ARGS

        EXITCODE=$?

        # Shutdown Ray
        echo "[Master] Shutting down Ray cluster..."
        ray stop

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