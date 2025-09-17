#!/bin/bash
#SBATCH --account=a-infra01
#SBATCH --job-name=emu3-img-txt
#SBATCH --environment=emu3
#SBATCH --nodes=15
#SBATCH --exclusive
#SBATCH --partition=normal
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=05:00:00
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/benchmark-image-tokenzier/vision_tokenization/logs/emu3_img_txt_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/benchmark-image-tokenzier/vision_tokenization/logs/emu3_img_txt_%j.err

###################### EMU3 Image-Text Pair Tokenization on Multiple Nodes ######################
# Monitor GPU usage: srun --jobid=<jobid> --overlap -w nid<node number> --pty nvidia-smi

# Essential configuration
export SLURM_JOB_NUM_NODES="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}"
export GPUS_PER_NODE=4

# Input/Output paths - Configure these for your dataset
export INPUT_PATTERN="/capstor/store/cscs/swissai/infra01/vision-datasets/conceptual-12m/data/*.tar"
export TOKENIZER_PATH="/capstor/store/cscs/swissai/infra01/MLLM/llama3_emu3_tokenizer"

# Optional: Range parameter (must be multiple of 4)
export RANGE="0:360" 

# Extract start and end from range
RANGE_START=$(echo ${RANGE} | cut -d':' -f1)
RANGE_END=$(echo ${RANGE} | cut -d':' -f2)
# Calculate actual end index (exclusive to inclusive)
RANGE_END_INCLUSIVE=$((RANGE_END - 1))
export OUTPUT_DIR="/capstor/store/cscs/swissai/infra01/vision-datasets/conceptual-12m/tokenized/cc12m_${RANGE_START}-${RANGE_END_INCLUSIVE}"


# Ray minimal config
export MASTER_NODE=$(hostname)
export MASTER_NODE_IP=$(hostname -i)
export RAY_PORT=6379
export RAY_ADDRESS="${MASTER_NODE_IP}:${RAY_PORT}"
export REDIS_PASSWORD="EMU3IMGTEXT"

echo "=================================================================================="
echo "EMU3 Image-Text Pair Tokenization - Multi-node Processing"
echo "=================================================================================="
echo "Nodes: ${SLURM_JOB_NUM_NODES}"
echo "Total GPUs: $((SLURM_JOB_NUM_NODES * GPUS_PER_NODE))"
echo "Input: ${INPUT_PATTERN}"
echo "Output: ${OUTPUT_DIR}"
echo "Master Node: ${MASTER_NODE} (${MASTER_NODE_IP})"
if [[ -n "${RANGE}" ]]; then
    echo "Range: ${RANGE}"
fi
echo "=================================================================================="

# Count input shards
SHARD_COUNT=$(ls -1 ${INPUT_PATTERN} 2>/dev/null | wc -l)
echo "Found ${SHARD_COUNT} shards to process"

# Execute on all nodes
srun -N ${SLURM_JOB_NUM_NODES} --tasks-per-node=1 -u bash -c '
    # Python path setup
    export PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/benchmark-image-tokenzier:$PYTHONPATH

    if [[ $SLURM_PROCID = 0 ]]; then
        echo "=================================================================================="
        echo "[Master] Starting Ray head on ${MASTER_NODE_IP}:${RAY_PORT}"
        echo "=================================================================================="

        # Start Ray head with redis password
        ray start --head \
            --node-ip-address=$MASTER_NODE_IP \
            --port=$RAY_PORT \
            --num-cpus=${SLURM_CPUS_PER_TASK} \
            --num-gpus=4 \
            --redis-password=$REDIS_PASSWORD

        # Wait for workers to connect
        echo "[Master] Waiting 30 seconds for workers to join..."
        sleep 30  # Workers sleep 15s, then need time to connect

        echo "[Master] Proceeding with image-text tokenization (workers should be connected)"

        # Show cluster resources
        echo "=================================================================================="
        echo "[Master] Ray Cluster Resources:"
        ray status
        echo "=================================================================================="

        # Build command with optional range parameter
        CMD="python /iopsstor/scratch/cscs/xyixuan/benchmark-image-tokenzier/vision_tokenization/core/webdataset_emu3_image_text_parallel.py \
            --input-pattern \"${INPUT_PATTERN}\" \
            --output-dir \"${OUTPUT_DIR}\" \
            --tokenizer-path \"${TOKENIZER_PATH}\" \
            --num-gpus $((SLURM_JOB_NUM_NODES * 4))"

        # Add range parameter if specified
        if [[ -n "${RANGE}" ]]; then
            CMD="${CMD} --range \"${RANGE}\""
        fi

        # Launch tokenization job
        echo "[Master] Starting EMU3 image-text pair tokenization..."
        echo "[Master] Command: ${CMD}"
        eval ${CMD}

        EXITCODE=$?

        if [ $EXITCODE -eq 0 ]; then
            echo "=================================================================================="
            echo "[Master] IMAGE-TEXT TOKENIZATION COMPLETED SUCCESSFULLY"
            echo "=================================================================================="

            # List output files
            echo "[Master] Output files:"
            ls -lah ${OUTPUT_DIR}/*.bin 2>/dev/null | head -10 || echo "No output files found yet"
            echo "[Master] Total output files: $(ls ${OUTPUT_DIR}/*.bin 2>/dev/null | wc -l) bin/idx pairs"

        else
            echo "=================================================================================="
            echo "[Master] IMAGE-TEXT TOKENIZATION FAILED WITH EXIT CODE: $EXITCODE"
            echo "=================================================================================="
        fi

        # Shutdown Ray
        ray stop

    else
        # Worker nodes
        sleep 15  # Give master time to start

        NODE_IP=$(hostname -i)
        echo "[Worker-${SLURM_PROCID}] Starting Ray worker on ${NODE_IP}"

        ray start --address="${RAY_ADDRESS}" \
                  --num-cpus=${SLURM_CPUS_PER_TASK} \
                  --num-gpus=4 \
                  --redis-password=$REDIS_PASSWORD \
                  --block

        echo "[Worker-${SLURM_PROCID}] Ray worker stopped"
    fi
'

echo "=================================================================================="
echo "Job completed at $(date)"
echo "=================================================================================="