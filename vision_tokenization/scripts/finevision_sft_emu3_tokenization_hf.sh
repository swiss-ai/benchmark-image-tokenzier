#!/bin/bash
#SBATCH --account=a-infra01
#SBATCH --job-name=emu3-sft-tok
#SBATCH --environment=emu3
#SBATCH --nodes=10
#SBATCH --exclusive
#SBATCH --partition=normal
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=12:00:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/benchmark-image-tokenizer/vision_tokenization/logs/emu3_sft_tok%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/benchmark-image-tokenizer/vision_tokenization/logs/emu3_sft_tok%j.err

###################### EMU3 SFT Vision Tokenization on Multiple Nodes ######################
# Monitor GPU usage: srun --jobid=<jobid> --overlap -w nid<node number> --pty nvidia-smi

# Essential configuration
export SLURM_JOB_NUM_NODES="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}"
export GPUS_PER_NODE=4

# Configuration - can be overridden when submitting job
# Example: CONFIG_NAME=ocrvqa sbatch finevision_sft_emu3_tokenization_hf.sh
export DATASET_NAME="lmms-lab/LLaVA-OneVision-1.5-Insturct-Data" # other could be: lmms-lab/LLaVA-OneVision-1.5-Insturct-Data OR HuggingFaceM4/FineVision
export CONFIG_NAME="${CONFIG_NAME:-aokvqa}"
#export EXISTING_DATA_FILES="${EXISTING_DATA_FILES:-/capstor/store/cscs/swissai/infra01/vision-datasets/hf_downloads/finevision/Unichart/**/*.parquet}"  # e.g. "/capstor/store/cscs/swissai/infra01/vision-datasets/hf_downloads/finevision/CC-MAIN-2024-10/**/*.parquet"

# Check if EXISTING_DATA_FILES is set and not empty
if [ -n "$EXISTING_DATA_FILES" ]; then
    export DATA_FILES_ARG="--data_files=$EXISTING_DATA_FILES"
else
    export DATA_FILES_ARG=""
fi

# Resolution filtering - can be overridden when submitting job
# Example: MIN_IMAGE_RES="512*512" MAX_IMAGE_RES="1024*1024" sbatch finevision_sft_emu3_tokenization_hf.sh
export MIN_IMAGE_RES="${MIN_IMAGE_RES:-256*256}"
export MAX_IMAGE_RES="${MAX_IMAGE_RES:-696*696}"

# set to anything except leving empty to skip res filtering for min and/or max res. Tokenizer will then resize
export SKIP_MIN_RES_FILTERING="${SKIP_MIN_RES_FILTERING:-true}"
export SKIP_MAX_RES_FILTERING="${SKIP_MAX_RES_FILTERING:-}"

# After defining the arguments, export them:
if [ -n "$SKIP_MIN_RES_FILTERING" ]; then
    echo "Skipping minimum image resolution filtering"
    export MIN_SKIP_ARG="--skip-min-resolution-check"
else
    export MIN_SKIP_ARG=""
fi

if [ -n "$SKIP_MAX_RES_FILTERING" ]; then
    echo "Skipping maximum image resolution filtering"
    export MAX_SKIP_ARG="--skip-max-resolution-check"
else
    export MAX_SKIP_ARG=""
fi

# Ray minimal config
export MASTER_NODE=$(hostname)
export MASTER_NODE_IP=$(hostname -i)
export RAY_PORT=6379
export RAY_ADDRESS="${MASTER_NODE_IP}:${RAY_PORT}"
export REDIS_PASSWORD="EMU3SFTOKEN"

echo "=================================================================================="
echo "HuggingFace FineVision SFT Dataset Tokenization - Multi-node Processing"
echo "=================================================================================="
echo "Nodes: ${SLURM_JOB_NUM_NODES}"
echo "Total GPUs: $((SLURM_JOB_NUM_NODES * GPUS_PER_NODE))"
echo "Dataset: HuggingFaceM4/FineVision/${CONFIG_NAME}"
echo "Output: /capstor/store/cscs/swissai/infra01/vision-datasets/FineVision/tokenized_sft/${OUTPUT_SUBDIR}"
echo "Master Node: ${MASTER_NODE} (${MASTER_NODE_IP})"
echo "Image Resolution Filter: ${MIN_IMAGE_RES} - ${MAX_IMAGE_RES}"
echo "===================================================================================="

# Execute on all nodes
srun -N ${SLURM_JOB_NUM_NODES} --tasks-per-node=1 -u bash -c '
    # Python path setup
    export PYTHONPATH=/iopsstor/scratch/cscs/${USER}/benchmark-image-tokenizer:$PYTHONPATH

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

        echo "[Master] Proceeding with SFT tokenization (workers should be connected)"

        # Show cluster resources
        echo "=================================================================================="
        echo "[Master] Ray Cluster Resources:"
        ray status
        echo "=================================================================================="

        # Launch SFT tokenization job
        echo "[Master] Starting HF FineVision SFT dataset tokenization..."
        echo "[Master] Processing config: ${CONFIG_NAME}"
        echo "[Master] Image resolution filter: ${MIN_IMAGE_RES} to ${MAX_IMAGE_RES}"

        python /iopsstor/scratch/cscs/${USER}/benchmark-image-tokenizer/vision_tokenization/tokenize_hf_datasets_sft.py \
            --dataset-name "${DATASET_NAME}" \
            --config-name "${CONFIG_NAME}" \
            --output-dir "/capstor/store/cscs/swissai/infra01/vision-datasets/FineVision/tokenized_sft" \
            --tokenizer-path "/capstor/store/cscs/swissai/infra01/MLLM/llama3_vision_instruct_emu3_tokenizer" \
            --num-gpus $((SLURM_JOB_NUM_NODES * 4)) \
            --batch-size 1000 \
            --min-image-res "${MIN_IMAGE_RES}" \
            --max-image-res "${MAX_IMAGE_RES}" \
            --text-field="conversations" \
            ${DATA_FILES_ARG} \
            ${MIN_SKIP_ARG} \
            ${MAX_SKIP_ARG}

        EXITCODE=$?

        if [ $EXITCODE -eq 0 ]; then
            echo "=================================================================================="
            echo "[Master] SFT TOKENIZATION COMPLETED SUCCESSFULLY"
            echo "=================================================================================="

            # List output files
            OUTPUT_PATH="/capstor/store/cscs/swissai/infra01/vision-datasets/FineVision/tokenized_sft/${OUTPUT_SUBDIR}"
            echo "[Master] Output files in ${OUTPUT_PATH}:"
            echo ""
            echo "Shard files:"
            ls -lah ${OUTPUT_PATH}/rank_*.bin 2>/dev/null | head -20 || echo "No .bin files found"
            ls -lah ${OUTPUT_PATH}/rank_*.idx 2>/dev/null | head -20 || echo "No .idx files found"

            echo ""
            echo "Total shards created:"
            NUM_BIN=$(ls ${OUTPUT_PATH}/rank_*.bin 2>/dev/null | wc -l)
            NUM_IDX=$(ls ${OUTPUT_PATH}/rank_*.idx 2>/dev/null | wc -l)
            echo "  .bin files: ${NUM_BIN}"
            echo "  .idx files: ${NUM_IDX}"

            # Show dataset info
            if [ -f "${OUTPUT_PATH}/dataset_info.json" ]; then
                echo ""
                echo "[Master] Dataset statistics:"
                python -c "
import json
with open(\"${OUTPUT_PATH}/dataset_info.json\") as f:
    info = json.load(f)
    stats = info.get(\"statistics\", {})
    print(f\"  Total samples processed: {stats.get(\"total_samples_processed\", 0):,}\")
    print(f\"  Samples skipped: {stats.get(\"samples_skipped\", 0):,}\")
    print(f\"  Resolution filtered: {stats.get(\"resolution_skipped\", 0):,}\")
    print(f\"  Total tokens: {stats.get(\"total_tokens\", 0):,}\")
    print(f\"  Image tokens: {stats.get(\"image_tokens\", 0):,} ({stats.get(\"image_tokens\", 0)/stats.get(\"total_tokens\", 1)*100:.1f}%)\")
    print(f\"  Text tokens: {stats.get(\"text_tokens\", 0):,} ({stats.get(\"text_tokens\", 0)/stats.get(\"total_tokens\", 1)*100:.1f}%)\")
    if \"processing\" in info:
        proc = info[\"processing\"]
        print(f\"  Processing time: {proc.get(\"processing_time_seconds\", 0):.1f}s\")
        print(f\"  Throughput: {proc.get(\"tokens_per_second\", 0):.0f} tokens/sec\")
"
            fi
        else
            echo "=================================================================================="
            echo "[Master] SFT TOKENIZATION FAILED WITH EXIT CODE: $EXITCODE"
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