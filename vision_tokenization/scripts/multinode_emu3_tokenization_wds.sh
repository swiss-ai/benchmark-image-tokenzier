#!/bin/bash
#SBATCH --account=a-infra01
#SBATCH --job-name=emu3-tok
#SBATCH --environment=emu3
#SBATCH --nodes=11
#SBATCH --exclusive
#SBATCH --partition=normal
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=12:00:00
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/benchmark-image-tokenzier/vision_tokenization/logs/emu3_tok%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/benchmark-image-tokenzier/vision_tokenization/logs/emu3_tok%j.err
#SBATCH --exclude=nid[006569,006601,006609,006622-006623,006628-006629,006632,006638,006651,006653-006655,006658-006662,006664-006665,006669-006671,006674-006677]

###################### EMU3 Vision Tokenization on Multiple Nodes ######################
# Monitor GPU usage: srun --jobid=<jobid> --overlap -w nid<node number> --pty nvidia-smi

# Essential configuration
export SLURM_JOB_NUM_NODES="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}"
export GPUS_PER_NODE=4

# Input/Output paths
export INPUT_PATTERN="/capstor/store/cscs/swissai/infra01/vision-datasets/imagenet-w21-wds/*.tar"
export OUTPUT_DIR="/capstor/store/cscs/swissai/infra01/vision-datasets/tokenized/imagenet-w21"
export TOKENIZER_PATH="/capstor/store/cscs/swissai/infra01/MLLM/llama3_emu3_tokenizer"

# Resolution filtering (optional - comment out if not needed)
export MIN_RESOLUTION="256*256"
export MAX_RESOLUTION="1024*1024"

# Range processing (optional - comment out to process all shards)
export RANGE="200:2048"  # Process first 100 shards

# Ray minimal config  /capstor/store/cscs/swissai/infra01/MLLM/llama3_emu3_tokenizer
export MASTER_NODE=$(hostname)
export MASTER_NODE_IP=$(hostname -i)
export RAY_PORT=6379
export RAY_ADDRESS="${MASTER_NODE_IP}:${RAY_PORT}"
export REDIS_PASSWORD="EMU3TOKEN"

echo "=================================================================================="
echo "EMU3 Vision Tokenization - Multi-node Processing"
echo "=================================================================================="
echo "Nodes: ${SLURM_JOB_NUM_NODES}"
echo "Total GPUs: $((SLURM_JOB_NUM_NODES * GPUS_PER_NODE))"
echo "Input: ${INPUT_PATTERN}"
echo "Output: ${OUTPUT_DIR}"
echo "Master Node: ${MASTER_NODE} (${MASTER_NODE_IP})"
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
        
        echo "[Master] Proceeding with tokenization (workers should be connected)"
        
        # Show cluster resources
        echo "=================================================================================="
        echo "[Master] Ray Cluster Resources:"
        ray status
        echo "=================================================================================="
        
        # Launch tokenization job
        echo "[Master] Starting EMU3 tokenization..."

        # Build command with optional parameters
        CMD="python /iopsstor/scratch/cscs/xyixuan/benchmark-image-tokenzier/vision_tokenization/core/webdataset_emu3_ray_dynamic_clean.py"
        CMD="${CMD} --input-pattern \"${INPUT_PATTERN}\""
        CMD="${CMD} --output-dir \"${OUTPUT_DIR}\""
        CMD="${CMD} --tokenizer-path \"${TOKENIZER_PATH}\""
        CMD="${CMD} --num-gpus $((SLURM_JOB_NUM_NODES * 4))"

        # Add resolution filtering if specified
        [ -n "${MIN_RESOLUTION}" ] && CMD="${CMD} --min-resolution \"${MIN_RESOLUTION}\""
        [ -n "${MAX_RESOLUTION}" ] && CMD="${CMD} --max-resolution \"${MAX_RESOLUTION}\""

        # Add range if specified
        [ -n "${RANGE}" ] && CMD="${CMD} --range \"${RANGE}\""

        echo "Running command: ${CMD}"
        eval ${CMD}
        
        EXITCODE=$?
        
        if [ $EXITCODE -eq 0 ]; then
            echo "=================================================================================="
            echo "[Master] TOKENIZATION COMPLETED SUCCESSFULLY"
            echo "=================================================================================="
            
            # List output files
            echo "[Master] Output files:"
            ls -lah ${OUTPUT_DIR}/*.bin 2>/dev/null || echo "No output files found yet"
            
            # Optional: Merge worker outputs
            # echo "[Master] Merging worker outputs..."
            # python /path/to/merge_script.py --input-pattern "${OUTPUT_PREFIX}_worker*.bin"
        else
            echo "=================================================================================="
            echo "[Master] TOKENIZATION FAILED WITH EXIT CODE: $EXITCODE"
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