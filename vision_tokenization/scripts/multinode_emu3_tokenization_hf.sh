#!/bin/bash
#SBATCH --account=a-infra01
#SBATCH --job-name=emu3-tok
#SBATCH --environment=emu3
#SBATCH --nodes=4
#SBATCH --exclusive
#SBATCH --partition=normal
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=12:00:00
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/benchmark-image-tokenzier/vision_tokenization/logs/emu3_tok%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/benchmark-image-tokenzier/vision_tokenization/logs/emu3_tok%j.err

###################### EMU3 Vision Tokenization on Multiple Nodes ######################
# Monitor GPU usage: srun --jobid=<jobid> --overlap -w nid<node number> --pty nvidia-smi

# Essential configuration
export SLURM_JOB_NUM_NODES="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}"
export GPUS_PER_NODE=4

# Configuration - can be overridden when submitting job
# Example: CONFIG_NAME=densefusion_1m sbatch multinode_emu3_tokenization_hf.sh
export CONFIG_NAME="${CONFIG_NAME:-latexformulas}"

# Ray minimal config  
export MASTER_NODE=$(hostname)
export MASTER_NODE_IP=$(hostname -i)
export RAY_PORT=6379
export RAY_ADDRESS="${MASTER_NODE_IP}:${RAY_PORT}"
export REDIS_PASSWORD="EMU3TOKEN"

echo "=================================================================================="
echo "HuggingFace Dataset Tokenization - Multi-node Processing"
echo "=================================================================================="
echo "Nodes: ${SLURM_JOB_NUM_NODES}"
echo "Total GPUs: $((SLURM_JOB_NUM_NODES * GPUS_PER_NODE))"
echo "Dataset: HuggingFaceM4/FineVision/${CONFIG_NAME}"
echo "Output: /capstor/store/cscs/swissai/infra01/vision-datasets/FineVision/tokenized/${CONFIG_NAME}"
echo "Master Node: ${MASTER_NODE} (${MASTER_NODE_IP})"
echo "===================================================================================="

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
        echo "[Master] Starting HF dataset tokenization..."
        python /iopsstor/scratch/cscs/xyixuan/benchmark-image-tokenzier/vision_tokenization/tokenize_hf_datasets.py \
            --dataset-name "HuggingFaceM4/FineVision" \
            --config-name "${CONFIG_NAME:-laion_gpt4v}" \
            --output-dir "/capstor/store/cscs/swissai/infra01/vision-datasets/FineVision/tokenized" \
            --tokenizer-path "/capstor/store/cscs/swissai/infra01/MLLM/llama3_emu3_tokenizer" \
            --num-gpus $((SLURM_JOB_NUM_NODES * 4)) \
            --batch-size 1000
        
        EXITCODE=$?
        
        if [ $EXITCODE -eq 0 ]; then
            echo "=================================================================================="
            echo "[Master] TOKENIZATION COMPLETED SUCCESSFULLY"
            echo "=================================================================================="
            
            # List output files
            OUTPUT_PATH="/capstor/store/cscs/swissai/infra01/vision-datasets/FineVision/tokenized/${CONFIG_NAME}"
            echo "[Master] Output files in ${OUTPUT_PATH}:"
            ls -lah ${OUTPUT_PATH}/*.bin 2>/dev/null || echo "No .bin files found"
            ls -lah ${OUTPUT_PATH}/*.idx 2>/dev/null || echo "No .idx files found"
            
            # Show dataset info
            if [ -f "${OUTPUT_PATH}/dataset_info.json" ]; then
                echo "[Master] Dataset info:"
                cat ${OUTPUT_PATH}/dataset_info.json
            fi
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