#!/bin/bash
#SBATCH --job-name=emu3_tokenize
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --mem=480GB
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/emu3_tokenize_%j.out
#SBATCH --error=logs/emu3_tokenize_%j.err

# Distributed EMU3 Tokenization Script for Multi-Node Processing

# Load required modules
module load python/3.10
module load cuda/11.8
module load nccl

# Activate virtual environment
source /path/to/venv/bin/activate

# Set environment variables
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))

# NCCL settings for optimal performance
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_LEVEL=2
export NCCL_P2P_DISABLE=0

# PyTorch settings
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Data paths
INPUT_SHARDS="/data/webdataset/imagenet/shard-{000000..001023}.tar"
OUTPUT_PREFIX="/data/processed/emu3_tokens/imagenet"
TOKENIZER_PATH="/models/emu3_tokenizer"
VISION_TOKENIZER_PATH="/models/vision_tokenizer"

# Processing settings
BATCH_SIZE=64
NUM_WORKERS=8
PREFETCH_FACTOR=2
BUFFER_SIZE=10000
CHECKPOINT_DIR="/checkpoints/emu3_processing"
CHECKPOINT_INTERVAL=5000

# Log configuration
echo "Starting distributed EMU3 tokenization"
echo "Master node: $MASTER_ADDR"
echo "World size: $WORLD_SIZE"
echo "Nodes: $SLURM_NNODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"

# Create necessary directories
mkdir -p $(dirname $OUTPUT_PREFIX)
mkdir -p $CHECKPOINT_DIR
mkdir -p logs

# Run distributed processing
srun python -m torch.distributed.run \
    --nproc_per_node=$SLURM_NTASKS_PER_NODE \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    utils/webdataset_emu3_distributed.py \
    --input-shards "$INPUT_SHARDS" \
    --output-prefix "$OUTPUT_PREFIX" \
    --tokenizer-path "$TOKENIZER_PATH" \
    --vision-tokenizer-path "$VISION_TOKENIZER_PATH" \
    --batch-size $BATCH_SIZE \
    --num-workers $NUM_WORKERS \
    --prefetch-factor $PREFETCH_FACTOR \
    --buffer-size $BUFFER_SIZE \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --checkpoint-interval $CHECKPOINT_INTERVAL \
    --backend nccl

# Check exit status
if [ $? -eq 0 ]; then
    echo "Processing completed successfully"
    
    # Clean up checkpoints
    rm -rf $CHECKPOINT_DIR/*
    
    # Verify output
    echo "Verifying output files..."
    ls -lh ${OUTPUT_PREFIX}.*
else
    echo "Processing failed with exit code $?"
    echo "Checkpoints preserved for resume"
fi