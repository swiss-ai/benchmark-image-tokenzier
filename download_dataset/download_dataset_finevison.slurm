#!/bin/bash
#SBATCH --account=a-infra01
#SBATCH --environment=emu3
#SBATCH --job-name=img2dataset
#SBATCH --output=/iopsstor/scratch/cscs/%u/benchmark-image-tokenizer/download_dataset/logs/finevision-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/benchmark-image-tokenizer/download_dataset/logs/finevision-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72

# Check if subset argument is provided
if [ -z "$1" ]; then
    echo "Error: No subset specified!"
    echo "Usage: sbatch $0 <subset_name>"
    echo "Example: sbatch $0 CC-MAIN-2024-10"
    exit 1
fi

SUBSET_NAME="$1"

# Create log directory
mkdir -p /iopsstor/scratch/cscs/$USER/benchmark-image-tokenzier/download_dataset/logs

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Configuration
BASE_OUTPUT_DIR="/capstor/store/cscs/swissai/infra01/vision-datasets/hf_downloads/finevision"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/" # ${SUBSET_NAME}" hf creates folder automatically

echo "=========================================="
echo "FineVision Dataset Download Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Subset: $SUBSET_NAME"
echo "Start time: $(date)"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# Create output directory
mkdir -p $OUTPUT_DIR
cd $OUTPUT_DIR

# Install and enable hf_transfer for faster downloads
echo "Setting up fast download with hf_transfer..."
pip install -q hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

# Download FineVision dataset subset from HuggingFace
echo "Starting download from HuggingFace..."
echo "Repository: FineVision/FineVision"
echo "Subset: $SUBSET_NAME"

# Download train split of finevision subset using huggingface-cli
hf download \
    HuggingFaceM4/FineVision \
    --repo-type dataset \
    --include "${SUBSET_NAME}/train-*" \
    --local-dir . \


# Check exit status
EXIT_CODE=$?

echo "=========================================="
echo "Job completed at: $(date)"
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Download completed successfully!"
    echo "Checking output..."
    echo "Files downloaded:"
    ls -lah $OUTPUT_DIR | head -20
    echo ""
    echo "Total files: $(find $OUTPUT_DIR -type f 2>/dev/null | wc -l)"
    echo "Total size: $(du -sh $OUTPUT_DIR | cut -f1)"
else
    echo "Download failed with exit code: $EXIT_CODE"
fi

echo "=========================================="