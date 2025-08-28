#!/bin/bash
#SBATCH --account=a-infra01
#SBATCH --environment=emu3
#SBATCH --job-name=img2dataset
#SBATCH --output=/iopsstor/scratch/cscs/%u/benchmark-image-tokenzier/download_dataset/logs/mint1t-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/benchmark-image-tokenzier/download_dataset/logs/mint1t-%A.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72

# Create log directory
mkdir -p /iopsstor/scratch/cscs/$USER/benchmark-image-tokenzier/download_dataset/logs

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Configuration
OUTPUT_DIR="/capstor/store/cscs/swissai/infra01/vision-datasets/Mint-1T"

echo "=========================================="
echo "MINT-1T Download Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
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

# Download MINT-1T-ArXiv dataset from HuggingFace
echo "Starting download from HuggingFace..."
huggingface-cli download \
    mlfoundations/MINT-1T-ArXiv \
    --repo-type dataset \
    --local-dir ./MINT-1T-ArXiv \
    --resume-download

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
    echo "Total tar files: $(ls $OUTPUT_DIR/*.tar 2>/dev/null | wc -l)"
    echo "Total size: $(du -sh $OUTPUT_DIR | cut -f1)"
else
    echo "Download failed with exit code: $EXIT_CODE"
fi

echo "=========================================="