#!/bin/bash
#SBATCH --account=a-infra01
#SBATCH --environment=emu3
#SBATCH --job-name=dwnldHFdset
#SBATCH --output=/iopsstor/scratch/cscs/%u/benchmark-image-tokenizer/download_dataset/logs/hf-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/benchmark-image-tokenizer/download_dataset/logs/hf-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1

################################################################################
# HuggingFace Dataset Downloader
################################################################################
# Downloads HF datasets to cache for tokenization pipeline.
# Use --cache-dir during tokenization to load from cache.
#
# USAGE:
#   sbatch download_dataset_hf.slurm <subset>
#   DATASET_NAME="google/docci" sbatch download_dataset_hf.slurm ""
#
# PARAMETERS (environment variables):
#   DATASET_NAME     - HF dataset repo (default: HuggingFaceM4/FineVision)
#   SUBSET_NAME      - Dataset subset/config (default: $1, use "" if none)
#   SPLIT            - Dataset split (default: train)
#   CACHE_DIR        - Cache path (default: /capstor/.../hf_cache)
#   NUM_PROC         - Download processes (default: auto = CPUs available)
#   FORCE_REDOWNLOAD - Re-download if cached (default: false)
#
# EXAMPLES:
#   sbatch download_dataset_hf.slurm laion_gpt4v
#   DATASET_NAME="google/docci" sbatch download_dataset_hf.slurm ""
#   CACHE_DIR="/custom/path" sbatch download_dataset_hf.slurm ocrvqa
#   NUM_PROC=64 sbatch download_dataset_hf.slurm lvis_instruct4v
################################################################################

# ========================================
# Configuration Parameters
# ========================================

# Dataset configuration (overridable via environment variables)
DATASET_NAME="${DATASET_NAME:-mvp-lab/LLaVA-OneVision-1.5-Mid-Training-85M}"
SUBSET_NAME="${SUBSET_NAME:-$1}"  # Use first argument if SUBSET_NAME not set
SPLIT="${SPLIT:-train}"
CACHE_DIR="${CACHE_DIR:-/capstor/store/cscs/swissai/infra01/vision-datasets/hf_cache}"
FORCE_REDOWNLOAD="${FORCE_REDOWNLOAD:-false}"

NUM_PROC=12

# Auto-detect number of processes if not set
if [ -z "$NUM_PROC" ]; then
    # Use SLURM CPUs if available, otherwise detect from system
    if [ -n "$SLURM_CPUS_PER_TASK" ]; then
        NUM_PROC=$SLURM_CPUS_PER_TASK
    else
        NUM_PROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 32)
    fi
fi

# Create log directory
mkdir -p /iopsstor/scratch/cscs/$USER/benchmark-image-tokenzier/download_dataset/logs

# Set threading
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# ========================================
# Display Configuration
# ========================================

echo "=========================================="
echo "HuggingFace Dataset Download Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $NUM_PROC"
echo "Start time: $(date)"
echo ""
echo "Configuration:"
echo "  Dataset:          $DATASET_NAME"
echo "  Subset:           ${SUBSET_NAME:-<none>}"
echo "  Split:            $SPLIT"
echo "  Cache Directory:  $CACHE_DIR"
echo "  Num Processes:    $NUM_PROC"
echo "  Force Redownload: $FORCE_REDOWNLOAD"
echo "=========================================="

# Create cache directory
mkdir -p "$CACHE_DIR"

# Clean stale lock files (only within this dataset)
echo ""
echo "Cleaning up stale lock files..."
LOCK_COUNT=$(find "$CACHE_DIR" -name "*.lock" 2>/dev/null | wc -l)
if [ "$LOCK_COUNT" -gt 0 ]; then
    echo "Found $LOCK_COUNT lock file(s) - removing..."
    find "$CACHE_DIR" -name "*.lock" -delete 2>/dev/null
    echo "Lock files cleaned up"
else
    echo "No stale lock files found"
fi

# ========================================
# Python Environment Setup
# ========================================

echo ""
echo "Setting up Python environment..."

# Check if required packages are installed, install with pinned versions if missing
python3 -c "import datasets, hf_transfer, huggingface_hub" 2>/dev/null || {
    echo "Installing required packages with pinned versions..."
    pip install -q \
        datasets==3.2.0 \
        hf_transfer==0.1.8 \
        huggingface_hub==0.26.2
    echo "Packages installed successfully"
}

# Enable fast transfers via hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

# ========================================
# Download Dataset using Builder
# ========================================
echo ""
echo "Starting dataset download (builder mode)..."
echo ""

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Build the Python command with arguments
PYTHON_CMD="python3 ${SCRIPT_DIR}/download_hf_dataset.py \
    --dataset-name \"${DATASET_NAME}\" \
    --cache-dir \"${CACHE_DIR}\" \
    --num-proc ${NUM_PROC}"

# Add optional subset name if provided
if [ -n "$SUBSET_NAME" ]; then
    PYTHON_CMD="${PYTHON_CMD} --subset-name \"${SUBSET_NAME}\""
fi

# Add force redownload flag if enabled
if [ "$FORCE_REDOWNLOAD" = "true" ]; then
    PYTHON_CMD="${PYTHON_CMD} --force-redownload"
fi

# Execute the Python script
eval "$PYTHON_CMD"

EXIT_CODE=$?

# ========================================
# Job Summary
# ========================================

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "SUCCESS! Dataset cached and ready for tokenization."
    echo ""
    echo "Cache location: $CACHE_DIR"
    echo "Cache size: $(du -sh "$CACHE_DIR" 2>/dev/null | cut -f1 || echo 'unknown')"
    echo ""
    echo "Use with tokenization via:"
    echo "  --cache-dir $CACHE_DIR"
    echo ""
else
    echo ""
    echo "ERROR: Download failed with exit code: $EXIT_CODE"
    echo "Check error logs for details."
    echo ""
fi

echo "=========================================="