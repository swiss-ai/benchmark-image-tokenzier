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
DATASET_NAME="${DATASET_NAME:-HuggingFaceM4/FineVision}"
SUBSET_NAME="${SUBSET_NAME:-$1}"  # Use first argument if SUBSET_NAME not set
SPLIT="${SPLIT:-train}"
CACHE_DIR="${CACHE_DIR:-/capstor/store/cscs/swissai/infra01/vision-datasets/hf_cache}"
FORCE_REDOWNLOAD="${FORCE_REDOWNLOAD:-false}"

# Auto-detect number of processes if not set
if [ -z "$NUM_PROC" ]; then
    # Use SLURM CPUs if available, otherwise detect from system
    if [ -n "$SLURM_CPUS_PER_TASK" ]; then
        NUM_PROC=$SLURM_CPUS_PER_TASK
    else
        NUM_PROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 32)
    fi
fi

# Validate SUBSET_NAME for datasets that require it
if [[ -z "$SUBSET_NAME" && "$DATASET_NAME" == "HuggingFaceM4/FineVision" ]]; then
    echo "=========================================="
    echo "ERROR: SUBSET_NAME required for FineVision!"
    echo "=========================================="
    echo "Usage: sbatch $0 <subset_name>"
    echo ""
    echo "Available subsets: laion_gpt4v, lvis_instruct4v, ocrvqa, etc."
    echo "Example: sbatch $0 laion_gpt4v"
    echo ""
    echo "For datasets without subsets (e.g., google/docci):"
    echo "  DATASET_NAME=\"google/docci\" sbatch $0 \"\""
    exit 1
fi

# Create log directory
mkdir -p /iopsstor/scratch/cscs/$USER/benchmark-image-tokenzier/download_dataset/logs

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ========================================
# Display Configuration
# ========================================

echo "=========================================="
echo "HuggingFace Dataset Download Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
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

# Install datasets library if needed
echo ""
echo "Setting up Python environment..."
pip install -q datasets hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

# ========================================
# Download Dataset using Python
# ========================================

echo ""
echo "Starting dataset download..."
echo ""

# Determine download mode
if [ "$FORCE_REDOWNLOAD" = "true" ]; then
    DOWNLOAD_MODE="force_redownload"
    echo "Mode: Force re-download (will ignore cache)"
else
    DOWNLOAD_MODE="reuse_cache_if_exists"
    echo "Mode: Reuse cache if exists"
fi

# Create Python download script
python3 << EOF
import sys
from datasets import load_dataset

print("=" * 80)
print("Loading dataset via HuggingFace datasets library")
print("=" * 80)

# Configuration
dataset_name = "$DATASET_NAME"
subset_name = "$SUBSET_NAME" if "$SUBSET_NAME" else None
split = "$SPLIT"
cache_dir = "$CACHE_DIR"
download_mode = "$DOWNLOAD_MODE"
num_proc = int("$NUM_PROC")

print(f"Dataset:      {dataset_name}")
print(f"Subset:       {subset_name}")
print(f"Split:        {split}")
print(f"Cache dir:    {cache_dir}")
print(f"Num processes: {num_proc}")
print(f"Download mode: {download_mode}")
print("")

try:
    # Load dataset (this will download and cache it)
    print("Downloading and caching dataset...")
    dataset = load_dataset(
        dataset_name,
        name=subset_name,
        split=split,
        cache_dir=cache_dir,
        download_mode=download_mode,
        num_proc=num_proc
    )

    print("")
    print("=" * 80)
    print("DOWNLOAD SUCCESSFUL!")
    print("=" * 80)
    print(f"Dataset: {dataset_name}")
    if subset_name:
        print(f"Subset: {subset_name}")
    print(f"Split: {split}")
    print(f"Number of samples: {len(dataset):,}")
    print(f"Cached in: {cache_dir}")

    # Show dataset features
    print("")
    print("Dataset features:")
    for feature_name, feature_type in dataset.features.items():
        print(f"  - {feature_name}: {feature_type}")

    print("")
    print("=" * 80)

except Exception as e:
    print("")
    print("=" * 80)
    print("DOWNLOAD FAILED!")
    print("=" * 80)
    print(f"Error: {e}")
    print("")
    import traceback
    traceback.print_exc()
    sys.exit(1)

EOF

# Check exit status
EXIT_CODE=$?

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
    echo "To use with tokenization, add:"
    echo "  --cache-dir $CACHE_DIR"
    echo ""
    echo "See download_dataset/README.md for usage examples."
else
    echo ""
    echo "ERROR: Download failed with exit code: $EXIT_CODE"
    echo "Check the error messages above for details."
    echo ""
fi

echo "=========================================="