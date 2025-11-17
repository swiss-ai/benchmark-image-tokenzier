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
pip install -q datasets hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

# ========================================
# Download Dataset using Builder
# ========================================

echo ""
echo "Starting dataset download (builder mode)..."
echo ""

python3 << EOF
import sys
from datasets import load_dataset_builder, DownloadMode, VerificationMode

print("=" * 80)
print("Downloading dataset via HuggingFace DatasetBuilder")
print("=" * 80)

dataset_name = "${DATASET_NAME}"
subset_name = "${SUBSET_NAME}"
cache_dir = "${CACHE_DIR}"
num_proc = int("${NUM_PROC}")
force_redownload = "${FORCE_REDOWNLOAD}".lower() == "true"

# datasets lib will always delete .incomplete folder of split gen on crash and start over with split generation 
download_mode = (
    DownloadMode.FORCE_REDOWNLOAD
    if force_redownload
    else DownloadMode.REUSE_CACHE_IF_EXISTS
)

# All checks only useful to test deterioation over time (hf hub files have hash as filename so coule run check directly on it manually) 
verification_mode = VerificationMode.BASIC_CHECKS

print(f"Dataset:       {dataset_name}")
print(f"Subset:        {subset_name or '<none>'}")
print(f"Cache dir:     {cache_dir}")
print(f"Num processes: {num_proc}")
print(f"Download mode: {download_mode}")
print("")

try:
    # Initialize dataset builder
    builder = load_dataset_builder(dataset_name, name=subset_name or None, cache_dir=cache_dir)
    print(f"Builder loaded: {builder.info.builder_name}")

    # Download and prepare dataset (resumable and cached)
    builder.download_and_prepare(
        download_mode=download_mode,
        num_proc=num_proc,
        verification_mode=verification_mode,
    )

    print("")
    print("=" * 80)
    print("✅ DOWNLOAD SUCCESSFUL!")
    print("=" * 80)
    print(f"Dataset: {dataset_name}")
    if subset_name:
        print(f"Subset: {subset_name}")
    print(f"Cached in: {cache_dir}")
    print(f"Splits: {builder.info.splits.keys()}")
    print(f"Features: {builder.info.features}")
    print("=" * 80)

except Exception as e:
    print("")
    print("=" * 80)
    print("❌ DOWNLOAD FAILED!")
    print("=" * 80)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

EOF

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