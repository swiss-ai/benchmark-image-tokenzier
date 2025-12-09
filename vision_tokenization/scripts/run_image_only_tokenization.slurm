#!/bin/bash
#SBATCH --account=a-infra01
#SBATCH --job-name=image-only-tok
#SBATCH --environment=emu3
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=normal
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=12:00:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/benchmark-image-tokenizer/vision_tokenization/logs/image_only_tok_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/benchmark-image-tokenizer/vision_tokenization/logs/image_only_tok_%j.err

# Script to run image-only tokenization with optional debug and resume modes
# Usage:
#   sbatch run_image_only_tokenization.slurm              # Normal mode
#   sbatch run_image_only_tokenization.slurm --debug      # Debug mode (small dataset)
#   sbatch run_image_only_tokenization.slurm --resume     # Resume from checkpoint
#   sbatch run_image_only_tokenization.slurm --debug --resume  # Debug + resume

# Get the project root directory
PROJECT_ROOT="/iopsstor/scratch/cscs/${USER}/benchmark-image-tokenizer"

# Configuration
CONFIG_PATH="${PROJECT_ROOT}/vision_tokenization/configs/llava_onevision_midtrain_image_only_to_emu3.5_apertus.json"

# Hub cache location
HF_HUB_CACHE="${HF_HUB_CACHE:-/capstor/store/cscs/swissai/infra01/vision-datasets/hf_hub_cache}"

# Allow environment variable override for num-gpus (used in multi-node mode)
NUM_GPUS="${NUM_GPUS:-}"

# Parse command-line arguments
DEBUG_MODE=false
RESUME_MODE=false

for arg in "$@"; do
    case $arg in
        --debug)
            DEBUG_MODE=true
            shift
            ;;
        --resume)
            RESUME_MODE=true
            shift
            ;;
        -h|--help)
            echo "Usage: sbatch $0 [--debug] [--resume]"
            echo ""
            echo "Options:"
            echo "  --debug     Run with small dataset (100 samples, 1 GPU, 1 shard)"
            echo "  --resume    Resume from checkpoint (skip completed shards)"
            echo ""
            exit 0
            ;;
        *)
            ;;
    esac
done

# Set up Python path
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
pip install -q omegaconf

echo "=================================================================================="
echo "LLaVA-OneVision Mid-Training Image-Only Tokenization"
echo "=================================================================================="
echo "Project root: ${PROJECT_ROOT}"
echo "Config:       ${CONFIG_PATH}"
echo "Debug mode:   ${DEBUG_MODE}"
echo "Resume mode:  ${RESUME_MODE}"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "HF HUB CACHE  ${HF_HUB_CACHE}"
echo "Node:         $(hostname)"
echo "=================================================================================="

# Build resume flag
RESUME_FLAG=""
if [ "$RESUME_MODE" = true ]; then
    RESUME_FLAG="--resume"
    echo "RESUME mode enabled - will skip completed shards"
fi

# Build GPU override flag (empty by default, set if NUM_GPUS environment variable is provided)
GPU_FLAG=""
if [ -n "${NUM_GPUS}" ]; then
    GPU_FLAG="--num-gpus ${NUM_GPUS}"
    echo "GPU override: ${NUM_GPUS} GPUs (from NUM_GPUS environment variable)"
fi

# Export HF_HUB_CACHE so Python scripts inherit it (important to set in case of central cache location in distributed envs and using default load method)
export HF_HUB_CACHE

if [ "$DEBUG_MODE" = true ]; then
    echo "Running in DEBUG mode (small dataset for testing)"
    echo "=================================================================================="
    OUTPUT_DIR="${PROJECT_ROOT}/vision_tokenization/tokenized_outputs/image_only_1000_s2"
    # Debug mode: override config with small dataset parameters
    python -m vision_tokenization.tokenize \
        --output-dir "${OUTPUT_DIR}" \
        --num-gpus 4 \
        --log-interval 100 \
        hf \
        --config "${CONFIG_PATH}" \
        --dataset-split "train[:1000]" \
        --num-shards 2 \
        ${RESUME_FLAG}
else
    echo "Running in NORMAL mode (full dataset)"
    echo "=================================================================================="

    # Normal mode: use config file as-is
    python -m vision_tokenization.tokenize ${GPU_FLAG} hf \
        --config "${CONFIG_PATH}" \
        ${RESUME_FLAG}
fi

EXITCODE=$?

if [ $EXITCODE -eq 0 ]; then
    echo "=================================================================================="
    echo "TOKENIZATION COMPLETED SUCCESSFULLY"
    echo "=================================================================================="

    # Show output directory
    if [ "$DEBUG_MODE" = false ]; then
        # Extract output_dir from config
        OUTPUT_DIR=$(python -c "import json; print(json.load(open('${CONFIG_PATH}'))['output_dir'])" 2>/dev/null || echo "")
    fi

    if [ -n "$OUTPUT_DIR" ] && [ -d "$OUTPUT_DIR" ]; then
        echo "Output directory: ${OUTPUT_DIR}"
        echo ""
        echo "Output files:"
        ls -lh "${OUTPUT_DIR}"/*.bin "${OUTPUT_DIR}"/*.idx 2>/dev/null | head -20 || echo "No output files found"

        # Show statistics if available
        if [ -f "${OUTPUT_DIR}/statistics.json" ]; then
            echo ""
            echo "Statistics:"
            cat "${OUTPUT_DIR}/statistics.json"
        fi

        # Show dataset_info.json if available
        if [ -f "${OUTPUT_DIR}/dataset_info.json" ]; then
            echo ""
            echo "Dataset Info:"
            cat "${OUTPUT_DIR}/dataset_info.json"
        fi
    fi
else
    echo "=================================================================================="
    echo "TOKENIZATION FAILED WITH EXIT CODE: $EXITCODE"
    echo "=================================================================================="
    exit $EXITCODE
fi

echo "=================================================================================="
echo "Job completed at $(date)"
echo "=================================================================================="