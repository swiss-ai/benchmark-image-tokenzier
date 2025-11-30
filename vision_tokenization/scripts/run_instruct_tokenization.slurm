#!/bin/bash
#SBATCH --account=a-infra01
#SBATCH --job-name=instruct-tok
#SBATCH --environment=emu3
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=normal
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=12:00:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/benchmark-image-tokenizer/vision_tokenization/logs/instruct_tok_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/benchmark-image-tokenizer/vision_tokenization/logs/instruct_tok_%j.err

# Script to run instruct tokenization with optional debug and resume modes
# Usage:
#   sbatch run_instruct_tokenization.slurm              # Normal mode
#   sbatch run_instruct_tokenization.slurm --debug      # Debug mode (small dataset)
#   sbatch run_instruct_tokenization.slurm --resume     # Resume from checkpoint
#   sbatch run_instruct_tokenization.slurm --debug --resume  # Debug + resume

# Get the project root directory
PROJECT_ROOT="/iopsstor/scratch/cscs/${USER}/benchmark-image-tokenizer"

# Configuration
CONFIG_PATH="${PROJECT_ROOT}/vision_tokenization/configs/llava_onevision_instruct_to_emu3.5_apertus_sft.json"

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

echo "=================================================================================="
echo "LLaVA-OneVision Instruct Tokenization"
echo "=================================================================================="
echo "Project root: ${PROJECT_ROOT}"
echo "Config:       ${CONFIG_PATH}"
echo "Debug mode:   ${DEBUG_MODE}"
echo "Resume mode:  ${RESUME_MODE}"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         $(hostname)"
echo "=================================================================================="

# Build resume flag
RESUME_FLAG=""
if [ "$RESUME_MODE" = true ]; then
    RESUME_FLAG="--resume"
    echo "RESUME mode enabled - will skip completed shards"
fi

if [ "$DEBUG_MODE" = true ]; then
    echo "Running in DEBUG mode (small dataset for testing)"
    echo "=================================================================================="

    # Debug mode: override config with small dataset parameters
    python -m vision_tokenization.tokenize \
        --output-dir "${PROJECT_ROOT}/vision_tokenization/tokenized_outputs/instruct_100" \
        --num-gpus 1 \
        hf \
        --config "${CONFIG_PATH}" \
        --config-name "CLEVR" \
        --dataset-split "train[:100]" \
        --num-shards 1 \
        ${RESUME_FLAG}
else
    echo "Running in NORMAL mode (full dataset)"
    echo "=================================================================================="

    # Normal mode: use config file as-is
    python -m vision_tokenization.tokenize hf \
        --config "${CONFIG_PATH}" \
        ${RESUME_FLAG}
fi

EXITCODE=$?

if [ $EXITCODE -eq 0 ]; then
    echo "=================================================================================="
    echo "TOKENIZATION COMPLETED SUCCESSFULLY"
    echo "=================================================================================="

    # Show output directory
    if [ "$DEBUG_MODE" = true ]; then
        OUTPUT_DIR="${PROJECT_ROOT}/vision_tokenization/tokenized_outputs/instruct_100"
    else
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