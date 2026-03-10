#!/bin/bash
#SBATCH --account=a-infra01
#SBATCH --job-name=batch-recon
#SBATCH --environment=emu3
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=normal
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=01:00:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/benchmark-image-tokenizer/logs/batch_recon_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/benchmark-image-tokenizer/logs/batch_recon_%j.err

# Script to run batch image reconstruction with various tokenizers
# Usage:
#   sbatch run_batch_reconstruct.slurm                          # Default (Emu3VisionTokenizer)
#   sbatch run_batch_reconstruct.slurm --tokenizer Emu3_5_IBQ --tokenizer-path /path/to/model
#   sbatch run_batch_reconstruct.slurm --help                   # Show help

# Help function
show_help() {
    cat << EOF
Usage: sbatch run_batch_reconstruct.slurm [OPTIONS]

Options:
  --tokenizer <name>       Tokenizer to use (default: Emu3VisionTokenizer)
  --tokenizer-path <path>  Path to tokenizer model (required for Emu3_5_IBQ)
  --input <path>           Input folder with original images
  --output <path>          Output folder for reconstructions
  --min-pixels <int>       Minimum pixels (default: 262144 = 512*512)
  --max-pixels <int>       Maximum pixels (default: 1048576 = 1024*1024)
  --debug                  Debug mode (for testing)
  -h, --help               Show this help

Available tokenizers:
  Emu3VisionTokenizer, Emu3_5_IBQ, Cosmos, UniTok, TokenFlow, TokenFlow_molmo,
  Selftok, Selftok_molmo, DetailFlow, FlowMoTok, LlamaGenTok, OpenMAGViT2,
  FQGAN, VILA, VQGAN, IBQ

Examples:
  # Basic usage with defaults (Emu3VisionTokenizer)
  sbatch run_batch_reconstruct.slurm

  # With Emu3.5 IBQ tokenizer & custom paths
  sbatch run_batch_reconstruct.slurm \
      --tokenizer Emu3_5_IBQ \
      --tokenizer-path /capstor/store/cscs/swissai/infra01/MLLM/Emu3.5-VisionTokenizer \
      --input assets/original/spectrum25m \
      --output assets/EMU3.5/spectrum25m

  # With Emu3 tokenizer & custom paths
  sbatch run_batch_reconstruct.slurm \
      --input assets/original/spectrum25m \
      --output assets/EMU3VisionTokenizer/spectrum25m

  # Custom resolution
  sbatch run_batch_reconstruct.slurm \
      --min-pixels 131072 \
      --max-pixels 524288

  # Debug mode
  sbatch run_batch_reconstruct.slurm --debug
EOF
}

# Get the project root directory
PROJECT_ROOT="/iopsstor/scratch/cscs/${USER}/benchmark-image-tokenizer"

# Default values
TOKENIZER="Emu3VisionTokenizer"
TOKENIZER_PATH=""
INPUT_FOLDER="${PROJECT_ROOT}/assets/original"
OUTPUT_FOLDER=""  # Will be set based on tokenizer
MIN_PIXELS=262144
MAX_PIXELS=1048576
DEBUG_MODE=false

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --tokenizer)
            TOKENIZER="$2"
            shift 2
            ;;
        --tokenizer-path)
            TOKENIZER_PATH="$2"
            shift 2
            ;;
        --input)
            INPUT_FOLDER="$2"
            shift 2
            ;;
        --output)
            OUTPUT_FOLDER="$2"
            shift 2
            ;;
        --min-pixels)
            MIN_PIXELS="$2"
            shift 2
            ;;
        --max-pixels)
            MAX_PIXELS="$2"
            shift 2
            ;;
        --debug)
            DEBUG_MODE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
done

# Set default output folder if not specified
if [ -z "$OUTPUT_FOLDER" ]; then
    OUTPUT_FOLDER="${PROJECT_ROOT}/assets/${TOKENIZER}"
fi

# Set up Python path
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

echo "=================================================================================="
echo "Batch Image Reconstruction"
echo "=================================================================================="
echo "Project root: ${PROJECT_ROOT}"
echo "Tokenizer:    ${TOKENIZER}"
if [ -n "$TOKENIZER_PATH" ]; then
    echo "Model path:   ${TOKENIZER_PATH}"
fi
echo "Input:        ${INPUT_FOLDER}"
echo "Output:       ${OUTPUT_FOLDER}"
echo "Min pixels:   ${MIN_PIXELS}"
echo "Max pixels:   ${MAX_PIXELS}"
echo "Debug mode:   ${DEBUG_MODE}"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         $(hostname)"
echo "Start time:   $(date)"
echo "=================================================================================="

# Validation
if [ ! -d "$INPUT_FOLDER" ]; then
    echo "ERROR: Input folder does not exist: ${INPUT_FOLDER}"
    exit 1
fi

if [ "$TOKENIZER" = "Emu3_5_IBQ" ] && [ -z "$TOKENIZER_PATH" ]; then
    echo "ERROR: Emu3_5_IBQ requires --tokenizer-path to be specified"
    echo "Example: --tokenizer-path /capstor/store/cscs/swissai/infra01/MLLM/Emu3.5-VisionTokenizer"
    exit 1
fi

# Build command
CMD="python ${PROJECT_ROOT}/batch_reconstruct.py"
CMD="$CMD --tokenizer ${TOKENIZER}"
CMD="$CMD --input ${INPUT_FOLDER}"
CMD="$CMD --output ${OUTPUT_FOLDER}"
CMD="$CMD --min-pixels ${MIN_PIXELS}"
CMD="$CMD --max-pixels ${MAX_PIXELS}"

if [ -n "$TOKENIZER_PATH" ]; then
    CMD="$CMD --tokenizer-path ${TOKENIZER_PATH}"
fi

# Install requirements
pip install -q omegaconf

# Run reconstruction
echo ""
echo "Running command:"
echo "$CMD"
echo "=================================================================================="
echo ""

eval $CMD

EXITCODE=$?

# Show results
if [ $EXITCODE -eq 0 ]; then
    echo ""
    echo "=================================================================================="
    echo "RECONSTRUCTION COMPLETED SUCCESSFULLY"
    echo "=================================================================================="
    echo "Output directory: ${OUTPUT_FOLDER}"

    if [ -d "$OUTPUT_FOLDER" ]; then
        echo ""
        echo "Reconstructed images:"
        ls -lh "${OUTPUT_FOLDER}"/*.png 2>/dev/null | head -20 || echo "No PNG images found"

        # Count images
        IMAGE_COUNT=$(ls -1 "${OUTPUT_FOLDER}"/*.png 2>/dev/null | wc -l)
        echo ""
        echo "Total reconstructed images: ${IMAGE_COUNT}"

        # Show disk usage
        echo ""
        echo "Disk usage:"
        du -sh "${OUTPUT_FOLDER}"
    fi

    echo ""
    echo "You can now run metrics calculation:"
    echo "  python calculate_metrics.py --comparison-path ${OUTPUT_FOLDER}"

else
    echo ""
    echo "=================================================================================="
    echo "RECONSTRUCTION FAILED WITH EXIT CODE: $EXITCODE"
    echo "=================================================================================="
    echo "Check the error log for details:"
    echo "  /iopsstor/scratch/cscs/${USER}/benchmark-image-tokenizer/logs/batch_recon_${SLURM_JOB_ID}.err"
    exit $EXITCODE
fi

echo ""
echo "=================================================================================="
echo "Job completed at $(date)"
echo "=================================================================================="
