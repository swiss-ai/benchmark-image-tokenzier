#!/bin/bash
#SBATCH --account=a-infra01
#SBATCH --job-name=sft-chat-template
#SBATCH --environment=emu3
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=normal
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=285
#SBATCH --time=12:00:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/benchmark-image-tokenizer/vision_tokenization/logs/sft_chat_template_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/benchmark-image-tokenizer/vision_tokenization/logs/sft_chat_template_%j.err

# Script to run apply_sft_chat_template with configurable dataset loading
# Usage:
#   sbatch run_apply_sft_chat_template.slurm --dataset HuggingFaceM4/FineVision --dataset-config lrv_chart --tokenizer-path /path/to/tokenizer --output-dir /path/to/output
#   sbatch run_apply_sft_chat_template.slurm --dataset HuggingFaceM4/FineVision --dataset-config lrv_chart --tokenizer-path /path/to/tokenizer --output-dir /path/to/output --offline --dataset-load-method builder_load
#   sbatch run_apply_sft_chat_template.slurm --debug --dataset HuggingFaceM4/FineVision --dataset-config lrv_chart --tokenizer-path /path/to/tokenizer --output-dir /path/to/output

# Get the project root directory
PROJECT_ROOT="/iopsstor/scratch/cscs/${USER}/benchmark-image-tokenizer"

# Hub cache location
HF_HUB_CACHE="${HF_HUB_CACHE:-/capstor/store/cscs/swissai/infra01/vision-datasets/hf_hub_cache}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HUB_CACHE}}"

# Defaults
DEBUG_MODE=false
OFFLINE_MODE=false
DATASET=""
DATASET_CONFIG=""
DATASET_SPLIT="train"
TOKENIZER_PATH=""
OUTPUT_DIR=""
DATASET_LOAD_METHOD="default"
CONVERSATION_TRANSFORM=""
NUM_PROC=64
MAX_SAMPLES=""
OUTPUT_FORMAT="parquet"
MESSAGE_COLUMN="texts"
TRANSFORM_PARAMS=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --dataset-config)
            DATASET_CONFIG="$2"
            shift 2
            ;;
        --dataset-split)
            DATASET_SPLIT="$2"
            shift 2
            ;;
        --tokenizer-path)
            TOKENIZER_PATH="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --dataset-load-method)
            DATASET_LOAD_METHOD="$2"
            shift 2
            ;;
        --conversation-transform)
            CONVERSATION_TRANSFORM="$2"
            shift 2
            ;;
        --transform-params)
            TRANSFORM_PARAMS="$2"
            shift 2
            ;;
        --output-format)
            OUTPUT_FORMAT="$2"
            shift 2
            ;;
        --message-column)
            MESSAGE_COLUMN="$2"
            shift 2
            ;;
        --num-proc)
            NUM_PROC="$2"
            shift 2
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --debug)
            DEBUG_MODE=true
            shift
            ;;
        --offline)
            OFFLINE_MODE=true
            shift
            ;;
        -h|--help)
            echo "Usage: sbatch $0 --dataset <name> --tokenizer-path <path> --output-dir <path> [options]"
            echo ""
            echo "Required:"
            echo "  --dataset <name>              HuggingFace dataset name or path"
            echo "  --tokenizer-path <path>       Path to tokenizer for chat template"
            echo "  --output-dir <path>           Output directory for formatted files"
            echo ""
            echo "Options:"
            echo "  --dataset-config <name>       Dataset configuration/subset name"
            echo "  --dataset-split <split>       Dataset split (default: train)"
            echo "  --dataset-load-method <m>     'default' or 'builder_load' (default: default)"
            echo "  --conversation-transform <n>  Conversation transform name"
            echo "  --transform-params <json>     Transform params JSON string or @file"
            echo "  --output-format <fmt>         'parquet' or 'jsonl' (default: parquet)"
            echo "  --message-column <col>        Column containing messages (default: texts)"
            echo "  --num-proc <n>                Number of processes (default: 64)"
            echo "  --max-samples <n>             Max samples to process per config"
            echo "  --debug                       Debug mode (3 debug samples, max 1000 samples)"
            echo "  --offline                     Enable HF datasets offline mode"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            shift
            ;;
    esac
done

# Validate required arguments
if [ -z "$DATASET" ]; then
    echo "ERROR: --dataset is required"
    exit 1
fi
if [ -z "$TOKENIZER_PATH" ]; then
    echo "ERROR: --tokenizer-path is required"
    exit 1
fi
if [ -z "$OUTPUT_DIR" ]; then
    echo "ERROR: --output-dir is required"
    exit 1
fi

# Set up Python path
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

echo "=================================================================================="
echo "Apply SFT Chat Template"
echo "=================================================================================="
echo "Project root:        ${PROJECT_ROOT}"
echo "Dataset:             ${DATASET}"
echo "Dataset config:      ${DATASET_CONFIG:-<none>}"
echo "Dataset split:       ${DATASET_SPLIT}"
echo "Load method:         ${DATASET_LOAD_METHOD}"
echo "Tokenizer:           ${TOKENIZER_PATH}"
echo "Output dir:          ${OUTPUT_DIR}"
echo "Output format:       ${OUTPUT_FORMAT}"
echo "Conv. transform:     ${CONVERSATION_TRANSFORM:-<none>}"
echo "Num proc:            ${NUM_PROC}"
echo "Debug mode:          ${DEBUG_MODE}"
echo "Offline mode:        ${OFFLINE_MODE}"
echo "Job ID:              ${SLURM_JOB_ID}"
echo "HF HUB CACHE:        ${HF_HUB_CACHE}"
echo "Node:                $(hostname)"
echo "=================================================================================="

# Export HF cache env vars
export HF_HUB_CACHE
export HF_DATASETS_CACHE

# Enable offline mode if requested
if [ "$OFFLINE_MODE" = true ]; then
    export HF_DATASETS_OFFLINE=1
    echo "OFFLINE mode enabled - will only use cached datasets (HF_DATASETS_OFFLINE=1)"
fi

# Build command
CMD="python -m vision_tokenization.utils.apply_sft_chat_template"
CMD="${CMD} --dataset ${DATASET}"
CMD="${CMD} --split ${DATASET_SPLIT}"
CMD="${CMD} --tokenizer-path ${TOKENIZER_PATH}"
CMD="${CMD} --output-dir ${OUTPUT_DIR}"
CMD="${CMD} --output-format ${OUTPUT_FORMAT}"
CMD="${CMD} --message-column ${MESSAGE_COLUMN}"
CMD="${CMD} --num-proc ${NUM_PROC}"
CMD="${CMD} --dataset-load-method ${DATASET_LOAD_METHOD}"

if [ -n "$DATASET_CONFIG" ]; then
    CMD="${CMD} --config ${DATASET_CONFIG}"
fi

if [ -n "$HF_DATASETS_CACHE" ]; then
    CMD="${CMD} --cache-dir ${HF_DATASETS_CACHE}"
fi

if [ "$OFFLINE_MODE" = true ]; then
    CMD="${CMD} --offline"
fi

if [ -n "$CONVERSATION_TRANSFORM" ]; then
    CMD="${CMD} --conversation-transform ${CONVERSATION_TRANSFORM}"
fi

if [ -n "$TRANSFORM_PARAMS" ]; then
    CMD="${CMD} --transform-params '${TRANSFORM_PARAMS}'"
fi

if [ -n "$MAX_SAMPLES" ]; then
    CMD="${CMD} --max-samples ${MAX_SAMPLES}"
fi

# Debug mode overrides
if [ "$DEBUG_MODE" = true ]; then
    echo "Running in DEBUG mode"
    CMD="${CMD} --debug-samples 3"
    if [ -z "$MAX_SAMPLES" ]; then
        CMD="${CMD} --max-samples 1000"
    fi
fi

echo "=================================================================================="
echo "Running command:"
echo "  ${CMD}"
echo "=================================================================================="

eval ${CMD}

EXITCODE=$?

if [ $EXITCODE -eq 0 ]; then
    echo "=================================================================================="
    echo "SFT CHAT TEMPLATE APPLICATION COMPLETED SUCCESSFULLY"
    echo "=================================================================================="
    echo "Output directory: ${OUTPUT_DIR}"
    echo ""
    echo "Output files:"
    ls -lh "${OUTPUT_DIR}"/*.parquet "${OUTPUT_DIR}"/*.jsonl 2>/dev/null | head -20 || echo "No output files found in ${OUTPUT_DIR}"
    # Check subfolders too
    ls -lh "${OUTPUT_DIR}"/**/*.parquet "${OUTPUT_DIR}"/**/*.jsonl 2>/dev/null | head -20 || true
else
    echo "=================================================================================="
    echo "SFT CHAT TEMPLATE APPLICATION FAILED WITH EXIT CODE: $EXITCODE"
    echo "=================================================================================="
    exit $EXITCODE
fi

echo "=================================================================================="
echo "Job completed at $(date)"
echo "=================================================================================="
