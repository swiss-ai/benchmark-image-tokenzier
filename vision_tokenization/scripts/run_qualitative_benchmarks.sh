#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=qual-bench
#SBATCH --environment=sgl_pdm_env
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=normal
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:30:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/benchmark-image-tokenizer/vision_tokenization/logs/qualitative_bench%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/benchmark-image-tokenizer/vision_tokenization/logs/qualitative_bench%j.err

PROJECT_ROOT="/iopsstor/scratch/cscs/${USER}/benchmark-image-tokenizer"

# Default values
DEFAULT_MODEL_PATH="/iopsstor/scratch/cscs/rkreft/Megatron-LM/logs/Meg-Runs/image-extension/llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-USR-MASKED-NOTIMG-FIXES-RPAD/HF"
DEFAULT_TOKENIZER_PATH="/capstor/store/cscs/swissai/infra01/MLLM/llama3_vision_instruct_emu3_tokenizer"

# Parse command-line arguments
EXPERIMENT_NAME=""
MODEL_PATH="$DEFAULT_MODEL_PATH"
TOKENIZER_PATH="$DEFAULT_TOKENIZER_PATH"
APPLY_CHAT_TEMPLATE=""
CHAT_FORMAT=""
IMAGE_COMPLETION=""
COMPLETION_PERCENTAGES=""
STRICT_ROW_COUNT=""
DEBUG=""

usage() {
    echo "Usage: $0 --experiment_name <name> [OPTIONS]"
    echo ""
    echo "Required arguments:"
    echo "  --experiment_name <name>    Name of the experiment"
    echo ""
    echo "Optional arguments:"
    echo "  --model_path <path>         Path to HF model (default: $DEFAULT_MODEL_PATH)"
    echo "  --tokenizer_path <path>     Path to tokenizer (default: $DEFAULT_TOKENIZER_PATH)"
    echo "  --chat-format <format>      Chat format to use (e.g., llama, apertus)"
    echo "  --no_chat_template          Do not apply chat template to prompts"
    echo ""
    echo "Image completion benchmark options:"
    echo "  --image-completion                      Run image completion benchmark instead of VLM Q&A"
    echo "  --completion-percentages <percentages>  Comma-separated completion percentages (default: 20,40,60,80)"
    echo "  --strict-row-count                      Require exact row count match for validity"
    echo "  --debug                                 Enable debug mode (prints tokens for first 3 samples)"
    echo ""
    echo "Example - Image completion benchmark with llama3 emu3:"
    echo "  sbatch vision_tokenization/scripts/run_qualitative_benchmarks.sh --experiment_name completion_30_60_80_90_20260122 --model_path /capstor/store/cscs/swissai/infra01/vision-ckpts/llama3-3b-15n-8192sl-120gbsz-0.9i-0.1t-long-run-0063500/HF --image-completion --completion-percentages 80,90 --tokenizer_path /capstor/store/cscs/swissai/infra01/MLLM/llama3_emu3_tokenizer"
    echo ""
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --experiment_name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --tokenizer_path)
            TOKENIZER_PATH="$2"
            shift 2
            ;;
        --chat-format)
            CHAT_FORMAT="$2"
            shift 2
            ;;
        --no_chat_template)
            APPLY_CHAT_TEMPLATE="--no_chat_template"
            shift
            ;;
        --image-completion)
            IMAGE_COMPLETION="--image-completion"
            shift
            ;;
        --completion-percentages)
            COMPLETION_PERCENTAGES="$2"
            shift 2
            ;;
        --strict-row-count)
            STRICT_ROW_COUNT="--strict-row-count"
            shift
            ;;
        --debug)
            DEBUG="--debug"
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            ;;
    esac
done

# Check required arguments
if [ -z "$EXPERIMENT_NAME" ]; then
    echo "Error: --experiment_name is required"
    usage
fi

cd /iopsstor/scratch/cscs/$USER/benchmark-image-tokenizer/vision_tokenization/qualitative_benchmark || exit
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

echo "=================================================================================="
if [ -n "$IMAGE_COMPLETION" ]; then
    echo "Running IMAGE COMPLETION benchmark with:"
    echo "  Experiment name:        $EXPERIMENT_NAME"
    echo "  Model path:             $MODEL_PATH"
    echo "  Tokenizer path:         $TOKENIZER_PATH"
    echo "  Completion percentages: $([ -z "$COMPLETION_PERCENTAGES" ] && echo "20,40,60,80 (default)" || echo "$COMPLETION_PERCENTAGES")"
    echo "  Strict row count:       $([ -z "$STRICT_ROW_COUNT" ] && echo "No" || echo "Yes")"
    echo "  Debug mode:             $([ -z "$DEBUG" ] && echo "No" || echo "Yes")"
else
    echo "Running VLM Q&A benchmark with:"
    echo "  Experiment name:     $EXPERIMENT_NAME"
    echo "  Model path:          $MODEL_PATH"
    echo "  Tokenizer path:      $TOKENIZER_PATH"
    echo "  Chat format:         $([ -z "$CHAT_FORMAT" ] && echo "None" || echo "$CHAT_FORMAT")"
    echo "  Apply chat template: $([ -z "$APPLY_CHAT_TEMPLATE" ] && echo "Yes" || echo "No")"
fi
echo "=================================================================================="

python vlm_benchmark.py --tokenizer_path "$TOKENIZER_PATH" \
                        --model_path "$MODEL_PATH" \
                        --experiment_name "$EXPERIMENT_NAME" \
                        $([ -n "$CHAT_FORMAT" ] && echo "--chat-format $CHAT_FORMAT") \
                        $([ -n "$COMPLETION_PERCENTAGES" ] && echo "--completion-percentages $COMPLETION_PERCENTAGES") \
                        $APPLY_CHAT_TEMPLATE \
                        $IMAGE_COMPLETION \
                        $STRICT_ROW_COUNT \
                        $DEBUG

echo "=================================================================================="
echo "Job completed at $(date)"
echo "=================================================================================="