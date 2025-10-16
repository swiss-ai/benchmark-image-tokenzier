#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=qual-bench
#SBATCH --environment=sgl_pdm_env
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=normal
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/benchmark-image-tokenizer/vision_tokenization/logs/qualitative_bench%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/benchmark-image-tokenizer/vision_tokenization/logs/qualitative_bench%j.err

# Default values
DEFAULT_MODEL_PATH="/iopsstor/scratch/cscs/rkreft/Megatron-LM/logs/Meg-Runs/image-extension/llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-USR-MASKED-NOTIMG-FIXES-RPAD/HF"
DEFAULT_TOKENIZER_PATH="/capstor/store/cscs/swissai/infra01/MLLM/llama3_vision_instruct_emu3_tokenizer"

# Parse command-line arguments
EXPERIMENT_NAME=""
MODEL_PATH="$DEFAULT_MODEL_PATH"
TOKENIZER_PATH="$DEFAULT_TOKENIZER_PATH"
APPLY_CHAT_TEMPLATE=""

usage() {
    echo "Usage: $0 --experiment_name <name> [--model_path <path>] [--tokenizer_path <path>] [--no_chat_template]"
    echo ""
    echo "Required arguments:"
    echo "  --experiment_name <name>    Name of the experiment"
    echo ""
    echo "Optional arguments:"
    echo "  --model_path <path>         Path to HF model (default: $DEFAULT_MODEL_PATH)"
    echo "  --tokenizer_path <path>     Path to tokenizer (default: $DEFAULT_TOKENIZER_PATH)"
    echo "  --no_chat_template          Do not apply chat template to prompts"
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
        --no_chat_template)
            APPLY_CHAT_TEMPLATE="--no_chat_template"
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

echo "=================================================================================="
echo "Running qualitative benchmark with:"
echo "  Experiment name: $EXPERIMENT_NAME"
echo "  Model path:      $MODEL_PATH"
echo "  Tokenizer path:  $TOKENIZER_PATH"
echo "  Apply chat template: $([ -z "$APPLY_CHAT_TEMPLATE" ] && echo "Yes" || echo "No")"
echo "=================================================================================="

python vlm_benchmark.py --tokenizer_path "$TOKENIZER_PATH" \
                        --model_path "$MODEL_PATH" \
                        --experiment_name "$EXPERIMENT_NAME" \
                        $APPLY_CHAT_TEMPLATE

echo "=================================================================================="
echo "Job completed at $(date)"
echo "=================================================================================="