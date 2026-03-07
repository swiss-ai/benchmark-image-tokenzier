#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=qual-bench
#SBATCH --environment=sgl_pdm_env
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=normal
#SBATCH --reservation=PA-2338-RL
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:30:00
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
VISION_TOKENIZER_TYPE=""
VISION_TOKENIZER_PATH=""
INFERENCER_TYPE=""
CAPTIONING=""
CAPTION_INIT_PHRASE=""
GREEDY=""
PROMPT_BUILDER=""
NO_KV_CACHE=""

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
    echo "  --prompt-builder <name>     Custom prompt builder bypassing chat template (e.g., emu3)"
    echo ""
    echo "Vision tokenizer options:"
    echo "  --vision-tokenizer-type <type>  Vision tokenizer type: emu3, emu3.5, emu3.5-ibq (default: emu3)"
    echo "  --vision-tokenizer-path <path>  Path to vision tokenizer model (required for emu3.5)"
    echo ""
    echo "Inference backend options:"
    echo "  --inferencer-type <type>        Inference backend: vllm (faster) or hf (HuggingFace, more compatible - needed for apertus) (default: vllm)"
    echo "  --greedy                        Use greedy decoding (temperature=0, top_p=1.0)"
    echo "  --no-kv-cache                   Disable KV caching (HF backend only, slower but fixes Emu3 compatibility)"
    echo ""
    echo "Image completion benchmark options:"
    echo "  --image-completion                      Run image completion benchmark instead of VLM Q&A"
    echo "  --completion-percentages <percentages>  Comma-separated completion percentages (default: 20,40,60,80)"
    echo "  --strict-row-count                      Require exact row count match for validity"
    echo "  --debug                                 Enable debug mode (prints tokens for first 3 samples)"
    echo ""
    echo "Captioning benchmark options:"
    echo "  --captioning                            Run captioning benchmark instead of VLM Q&A"
    echo "  --caption-init-phrase <phrase>          Optional init phrase for caption generation (e.g., 'The image shows')"
    echo ""
    echo "Example - Image completion benchmark with llama3 emu3:"
    echo "  sbatch vision_tokenization/scripts/run_qualitative_benchmarks.sh --experiment_name completion_30_60_80_90_llama3_emu3 --model_path /capstor/store/cscs/swissai/infra01/vision-ckpts/llama3-3b-15n-8192sl-120gbsz-0.9i-0.1t-long-run-0063500/HF --image-completion --completion-percentages 80,90 --tokenizer_path /capstor/store/cscs/swissai/infra01/MLLM/llama3_emu3_tokenizer"
    echo ""
    echo "Example - Image completion benchmark with apertus8b emu3.5:"
    echo "  sbatch vision_tokenization/scripts/run_qualitative_benchmarks.sh --experiment_name completion_30_60_80_90_apertus_emu3_5 --model_path /users/rkreft/megatron-repo/logs/Meg-Runs/apertus_image_extension/apertus-8b-img-pretrain-64nodes-gbs1024-mbs1-steps7003-img0.9-text0.1-seqlen8192/HF --vision-tokenizer-type emu3.5 --image-completion --completion-percentages 30,60,80,90 --tokenizer_path /capstor/store/cscs/swissai/infra01/MLLM/apertus_emu3.5_tokenizer --vision-tokenizer-type emu3.5 --inferencer-type hf"
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
        --vision-tokenizer-type)
            VISION_TOKENIZER_TYPE="$2"
            shift 2
            ;;
        --vision-tokenizer-path)
            VISION_TOKENIZER_PATH="$2"
            shift 2
            ;;
        --inferencer-type)
            INFERENCER_TYPE="$2"
            shift 2
            ;;
        --captioning)
            CAPTIONING="--captioning"
            shift
            ;;
        --caption-init-phrase)
            CAPTION_INIT_PHRASE="$2"
            shift 2
            ;;
        --prompt-builder)
            PROMPT_BUILDER="$2"
            shift 2
            ;;
        --greedy)
            GREEDY="--greedy"
            shift
            ;;
        --no-kv-cache)
            NO_KV_CACHE="--no-kv-cache"
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


ensure_packages() {
    echo "Make sure packages are installed...."
    pip install -U "transformers>=4.56,<5.0.0" #"vllm>=0.14.0" "numpy<2"
    pip install lpips scikit-image
}

cd /iopsstor/scratch/cscs/$USER/benchmark-image-tokenizer/vision_tokenization/qualitative_benchmark || exit
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

echo "=================================================================================="
if [ -n "$IMAGE_COMPLETION" ]; then
    echo "Running IMAGE COMPLETION benchmark with:"
    echo "  Experiment name:        $EXPERIMENT_NAME"
    echo "  Model path:             $MODEL_PATH"
    echo "  Tokenizer path:         $TOKENIZER_PATH"
    echo "  Inferencer type:        $([ -z "$INFERENCER_TYPE" ] && echo "vllm (default)" || echo "$INFERENCER_TYPE")"
    echo "  Vision tokenizer type:  $([ -z "$VISION_TOKENIZER_TYPE" ] && echo "emu3 (default)" || echo "$VISION_TOKENIZER_TYPE")"
    echo "  Vision tokenizer path:  $([ -z "$VISION_TOKENIZER_PATH" ] && echo "None" || echo "$VISION_TOKENIZER_PATH")"
    echo "  Prompt builder:         $([ -z "$PROMPT_BUILDER" ] && echo "None" || echo "$PROMPT_BUILDER")"
    echo "  Completion percentages: $([ -z "$COMPLETION_PERCENTAGES" ] && echo "20,40,60,80 (default)" || echo "$COMPLETION_PERCENTAGES")"
    echo "  Strict row count:       $([ -z "$STRICT_ROW_COUNT" ] && echo "No" || echo "Yes")"
    echo "  Greedy decoding:        $([ -z "$GREEDY" ] && echo "No" || echo "Yes")"
    echo "  No KV cache:            $([ -z "$NO_KV_CACHE" ] && echo "No" || echo "Yes")"
    echo "  Debug mode:             $([ -z "$DEBUG" ] && echo "No" || echo "Yes")"
elif [ -n "$CAPTIONING" ]; then
    echo "Running CAPTIONING benchmark with:"
    echo "  Experiment name:        $EXPERIMENT_NAME"
    echo "  Model path:             $MODEL_PATH"
    echo "  Tokenizer path:         $TOKENIZER_PATH"
    echo "  Inferencer type:        $([ -z "$INFERENCER_TYPE" ] && echo "vllm (default)" || echo "$INFERENCER_TYPE")"
    echo "  Vision tokenizer type:  $([ -z "$VISION_TOKENIZER_TYPE" ] && echo "emu3 (default)" || echo "$VISION_TOKENIZER_TYPE")"
    echo "  Vision tokenizer path:  $([ -z "$VISION_TOKENIZER_PATH" ] && echo "None" || echo "$VISION_TOKENIZER_PATH")"
    echo "  Prompt builder:         $([ -z "$PROMPT_BUILDER" ] && echo "None" || echo "$PROMPT_BUILDER")"
    echo "  Caption init phrase:    $([ -z "$CAPTION_INIT_PHRASE" ] && echo "None" || echo "$CAPTION_INIT_PHRASE")"
else
    echo "Running VLM Q&A benchmark with:"
    echo "  Experiment name:        $EXPERIMENT_NAME"
    echo "  Model path:             $MODEL_PATH"
    echo "  Tokenizer path:         $TOKENIZER_PATH"
    echo "  Inferencer type:        $([ -z "$INFERENCER_TYPE" ] && echo "vllm (default)" || echo "$INFERENCER_TYPE")"
    echo "  Vision tokenizer type:  $([ -z "$VISION_TOKENIZER_TYPE" ] && echo "emu3 (default)" || echo "$VISION_TOKENIZER_TYPE")"
    echo "  Vision tokenizer path:  $([ -z "$VISION_TOKENIZER_PATH" ] && echo "None" || echo "$VISION_TOKENIZER_PATH")"
    echo "  Prompt builder:         $([ -z "$PROMPT_BUILDER" ] && echo "None" || echo "$PROMPT_BUILDER")"
    echo "  Chat format:            $([ -z "$CHAT_FORMAT" ] && echo "None" || echo "$CHAT_FORMAT")"
    echo "  Apply chat template:    $([ -z "$APPLY_CHAT_TEMPLATE" ] && echo "Yes" || echo "No")"
fi
echo "=================================================================================="

# make sure compatible packages are installed
# Note: transformers 5.0.0 breaks Emu3VisionTokenizer loading, so pin to <5.0.0
ensure_packages

python vlm_benchmark.py --tokenizer_path "$TOKENIZER_PATH" \
                        --model_path "$MODEL_PATH" \
                        --experiment_name "$EXPERIMENT_NAME" \
                        $([ -n "$CHAT_FORMAT" ] && echo "--chat-format $CHAT_FORMAT") \
                        $([ -n "$PROMPT_BUILDER" ] && echo "--prompt-builder $PROMPT_BUILDER") \
                        $([ -n "$COMPLETION_PERCENTAGES" ] && echo "--completion-percentages $COMPLETION_PERCENTAGES") \
                        $([ -n "$VISION_TOKENIZER_TYPE" ] && echo "--vision-tokenizer-type $VISION_TOKENIZER_TYPE") \
                        $([ -n "$VISION_TOKENIZER_PATH" ] && echo "--vision-tokenizer-path $VISION_TOKENIZER_PATH") \
                        $([ -n "$INFERENCER_TYPE" ] && echo "--inferencer-type $INFERENCER_TYPE") \
                        $([ -n "$CAPTION_INIT_PHRASE" ] && echo "--caption-init-phrase \"$CAPTION_INIT_PHRASE\"") \
                        $APPLY_CHAT_TEMPLATE \
                        $IMAGE_COMPLETION \
                        $STRICT_ROW_COUNT \
                        $DEBUG \
                        $CAPTIONING \
                        $GREEDY \
                        $NO_KV_CACHE

echo "=================================================================================="
echo "Job completed at $(date)"
echo "=================================================================================="