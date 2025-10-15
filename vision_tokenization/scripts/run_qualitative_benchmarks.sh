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

cd /iopsstor/scratch/cscs/$USER/benchmark-image-tokenizer/vision_tokenization/qualitative_benchmark || exit

EXPERIMENT_NAME="test"
MODEL_PATH="/iopsstor/scratch/cscs/rkreft/Megatron-LM/logs/Meg-Runs/image-extension/llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-USR-MASKED-NOTIMG-FIXES-RPAD/HF"
TOKENIZER_PATH="/capstor/store/cscs/swissai/infra01/MLLM/llama3_vision_instruct_emu3_tokenizer"

python vlm_benchmark.py --tokenizer_path $TOKENIZER_PATH \
                        --model_path $MODEL_PATH \
                        --experiment_name $EXPERIMENT_NAME

echo "=================================================================================="
echo "Job completed at $(date)"
echo "=================================================================================="