#!/bin/bash
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5
#SBATCH --job-name=google-rsrcc-tokenize
#SBATCH --environment=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/scripts/envs/nemo_25_11.toml
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/google_rsrcc_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/google_rsrcc_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --partition=normal
#SBATCH --exclusive

set -e

# srun's switch/hpe_slingshot plugin links against libjson-c.so.5, which is
# not on the container's default loader path. The wheelhouse ships a copy.
export LD_LIBRARY_PATH=/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse:${LD_LIBRARY_PATH:-}

echo "=========================================="
echo "google/RSRCC Multi-Image SFT Tokenization (with CoT)"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   $(hostname -s)"
export NUM_GPUS=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))
echo "GPUs:   ${NUM_GPUS}"
echo "=========================================="

mkdir -p /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs

srun -u --kill-on-bad-exit=0 bash -c "
    set -e
    uv pip install --python /opt/venv/bin/python3 orjson

    export PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/apertus/Megatron-LM:/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier:\$PYTHONPATH

    cd /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier

    python -m vision_tokenization.tokenize \
        mode=sft dataset=sft/google_rsrcc \
        num_gpus=${NUM_GPUS} \
        wandb.enabled=true \
        resume=true
"

echo ""
echo "=========================================="
echo "Tokenization completed!"
echo "After completion, run merge --strip-thinking to produce both"
echo "with-CoT (merged.bin/idx) and no-CoT (merged_no_cot.bin/idx) variants:"
echo ""
echo "  python -m vision_tokenization.pipeline.output.merge \\"
echo "    /capstor/store/cscs/swissai/infra01/vision-datasets/tokenized/sft/google_rsrcc \\"
echo " --strip-thinking --resolve-thinking-ids"
echo "=========================================="
