#!/bin/bash
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --job-name=chartverse-sft
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/chartverse_1_8m_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/chartverse_1_8m_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=40
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --partition=normal
#SBATCH --exclusive

set -e

# srun's switch/hpe_slingshot plugin links against libjson-c.so.5 which is
# not on the container's default loader path. The wheelhouse ships a copy.
export LD_LIBRARY_PATH=/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse:${LD_LIBRARY_PATH:-}

export NUM_GPUS=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))

echo "=========================================="
echo "ChartVerse-SFT-1.8M Tokenization (with CoT)"
echo "  GPUs: ${NUM_GPUS}, Nodes: ${SLURM_NNODES}"
echo "=========================================="

mkdir -p /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs

srun -u --kill-on-bad-exit=0 bash -c "
    set -e
    uv pip install --python /opt/venv/bin/python3 orjson

    export PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/apertus/Megatron-LM:/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier:\$PYTHONPATH

    cd /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier

    python -m vision_tokenization.tokenize \
        mode=sft dataset=sft/chartverse_1_8m \
        num_gpus=${NUM_GPUS} \
        wandb.enabled=true \
        resume=true
"

echo ""
echo "=========================================="
echo "Tokenization completed!"
echo "After completion, run merge inline (no --strip-thinking needed):"
echo ""
echo "  python -m vision_tokenization.pipeline.output.merge \\"
echo "    /capstor/store/cscs/swissai/infra01/vision-datasets/tokenized/sft/chartverse_1_8m \\"
echo ""
echo ""
echo "With seqlen_threshold=8192, the merge produces TWO output pairs:"
echo "  stage2 (≤8K tokens) → merged.bin/idx       → Apertus1p5_sft_tokenized/chartverse_1_8m.bin/idx"
echo "  lct    (>8K tokens) → merged_lct.bin/idx   → Apertus1p5_sft_lct_tokenized/chartverse_1_8m.bin/idx"
echo "=========================================="
