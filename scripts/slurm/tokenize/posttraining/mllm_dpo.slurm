#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=mllm-dpo-store
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/mllm_dpo_store_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/mllm_dpo_store_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --partition=debug
#SBATCH --no-requeue

set -e
export LD_LIBRARY_PATH=/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse:${LD_LIBRARY_PATH:-}

echo "=========================================="
echo "mllm-dpo (5,182 pairs) -> views+media alignment store (Part C1)"
echo "  mode=posttraining task=preference, single-rank, 1 GPU"
echo "  out: .../alignment-tokenized/mllm_dpo_views_media/alignment/mllm_dpo"
echo "=========================================="

srun -u --kill-on-bad-exit=0 bash -c "
    set -e
    export PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier:\$PYTHONPATH
    cd /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier
    python -m vision_tokenization.tokenize \
        mode=posttraining dataset=posttraining/mllm_dpo \
        num_gpus=1
"

echo "Gate 2:"
srun -u --overlap bash -c "
    export PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier:\$PYTHONPATH
    cd /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier
    python vision_tokenization/tests/alignment/check_store.py \
        /capstor/store/cscs/swissai/infra01/vision-datasets/alignment-tokenized/mllm_dpo_views_media/alignment/mllm_dpo
"
