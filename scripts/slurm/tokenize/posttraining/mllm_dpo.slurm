#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=mllm-dpo-store
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/mllm_dpo_store_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/mllm_dpo_store_%j.err
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --partition=debug
#SBATCH --no-requeue

# 1 node / 1 GPU: mllm-dpo is a single tiny shard (5,182 pairs), so one
# single-rank posttraining store fits in one process. Larger preference sets
# (e.g. MMPR-v1.2, 8 shards) are sharded and packed onto multiple GPUs — see
# scripts/slurm/tokenize/posttraining/mmpr_v1_2.slurm.

set -e
export LD_LIBRARY_PATH=/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse:${LD_LIBRARY_PATH:-}

echo "=========================================="
echo "mllm-dpo (5,182 pairs) -> views+media preference store (Part C1)"
echo "  mode=posttraining task=preference, single-rank, 1 GPU"
echo "  out: .../vision-datasets/tokenized/preference/mllm_dpo"
echo "=========================================="

srun -u --kill-on-bad-exit=0 bash -c "
    set -e
    export PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier:\$PYTHONPATH
    cd /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier
    python -m vision_tokenization.tokenize \
        mode=posttraining dataset=posttraining/mllm_dpo \
        num_gpus=1 \
        wandb.enabled=false
"

echo "Gate 2:"
srun -u --overlap bash -c "
    export PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier:\$PYTHONPATH
    cd /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier
    python vision_tokenization/tests/alignment/check_store.py \
        /capstor/store/cscs/swissai/infra01/vision-datasets/tokenized/preference/mllm_dpo
"
