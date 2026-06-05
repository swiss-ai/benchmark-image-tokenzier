#!/bin/bash
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --job-name=llava-ov-perm-sft
#SBATCH --environment=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/scripts/envs/nemo_25_11.toml
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/llava_ov_permissive_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/llava_ov_permissive_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=128
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --partition=normal
#SBATCH --exclusive

set -e

export NUM_GPUS=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))

echo "=========================================="
echo "LLaVA-OV permissive SFT tokenize"
echo "  GPUs: ${NUM_GPUS}, Nodes: ${SLURM_NNODES}"
echo "  Manifest: /capstor/scratch/cscs/xyixuan/manifests/llava_ov_permissive/manifest.parquet (10,450,286 rows / 125 subsets)"
echo "=========================================="

mkdir -p /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs

srun -u --kill-on-bad-exit=0 bash -c "
    set -e
    uv pip install --python /opt/venv/bin/python3 orjson

    export PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/apertus/Megatron-LM:/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier:\$PYTHONPATH

    cd /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier

    python -m vision_tokenization.tokenize \
        mode=sft dataset=sft/llava_ov_permissive \
        num_gpus=${NUM_GPUS} \
        wandb.enabled=true \
        resume=true
"
