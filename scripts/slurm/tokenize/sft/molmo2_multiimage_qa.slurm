#!/bin/bash
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --job-name=molmo2-mmqa-sft-tokenize
#SBATCH --environment=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/scripts/envs/nemo_25_11.toml
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/molmo2_multiimage_qa_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/molmo2_multiimage_qa_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --partition=normal
#SBATCH --exclusive

set -e

echo "=========================================="
echo "Molmo2-MultiImageQA — multi-image SFT tokenization"
echo "=========================================="
echo "Job ID:   ${SLURM_JOB_ID}"
echo "Nodes:    ${SLURM_NNODES}"
export NUM_GPUS=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))
echo "GPUs:     ${NUM_GPUS}"
echo
echo "Source:   /capstor/store/cscs/swissai/infra01/vision-datasets/processed/hf___allenai___Molmo2-MultiImageQA___doc_keyed/"
echo "Manifest: /capstor/store/cscs/swissai/infra01/vision-datasets/manifest/molmo2_multiimage_qa/manifest.parquet"
echo "Target:   /capstor/store/cscs/swissai/infra01/vision-datasets/tokenized/sft/molmo2_multiimage_qa/"
echo "Note:     joined doc-keyed (strict drop on any failed URL fetch, ~59% survive)"
echo "=========================================="

mkdir -p /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs

srun -u --kill-on-bad-exit=0 bash -c "
    set -e
    uv pip install --python /opt/venv/bin/python3 orjson

    export PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/apertus/Megatron-LM:/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier:\$PYTHONPATH

    cd /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier

    python -m vision_tokenization.tokenize \
        mode=sft dataset=molmo2_multiimage_qa \
        num_gpus=${NUM_GPUS} \
        wandb.enabled=true \
        resume=true
"

echo ""
echo "=========================================="
echo "Tokenization completed!"
echo "=========================================="
