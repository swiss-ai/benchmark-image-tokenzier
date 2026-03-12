#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=pmc-oa-tokenize
#SBATCH --environment=nemo
#SBATCH --output=logs/pmc_oa_%j.out
#SBATCH --error=logs/pmc_oa_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --partition=normal
#SBATCH --exclusive

set -e

echo "=========================================="
echo "PMC-OA Image2Text Tokenization"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname -s)"
echo "GPUs: 4"
echo "=========================================="

cd /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier
mkdir -p logs

srun torchrun --nproc_per_node=4 \
    -m vision_tokenization.tokenize \
    mode=image2text dataset=pmc_oa \
    num_gpus=4

echo ""
echo "=========================================="
echo "Tokenization completed!"
echo "=========================================="
