#!/bin/bash
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5
#SBATCH --job-name=visual-genome-tokenize
#SBATCH --environment=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/scripts/envs/nemo_25_11.toml
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/visual_genome_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/visual_genome_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --partition=normal
#SBATCH --exclusive

set -e

echo "=========================================="
echo "Visual Genome — image2text tokenization (BLIP3-Grounding format)"
echo "=========================================="
echo "Job ID:   ${SLURM_JOB_ID}"
echo "Nodes:    ${SLURM_NNODES}"
export NUM_GPUS=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))
echo "GPUs:     ${NUM_GPUS}"
echo
echo "Source:   /capstor/store/cscs/swissai/infra01/vision-datasets/raw/stage2/visual_genome/parquet_blip3_clean/  (52 shards, 14.35 GB)"
echo "Manifest: /capstor/store/cscs/swissai/infra01/vision-datasets/manifest/visual_genome/manifest.parquet  (102,005 rows)"
echo "Target:   /capstor/store/cscs/swissai/infra01/vision-datasets/tokenized/cooldown/image2text/visual_genome/"
echo "Plan:     102,004 docs, ~88M predicted image tokens"
echo "Note:     RefCOCO/+/g eval-image-id overlaps EXCLUDED (6,068 dropped, 0% leak)"
echo "=========================================="

mkdir -p /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs

srun -u --kill-on-bad-exit=0 bash -c "
    set -e
    uv pip install --python /opt/venv/bin/python3 orjson

    export PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/apertus/Megatron-LM:/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier:\$PYTHONPATH

    cd /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier

    python -m vision_tokenization.tokenize \
        mode=image2text dataset=visual_genome \
        num_gpus=${NUM_GPUS} \
        wandb.enabled=true \
        resume=true
"

echo ""
echo "=========================================="
echo "Tokenization completed!"
echo "=========================================="
