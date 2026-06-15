#!/bin/bash
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --job-name=nemotron-v3-sft-tokenize
#SBATCH --environment=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/scripts/envs/nemo_25_11.toml
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/nemotron_image_training_v3_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/nemotron_image_training_v3_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=64
#SBATCH --exclude=nid006137
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --partition=normal
#SBATCH --exclusive

set -e

echo "=========================================="
echo "Nemotron-Image-Training-v3 — multi-image SFT tokenization"
echo "=========================================="
echo "Job ID:   ${SLURM_JOB_ID}"
echo "Nodes:    ${SLURM_NNODES}"
export NUM_GPUS=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))
echo "GPUs:     ${NUM_GPUS}"
echo
echo "Source:   /capstor/store/cscs/swissai/infra01/vision-datasets/raw/sft/nemotron_image_training_v3/swissai___Nemotron-Image-Training-v3/"
echo "Manifest: /capstor/store/cscs/swissai/infra01/vision-datasets/manifest/nemotron_image_training_v3/manifest.parquet"
echo "Target:   /capstor/store/cscs/swissai/infra01/vision-datasets/tokenized/sft/nemotron_image_training_v3/"
echo "Plan:     6,725,124 docs, 28,575,943 images, 4,334,990 batches, ~133.5B image tokens"
echo "Layout:   stage2/ (<=8K) + lct/ (>8K) split at seqlen_threshold=8192"
echo "Note:     <think> CoT preserved in assistant messages (STRIP_THINKING=false at merge)"
echo "=========================================="

mkdir -p /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs

srun -u --kill-on-bad-exit=0 bash -c "
    set -e
    # NeMo 25.11 ships pyarrow 20.0.0, which can't read nemotron-v3's
    # map<string,binary> images column (ArrowNotImplementedError on nested
    # chunked output). Bumping to >=22 lets the scanner's iter_batches
    # workaround read the row groups end-to-end.
    uv pip install --python /opt/venv/bin/python3 orjson --upgrade \"pyarrow>=22\"

    export PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/apertus/Megatron-LM:/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier:\$PYTHONPATH

    cd /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier

    python -m vision_tokenization.tokenize \
        mode=sft dataset=nemotron_image_training_v3 \
        num_gpus=${NUM_GPUS} \
        wandb.enabled=true \
        resume=true
"

echo ""
echo "=========================================="
echo "Tokenization completed!"
echo "=========================================="
