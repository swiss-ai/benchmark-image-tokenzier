#!/bin/bash
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5
#SBATCH --job-name=dailymed-spl-tokenize
#SBATCH --environment=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/scripts/envs/nemo_25_11.toml
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/dailymed_spl_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/dailymed_spl_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --partition=normal
#SBATCH --exclusive

set -e

echo "=========================================="
echo "DailyMed SPL (interleaved2, deduped) — interleave tokenization"
echo "=========================================="
echo "Job ID:   ${SLURM_JOB_ID}"
echo "Nodes:    ${SLURM_NNODES}"
export NUM_GPUS=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))
echo "GPUs:     ${NUM_GPUS}"
echo
echo "Source:   /capstor/store/cscs/swissai/infra01/medical-datasets/raw/dailymed_spl/interleaved2/"
echo "Manifest: /capstor/store/cscs/swissai/infra01/vision-datasets/manifest/dailymed_spl_interleaved2/manifest.parquet"
echo "Target:   /capstor/store/cscs/swissai/infra01/vision-datasets/tokenized/interleave/dailymed_spl/"
echo "Plan:     88,287 docs, 162,557 image components, ~466M image tokens"
echo "Layout:   stage2/ (≤8K) + lct/ (>8K) split at seqlen_threshold=8192"
echo "=========================================="

mkdir -p /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs

srun -u --kill-on-bad-exit=0 bash -c "
    set -e
    # NeMo 25.11 ships pyarrow 20.0.0, which can't materialize chunked
    # list<large_binary> columns (the dedup tool wrote interleaved2 with that
    # type). Bumping to >=22 fixes ArrowNotImplementedError on shard reads.
    uv pip install --python /opt/venv/bin/python3 orjson --upgrade \"pyarrow>=22\"

    export PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/apertus/Megatron-LM:/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier:\$PYTHONPATH

    cd /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier

    python -m vision_tokenization.tokenize \
        mode=interleave dataset=dailymed_spl \
        num_gpus=${NUM_GPUS} \
        wandb.enabled=true \
        resume=true
"

echo ""
echo "=========================================="
echo "Tokenization completed!"
echo "=========================================="
