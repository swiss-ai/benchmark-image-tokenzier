#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=mmpr-v1.2
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/mmpr_v1_2_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/mmpr_v1_2_%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --partition=normal
#SBATCH --exclusive
#SBATCH --no-requeue

# MMPR-v1.2 -> one preference media store from sized processed parquet parts.
# alignment is single-rank per store; scanner reads data/part-*.parquet and
# dedups globally before the 1-GPU encode loop. alignment is seal-at-end with
# NO resume.
#
# Submit from a clean env:
#   env -i HOME="$HOME" USER="$USER" /usr/bin/sbatch \
#       /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/scripts/slurm/tokenize/alignment/mmpr_v1_2.slurm

set -e
export LD_LIBRARY_PATH=/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse:${LD_LIBRARY_PATH:-}

echo "=========================================="
echo "MMPR-v1.2 -> one views+media preference store"
echo "  mode=alignment task=preference; input_pattern=data/part-*.parquet"
echo "  out: .../vision-datasets/tokenized/alignment/mmpr_v1_2"
echo "=========================================="

srun -u --mpi=pmix --environment=nemo_26_02 bash -c '
    set -e
    unset LD_PRELOAD || true
    python -c "import orjson" 2>/dev/null || uv pip install --python "$(command -v python)" --break-system-packages orjson
    TOKENIZERS_0222=/iopsstor/scratch/cscs/xyixuan/.cache/uv/archive-v0/1CtSrhkL_p104lhucBDaz
    export PYTHONPATH=${TOKENIZERS_0222}:/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier:${PYTHONPATH:-}
    export WANDB_DIR=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs
    cd /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier
    export RANK=0 WORLD_SIZE=1
    python - <<'"'"'PY'"'"'
import tokenizers, orjson
if tokenizers.__version__ != "0.22.2":
    raise SystemExit(f"expected tokenizers 0.22.2, got {tokenizers.__version__}")
print("tokenizers", tokenizers.__version__)
print("orjson", orjson.__version__)
PY

    python -m vision_tokenization.tokenize \
        mode=alignment dataset=alignment/mmpr_v1_2 \
        num_gpus=1 \
        wandb.enabled=true wandb.group=mmpr_v1_2 wandb.name=mmpr_v1_2

    python vision_tokenization/tests/alignment/check_store.py \
        /capstor/store/cscs/swissai/infra01/vision-datasets/tokenized/alignment/mmpr_v1_2
'
echo "END TIME: $(date)"
