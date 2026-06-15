#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=mllm-dpo-encode
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/mllm_dpo_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/mllm_dpo_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --partition=debug
#SBATCH --no-requeue

# mllm-dpo alignment ENCODE phase (single-rank, 1 GPU). mllm-dpo is a tiny
# single shard (5,182 pairs), so one rank encodes all media in one process.
#
# THREE-PHASE FLOW — scan + merge run INLINE on the head node (pa-24 venv),
# only the encode is a slurm job:
#
#   1. SCAN (inline, torch-free):
#        /iopsstor/scratch/cscs/xyixuan/apertus/venvs/scan-pa21/bin/python \
#          -m vision_tokenization.alignment_scan \
#          mode=alignment dataset=alignment/mllm_dpo
#
#   2. ENCODE (this job):
#        env -i HOME="$HOME" USER="$USER" \
#          LD_LIBRARY_PATH=/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse \
#          /usr/bin/sbatch \
#          /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/scripts/slurm/tokenize/alignment/mllm_dpo.slurm
#
#   3. MERGE (inline, torch-free):
#        /iopsstor/scratch/cscs/xyixuan/apertus/venvs/scan-pa21/bin/python \
#          -m vision_tokenization.pipeline.output.alignment_merge \
#          /capstor/store/cscs/swissai/infra01/vision-datasets/tokenized/alignment/mllm_dpo

set -e
export LD_LIBRARY_PATH=/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse:${LD_LIBRARY_PATH:-}
export NUM_GPUS=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))
mkdir -p /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs

echo "=========================================="
echo "mllm-dpo ENCODE: ${NUM_GPUS} GPU (single-rank, 5,182 pairs)"
echo "  reads pre-built scan -> spills to .../alignment/mllm_dpo/_spill"
echo "=========================================="

srun -u --kill-on-bad-exit=0 bash -c "
    set -e
    uv pip install --python /opt/venv/bin/python3 orjson
    export PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier:\$PYTHONPATH
    cd /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier
    python -m vision_tokenization.tokenize \
        mode=alignment dataset=alignment/mllm_dpo \
        num_gpus=${NUM_GPUS} \
        wandb.enabled=false
"
echo "ENCODE done -> $(date). Now run the inline MERGE (pa-24 venv):"
echo "  /iopsstor/scratch/cscs/xyixuan/apertus/venvs/scan-pa21/bin/python \\"
echo "    -m vision_tokenization.pipeline.output.alignment_merge \\"
echo "    /capstor/store/cscs/swissai/infra01/vision-datasets/tokenized/alignment/mllm_dpo"
