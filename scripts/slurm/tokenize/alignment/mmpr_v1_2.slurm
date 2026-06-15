#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=mmpr-v1.2-encode
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/mmpr_v1_2_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/mmpr_v1_2_%j.err
#SBATCH --time=01:30:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --partition=normal
#SBATCH --exclusive
#SBATCH --no-requeue

# MMPR-v1.2 alignment ENCODE phase (multi-rank, 4x4 = 16 GPUs). Reads the
# pre-built scan and spills each rank's disjoint media slice via SpillBackend.
# Seal-at-end, no resume — each unique media is encoded by exactly one rank.
#
# THREE-PHASE FLOW — scan + merge run INLINE on the head node (pa-24 venv),
# only the encode is a slurm job:
#
#   1. SCAN (inline, torch-free):
#        /iopsstor/scratch/cscs/xyixuan/apertus/venvs/scan-pa21/bin/python \
#          -m vision_tokenization.alignment_scan \
#          mode=alignment dataset=alignment/mmpr_v1_2
#
#   2. ENCODE (this job):
#        env -i HOME="$HOME" USER="$USER" \
#          LD_LIBRARY_PATH=/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse \
#          /usr/bin/sbatch \
#          /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/scripts/slurm/tokenize/alignment/mmpr_v1_2.slurm
#
#   3. MERGE (inline, torch-free):
#        /iopsstor/scratch/cscs/xyixuan/apertus/venvs/scan-pa21/bin/python \
#          -m vision_tokenization.pipeline.output.alignment_merge \
#          /capstor/store/cscs/swissai/infra01/vision-datasets/tokenized/alignment/mmpr_v1_2

set -e
export LD_LIBRARY_PATH=/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse:${LD_LIBRARY_PATH:-}
export NUM_GPUS=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))
mkdir -p /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs

echo "=========================================="
echo "MMPR-v1.2 ENCODE: ${NUM_GPUS} GPUs across ${SLURM_NNODES} nodes"
echo "  reads pre-built scan -> spills to .../alignment/mmpr_v1_2/_spill"
echo "=========================================="

srun -u --kill-on-bad-exit=0 bash -c "
    set -e
    uv pip install --python /opt/venv/bin/python3 orjson
    export PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier:\$PYTHONPATH
    cd /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier
    python -m vision_tokenization.tokenize \
        mode=alignment dataset=alignment/mmpr_v1_2 \
        num_gpus=${NUM_GPUS} \
        wandb.enabled=true wandb.group=mmpr_v1_2 wandb.name=mmpr_v1_2_encode
"
echo "ENCODE done -> $(date). Now run the inline MERGE (pa-24 venv):"
echo "  /iopsstor/scratch/cscs/xyixuan/apertus/venvs/scan-pa21/bin/python \\"
echo "    -m vision_tokenization.pipeline.output.alignment_merge \\"
echo "    /capstor/store/cscs/swissai/infra01/vision-datasets/tokenized/alignment/mmpr_v1_2"
