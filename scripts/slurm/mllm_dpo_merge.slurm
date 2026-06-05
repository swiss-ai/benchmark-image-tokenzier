#!/bin/bash
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --job-name=mllm-dpo-merge
#SBATCH --output=/iopsstor/scratch/cscs/rkreft/mllm-dpo/logs/mllm_dpo_merge_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/rkreft/mllm-dpo/logs/mllm_dpo_merge_%j.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=16
#SBATCH --partition=normal
#
# Cross-rank merge for ONE variant, with provenance:
#   merged.{bin,idx} + merged.src.npy + merged.provenance.parquet
#
#   VARIANT=accepted EXPECTED_RANKS=4 sbatch scripts/slurm/mllm_dpo_merge.slurm
#   VARIANT=rejected EXPECTED_RANKS=4 sbatch scripts/slurm/mllm_dpo_merge.slurm
#
# ──────────────────────────────────────────────────────────────────────────────
# EDIT THESE to match your cluster checkout (must contain the provenance code):
REPO=/users/rkreft/apertus/benchmark-image-tokenizer
MEGATRON=/iopsstor/scratch/cscs/xyixuan/apertus/Megatron-LM   # <- confirm your Megatron-LM path
BASE=/iopsstor/scratch/cscs/rkreft/mllm-dpo
ENV_TOML=${REPO}/scripts/envs/nemo_25_11.toml                 # container; --environment goes on srun
# ──────────────────────────────────────────────────────────────────────────────

set -e
: "${VARIANT:?Set VARIANT=accepted or VARIANT=rejected}"
EXPECTED_RANKS="${EXPECTED_RANKS:-4}"
OUTPUT_DIR=${BASE}/tokenized/sft/mllm_dpo_${VARIANT}
MANIFEST=${BASE}/manifest/manifest.parquet

mkdir -p ${BASE}/logs
echo "=========================================="
echo "mllm-dpo merge — variant=${VARIANT}"
echo "OUTPUT_DIR:     ${OUTPUT_DIR}"
echo "EXPECTED_RANKS: ${EXPECTED_RANKS}"
echo "MANIFEST:       ${MANIFEST}"
echo "=========================================="

srun -u --environment=${ENV_TOML} --kill-on-bad-exit=1 bash -c "
    set -e
    export PYTHONPATH=${MEGATRON}:${REPO}:\$PYTHONPATH
    cd ${REPO}
    python -m vision_tokenization.pipeline.output.merge \
        ${OUTPUT_DIR} \
        --expected-ranks ${EXPECTED_RANKS} \
        --emit-provenance \
        --manifest ${MANIFEST}
"

echo ""
echo "Outputs in ${OUTPUT_DIR}:"
echo "  merged.bin / merged.idx"
echo "  merged.src.npy"
echo "  merged.provenance.parquet"
