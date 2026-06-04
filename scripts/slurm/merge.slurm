#!/bin/bash
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5
#SBATCH --job-name=tokenize-merge
#SBATCH --environment=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/scripts/envs/nemo_25_11.toml
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/merge_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/merge_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=16
#SBATCH --partition=normal
#
# Generic cross-rank merge step. Submit as a dependency on a tokenization job:
#
#   JOBID=$(sbatch --parsable scripts/slurm/google_rsrcc.slurm)
#   sbatch --dependency=afterany:$JOBID \
#          --export=ALL,OUTPUT_DIR=/capstor/store/cscs/swissai/infra01/vision-datasets/tokenized/sft/google_rsrcc,EXPECTED_RANKS=4 \
#          scripts/slurm/merge.slurm
#
# `afterany` (not `afterok`) so the merge still attempts even if tokenization
# exited non-zero on a benign signal (SIGTERM at time limit, etc.) — the merge
# itself fails cleanly if shards are incomplete.
#
# Required env:
#   OUTPUT_DIR        rank shard directory (same dir tokenize wrote to)
#   EXPECTED_RANKS    number of ranks (NUM_GPUS from the parent job)
#
# Optional env:
#   STRIP_THINKING    "true" (default) to also produce merged_no_cot.bin/idx
#   TOKENIZER_PATH    default apertus_emu3.5_instruct; only consulted with strip
#   EMIT_PROVENANCE   "true" to require per-shard .src.npy sidecars and emit
#                     merged.src.npy (set this when the tokenize job ran with
#                     emit_provenance=true)
#   MANIFEST          manifest parquet used to build the plan; with provenance,
#                     resolves merged.src.npy -> merged.provenance.parquet
#

set -euo pipefail

: "${OUTPUT_DIR:?OUTPUT_DIR is required}"
: "${EXPECTED_RANKS:?EXPECTED_RANKS is required}"
STRIP_THINKING="${STRIP_THINKING:-true}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_instruct}"
EMIT_PROVENANCE="${EMIT_PROVENANCE:-false}"
MANIFEST="${MANIFEST:-}"

echo "=========================================="
echo "Cross-rank merge"
echo "=========================================="
echo "Job ID:          ${SLURM_JOB_ID}"
echo "Node:            $(hostname -s)"
echo "OUTPUT_DIR:      ${OUTPUT_DIR}"
echo "EXPECTED_RANKS:  ${EXPECTED_RANKS}"
echo "STRIP_THINKING:  ${STRIP_THINKING}"
echo "TOKENIZER_PATH:  ${TOKENIZER_PATH}"
echo "EMIT_PROVENANCE: ${EMIT_PROVENANCE}"
echo "MANIFEST:        ${MANIFEST:-<none>}"
echo "=========================================="

mkdir -p /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs

EXTRA_FLAGS=()
if [ "${STRIP_THINKING}" = "true" ]; then
    EXTRA_FLAGS+=(--strip-thinking --resolve-thinking-ids --tokenizer-path "${TOKENIZER_PATH}")
fi
if [ "${EMIT_PROVENANCE}" = "true" ]; then
    EXTRA_FLAGS+=(--emit-provenance)
fi
if [ -n "${MANIFEST}" ]; then
    EXTRA_FLAGS+=(--manifest "${MANIFEST}")
fi

srun -u --kill-on-bad-exit=1 bash -c "
    set -e
    export PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/apertus/Megatron-LM:/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier:\$PYTHONPATH
    cd /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier

    python -m vision_tokenization.pipeline.output.merge \
        ${OUTPUT_DIR} \
        --expected-ranks ${EXPECTED_RANKS} \
        ${EXTRA_FLAGS[*]}
"

echo ""
echo "=========================================="
echo "Merge completed!"
echo "Outputs:"
echo "  ${OUTPUT_DIR}/merged.bin / merged.idx"
if [ "${STRIP_THINKING}" = "true" ]; then
    echo "  ${OUTPUT_DIR}/merged_no_cot.bin / merged_no_cot.idx"
fi
if [ "${EMIT_PROVENANCE}" = "true" ]; then
    echo "  ${OUTPUT_DIR}/merged.src.npy"
    if [ -n "${MANIFEST}" ]; then
        echo "  ${OUTPUT_DIR}/merged.provenance.parquet"
    fi
fi
echo "=========================================="
