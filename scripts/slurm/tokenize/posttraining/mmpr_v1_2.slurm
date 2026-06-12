#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=mmpr-v1.2-store
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/mmpr_v1_2_store_p%a_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/mmpr_v1_2_store_p%a_%j.err
#SBATCH --time=03:00:00
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --partition=normal
#SBATCH --no-requeue

# MMPR-v1.2 -> 8 preference stores (one per converted shard), posttraining
# mode, single-rank/1-GPU per shard (pipeline contract), array task = shard.
#
# Walltime: mllm_dpo ran 4,814 unique imgs (0.22 MPx mean) in 148 s
# end-to-end. Each shard here is ~33.4k unique imgs at 0.77 MPx capped mean:
# image-scaled estimate 17 min/shard, pixel-scaled 59 min/shard (encode is
# closer to pixel-bound) + ~10 min scan (sha256+dims over ~26 GB parquet),
# model load, publish. Padded ~2.5x of the upper band -> 3h; exceeds debug's
# 1h30, hence normal partition.
# posttraining is seal-at-end with NO resume: a killed shard re-runs whole.
#
# Submit from a clean env (dev-shell LD contamination):
#   env -i HOME="$HOME" USER="$USER" PATH=/usr/bin:/bin:/usr/local/bin \
#       sbatch /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/scripts/slurm/tokenize/posttraining/mmpr_v1_2.slurm

set -e
export LD_LIBRARY_PATH=/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse:${LD_LIBRARY_PATH:-}

PART=$(printf "%02d" "${SLURM_ARRAY_TASK_ID}")
SRC_DIR=/capstor/store/cscs/swissai/infra01/vision-datasets/alignment-processed/mmpr-v1.2
INPUT=${SRC_DIR}/part_${PART}.parquet
OUT_NAME=mmpr_v1_2_part${PART}

# Pre-flight: the converter seals shards via .tmp rename and writes the
# report last — both must exist before any GPU minute is spent.
test -f "${INPUT}" || { echo "FATAL: ${INPUT} missing (converter not finished?)"; exit 1; }
test -f "${SRC_DIR}/_conversion_report.json" || { echo "FATAL: conversion report missing — converter still running"; exit 1; }

echo "=========================================="
echo "MMPR-v1.2 shard ${PART} -> views+media preference store"
echo "  mode=posttraining task=preference, single-rank, 1 GPU"
echo "  in:  ${INPUT}"
echo "  out: .../vision-datasets/tokenized/preference/${OUT_NAME}"
echo "=========================================="

srun -u --kill-on-bad-exit=0 bash -c "
    set -e
    export PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier:\$PYTHONPATH
    cd /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier
    python -m vision_tokenization.tokenize \
        mode=posttraining dataset=posttraining/mmpr_v1_2 \
        dataset.input_parquet=${INPUT} \
        dataset.output_name=${OUT_NAME} \
        num_gpus=1 \
        wandb.enabled=false
"

echo "Gate 2:"
srun -u --overlap bash -c "
    export PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier:\$PYTHONPATH
    cd /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier
    python vision_tokenization/tests/alignment/check_store.py \
        /capstor/store/cscs/swissai/infra01/vision-datasets/tokenized/preference/${OUT_NAME}
"
