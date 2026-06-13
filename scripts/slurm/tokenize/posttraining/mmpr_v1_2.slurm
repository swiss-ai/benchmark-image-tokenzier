#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=mmpr-v1.2-store
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/mmpr_v1_2_store_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/mmpr_v1_2_store_%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --partition=normal
#SBATCH --exclusive
#SBATCH --no-requeue

# MMPR-v1.2 -> 8 preference stores (one per converted shard).
#
# GEOMETRY (matches the multi-node SFT tokenize scripts): ONE job, 2 nodes,
# 4 GPUs/node, 4 tasks/node = 8 ranks, fully packing both nodes. mllm_dpo.slurm
# uses 1 node/1 GPU only because it is a single tiny shard (5,182 pairs); MMPR
# has 8 shards, so it packs like SFT instead of fanning into 8 whole nodes.
#
# PER-TASK SEMANTICS (matches mllm_dpo.slurm): posttraining is single-rank per
# STORE (pipeline/__init__.py:152 hard-errors unless world_size==1), so this is
# NOT an 8-way distributed job — it is 8 INDEPENDENT single-rank pipelines, one
# per shard. Each srun task therefore overrides WORLD_SIZE=1/RANK=0 (the pipeline
# reads WORLD_SIZE||SLURM_NTASKS, which would otherwise be 8) and maps
# SLURM_PROCID (0..7) -> shard. GPU pinning is left to the pipeline's
# torch.cuda.set_device(local_rank) with local_rank=SLURM_LOCALID (0..3), the
# same mechanism the SFT scripts use — the 4 tasks on a node take GPUs 0..3.
# --kill-on-bad-exit=0: one shard failing must not abort the other 7.
#
# Walltime: ~33.4k unique imgs/shard at 0.77 MPx capped mean; pixel-scaled
# ~59 min/shard, all 8 in parallel -> ~1 h wall + scan/load/publish. 3 h pad
# (>debug's 1h30) on normal. posttraining is seal-at-end with NO resume.
#
# Submit from a clean env (dev-shell LD contamination + slingshot libjson):
#   env -i HOME="$HOME" USER="$USER" LD_LIBRARY_PATH=/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse \
#       /usr/bin/sbatch /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/scripts/slurm/tokenize/posttraining/mmpr_v1_2.slurm

set -e
export LD_LIBRARY_PATH=/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse:${LD_LIBRARY_PATH:-}

# Data location: single source of truth is the dataset config's `source_dir`
# (not duplicated here). The pipeline reads the same yaml; we just derive the
# per-shard input path from it below.
DATASET_CFG=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/vision_tokenization/configs/dataset/posttraining/mmpr_v1_2.yaml
SRC_DIR=$(awk -F': +' '/^source_dir:/{print $2; exit}' "$DATASET_CFG")
test -n "$SRC_DIR" || { echo "FATAL: source_dir not found in $DATASET_CFG"; exit 1; }
# Pre-flight on the head node: converter seals shards via .tmp rename and writes
# the report last — all 8 shards + the report must exist before any GPU minute.
test -f "${SRC_DIR}/_conversion_report.json" || { echo "FATAL: conversion report missing — converter not finished"; exit 1; }
for i in 0 1 2 3 4 5 6 7; do
    test -f "${SRC_DIR}/part_0${i}.parquet" || { echo "FATAL: ${SRC_DIR}/part_0${i}.parquet missing"; exit 1; }
done

echo "=========================================="
echo "MMPR-v1.2 -> 8 views+media preference stores (1 job, 2 nodes, 8 GPUs)"
echo "  mode=posttraining task=preference; 8 independent single-rank shards"
echo "  out: .../vision-datasets/tokenized/preference/mmpr_v1_2_part0{0..7}"
echo "=========================================="

srun -ul --kill-on-bad-exit=0 bash -c '
    set -e
    export LD_LIBRARY_PATH=/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse:${LD_LIBRARY_PATH:-}
    export PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier:$PYTHONPATH
    cd /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier

    # This task owns exactly one shard; make it a clean single-rank world.
    SHARD=$(printf "%02d" "${SLURM_PROCID}")
    export RANK=0 WORLD_SIZE=1            # posttraining single-rank contract (world_size==1)
    unset LOCAL_RANK                       # let pipeline read SLURM_LOCALID -> set_device(0..3)
    INPUT='"${SRC_DIR}"'/part_${SHARD}.parquet
    OUT_NAME=mmpr_v1_2_part${SHARD}
    echo "[task ${SLURM_PROCID} local ${SLURM_LOCALID}] shard ${SHARD} -> ${OUT_NAME}"

    python -m vision_tokenization.tokenize \
        mode=posttraining dataset=posttraining/mmpr_v1_2 \
        dataset.input_parquet=${INPUT} \
        dataset.output_name=${OUT_NAME} \
        num_gpus=1 \
        wandb.enabled=true wandb.name=mmpr_v1_2_part${SHARD}

    echo "[task ${SLURM_PROCID}] Gate 2: ${OUT_NAME}"
    python vision_tokenization/tests/alignment/check_store.py \
        /capstor/store/cscs/swissai/infra01/vision-datasets/tokenized/preference/${OUT_NAME}
'
echo "END TIME: $(date)"
