#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=profile-emu3p5
#SBATCH --environment=nemo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --reservation=PA-2338-RL
#SBATCH --chdir=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/vision_tokenization/profile/logs/profile_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/vision_tokenization/profile/logs/profile_%j.err

# Profile Emu3.5 vision tokenizer with nsys on a clean, isolated GPU.
#
# Usage:
#   sbatch vision_tokenization/profile/run_profile.slurm
#
# Multi-GPU profiling (4 GPUs, each runs independently):
#   sbatch --gpus-per-node=4 --ntasks-per-node=4 \
#       vision_tokenization/profile/run_profile.slurm --multi-gpu

PROJECT_ROOT="/iopsstor/scratch/cscs/${USER}/apertus/benchmark-image-tokenzier"
REPORT_DIR="${PROJECT_ROOT}/vision_tokenization/profile/reports"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

mkdir -p "${REPORT_DIR}"
mkdir -p "${PROJECT_ROOT}/vision_tokenization/profile/logs"

# Parse args
MULTI_GPU=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --multi-gpu) MULTI_GPU=true; shift ;;
        *) shift ;;
    esac
done

echo "=================================================================="
echo "Emu3.5 Vision Tokenizer Profiling (nsys)"
echo "=================================================================="
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         $(hostname)"
echo "GPUs:         $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "multi-gpu:    ${MULTI_GPU}"
echo "Report dir:   ${REPORT_DIR}"
echo "=================================================================="

nvidia-smi

if [ "$MULTI_GPU" = true ]; then
    echo ""
    echo "--- Multi-GPU profiling (one tokenizer per GPU) ---"
    srun nsys profile \
        --trace=cuda,nvtx \
        --force-overwrite=true \
        --stats=true \
        -o "${REPORT_DIR}/emu3p5_rank%q{SLURM_PROCID}_${SLURM_JOB_ID}" \
        python -m vision_tokenization.profile.profile_emu3p5
else
    echo ""
    echo "--- Single-GPU profiling ---"
    nsys profile \
        --trace=cuda,nvtx \
        --force-overwrite=true \
        --stats=true \
        -o "${REPORT_DIR}/emu3p5_${SLURM_JOB_ID}" \
        python -m vision_tokenization.profile.profile_emu3p5
fi

echo ""
echo "=================================================================="
echo "Profiling complete at $(date)"
echo "Reports saved to: ${REPORT_DIR}/"
ls -lh "${REPORT_DIR}"/*${SLURM_JOB_ID}* 2>/dev/null
echo ""
echo "Inspect with:"
echo "  nsys stats ${REPORT_DIR}/emu3p5_${SLURM_JOB_ID}.nsys-rep"
echo "=================================================================="
