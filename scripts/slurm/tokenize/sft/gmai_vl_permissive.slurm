#!/bin/bash
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --job-name=gmai-vl-perm-sft
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/gmai_vl_permissive_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/gmai_vl_permissive_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --partition=normal
#SBATCH --exclusive
#SBATCH --no-requeue

set -e
export LD_LIBRARY_PATH=/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse:${LD_LIBRARY_PATH:-}
export NUM_GPUS=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))

echo "=========================================="
echo "GMAI-VL-5.5M permissive (CC BY 4.0) medical SFT Tokenization"
echo "  3 sources: TotalSegmentator_v2 + crc100k + PAD-UFES-20"
echo "  GPUs: ${NUM_GPUS}, Nodes: ${SLURM_NNODES}"
echo "=========================================="

srun -u --kill-on-bad-exit=0 bash -c "
    set -e
    uv pip install --python /opt/venv/bin/python3 orjson
    export PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/apertus/Megatron-LM:/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier:\$PYTHONPATH
    cd /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier
    python -m vision_tokenization.tokenize \
        mode=sft dataset=sft/gmai_vl_permissive \
        num_gpus=${NUM_GPUS} \
        wandb.enabled=true \
        resume=true
"

echo ""
echo "=========================================="
echo "After completion, merge inline:"
echo "  PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/apertus/Megatron-LM python -m vision_tokenization.pipeline.output.merge \\"
echo "    /capstor/store/cscs/swissai/infra01/vision-datasets/tokenized/sft/gmai_vl_permissive \\"
echo ""
echo "=========================================="
