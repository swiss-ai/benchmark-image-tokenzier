#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=mllm-dpo-binidx
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/mllm_dpo_binidx_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/mllm_dpo_binidx_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=debug
#SBATCH --no-requeue

# mllm_dpo DPO binidx phase (4th step, named hydra phase). Reads the published store, builds one
# contiguous Megatron document per pair -- [prompt(text + inlined vision) | chosen | rejected] --
# plus index_{split}.parquet (slice boundaries + image spans), registers them in manifest.json
# (schema 4), and retires the now-redundant deduped tokens/. CPU-only, parallel; enable_thinking
# follows the CHOSEN.

set -e
export LD_LIBRARY_PATH=/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/apertus/Megatron-LM:/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier:${PYTHONPATH:-}
cd /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier
python -c "import orjson" 2>/dev/null || uv pip install --python /opt/venv/bin/python3 orjson

python -m vision_tokenization.dpo_binidx mode=alignment dataset=alignment/mllm_dpo dataset.binidx_delete_tokens=true
echo "BINIDX DONE: mllm_dpo"
