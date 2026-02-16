#!/bin/bash
#SBATCH --account=a-infra01
#SBATCH --job-name=inspect-tok-data
#SBATCH --environment=emu3
#SBATCH --nodes=1
#SBATCH --partition=normal
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:10:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/benchmark-image-tokenizer/vision_tokenization/logs/inspect_tok_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/benchmark-image-tokenizer/vision_tokenization/logs/inspect_tok_%j.err

# Inspect a tokenized Megatron dataset (bin/idx files).
# Loads the text tokenizer and prints 10 random samples showing:
#   - First/last 30 token IDs
#   - First/last 300 characters of decoded text
#
# Usage:
#   sbatch inspect_tokenized_dataset.slurm <data_prefix>
#   sbatch inspect_tokenized_dataset.slurm <data_prefix> --tokenizer <tokenizer_path> --num-samples 20
#
# Example:
#   sbatch inspect_tokenized_dataset.slurm /path/to/tokenized_outputs/rank_0_shard_0_1

DATA_PREFIX="${1:?ERROR: Please provide data prefix as first argument (path without .bin/.idx extension)}"
shift

# Defaults
TOKENIZER_PATH="/capstor/store/cscs/swissai/infra01/MLLM/apertus_emu3.5_tokenizer"
NUM_SAMPLES=10

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --tokenizer)
            TOKENIZER_PATH="$2"
            shift 2
            ;;
        --num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

PROJECT_ROOT="/iopsstor/scratch/cscs/${USER}/benchmark-image-tokenizer"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

pip install -q omegaconf

echo "=================================================================================="
echo "Inspect Tokenized Megatron Dataset"
echo "=================================================================================="
echo "Data prefix:  ${DATA_PREFIX}"
echo "Tokenizer:    ${TOKENIZER_PATH}"
echo "Num samples:  ${NUM_SAMPLES}"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         $(hostname)"
echo "=================================================================================="

python - "$DATA_PREFIX" "$TOKENIZER_PATH" "$NUM_SAMPLES" <<'PYEOF'
import sys
import struct
import random
import numpy as np
import torch
from transformers import AutoTokenizer

data_prefix = sys.argv[1]
tokenizer_path = sys.argv[2]
num_samples = int(sys.argv[3])

# ---------------------------------------------------------------------------
# Megatron indexed dataset reader (from notebook)
# ---------------------------------------------------------------------------
class SimpleIndexedDatasetReader:
    def __init__(self, path_prefix):
        self.idx_path = path_prefix + '.idx'
        self.bin_path = path_prefix + '.bin'

        with open(self.idx_path, 'rb') as f:
            header = f.read(9)
            assert header == b"MMIDIDX\x00\x00", f"Invalid header: {header}"
            version = struct.unpack('<Q', f.read(8))[0]
            assert version == 1, f"Unsupported version: {version}"
            dtype_code = struct.unpack('<B', f.read(1))[0]
            self.dtype = {
                1: np.uint8, 2: np.int8, 3: np.int16, 4: np.int32,
                5: np.int64, 6: np.float64, 7: np.float32, 8: np.uint16,
            }.get(dtype_code, np.int32)

            self.sequence_count = struct.unpack('<Q', f.read(8))[0]
            self.document_count = struct.unpack('<Q', f.read(8))[0]

            self.sequence_lengths = np.frombuffer(f.read(self.sequence_count * 4), dtype=np.int32)
            self.sequence_pointers = np.frombuffer(f.read(self.sequence_count * 8), dtype=np.int64)
            self.document_indices = np.frombuffer(f.read(self.document_count * 8), dtype=np.int64)

        self.bin_data = np.memmap(self.bin_path, mode='r', order='C', dtype=self.dtype)

    def __len__(self):
        return self.sequence_count

    def get(self, idx):
        offset = self.sequence_pointers[idx]
        length = self.sequence_lengths[idx]
        elem_offset = offset // self.dtype().itemsize
        return self.bin_data[elem_offset:elem_offset + length]


# ---------------------------------------------------------------------------
# Load dataset and tokenizer
# ---------------------------------------------------------------------------
print("Loading dataset ...")
dataset = SimpleIndexedDatasetReader(data_prefix)
print(f"  Sequence count : {dataset.sequence_count}")
print(f"  Document count : {dataset.document_count}")
print(f"  Data type      : {dataset.dtype}")

print("Loading tokenizer ...")
text_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, fast=True)
print(f"  Vocab size     : {text_tokenizer.vocab_size}")

# ---------------------------------------------------------------------------
# Sample and display
# ---------------------------------------------------------------------------
N = min(num_samples, len(dataset))
indices = sorted(random.sample(range(len(dataset)), N))

SHOW_IDS = 30
SHOW_CHARS = 300

print(f"\nShowing {N} random samples (indices: {indices})")
print("=" * 80)

for i, idx in enumerate(indices):
    tokens = dataset.get(idx)
    token_ids = tokens.tolist()
    total = len(token_ids)

    # Decode full sequence
    decoded = text_tokenizer.decode(token_ids, skip_special_tokens=False)

    # Truncated views
    if total <= 2 * SHOW_IDS:
        ids_str = str(token_ids)
    else:
        ids_str = f"{token_ids[:SHOW_IDS]}  ...  {token_ids[-SHOW_IDS:]}"

    if len(decoded) <= 2 * SHOW_CHARS:
        text_str = decoded
    else:
        text_str = decoded[:SHOW_CHARS] + "\n  ...\n  " + decoded[-SHOW_CHARS:]

    print(f"\n--- Sample {i+1}/{N}  (dataset index {idx}, {total} tokens) ---")
    print(f"  Token IDs: {ids_str}")
    print(f"  Decoded  : {text_str}")
    print()

print("=" * 80)
print("Done.")
PYEOF

echo "=================================================================================="
echo "Job completed at $(date)"
echo "=================================================================================="
