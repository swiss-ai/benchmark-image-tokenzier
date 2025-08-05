#!/bin/bash
# Script to tokenize the LLaVA dataset with multimodal tokenizer support

# Set paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TOKENIZER_SCRIPT="${SCRIPT_DIR}/tokenize_images.py"
VOCAB_DETECTOR="${SCRIPT_DIR}/utils/detect_vocab_size.py"

# Dataset configuration
DATASET_NAME="llava"
INPUT_PATH="/capstor/store/cscs/swissai/infra01/vision-datasets/llava_pretrain/LLaVA-Pretrain"
OUTPUT_DIR="/iopsstor/scratch/cscs/xyixuan/tokenized_datasets"

# Processing configuration
BATCH_SIZE=16
NUM_SHARDS=100  # More shards for 558K images
DEVICE="cuda"

# Vocabulary configuration
TEXT_TOKENIZER="alehc/swissai-tokenizer"
TEXT_VOCAB_SIZE=0  # Set to 0 for auto-detection or specify manually
IMAGE_VOCAB_SIZE=131072  # Emu3 vocabulary size (2^17)

# Auto-detect text vocabulary size if not specified
if [ "${TEXT_VOCAB_SIZE}" -eq 0 ]; then
    echo "Auto-detecting text vocabulary size..."
    echo "Running: python ${VOCAB_DETECTOR} ${TEXT_TOKENIZER}"
    echo "----------------------------------------"
    
    # Run vocabulary detection (you can manually set TEXT_VOCAB_SIZE above if this fails)
    python "${VOCAB_DETECTOR}" "${TEXT_TOKENIZER}" || {
        echo "❌ Failed to auto-detect vocabulary size"
        echo "Please manually set TEXT_VOCAB_SIZE in this script"
        echo "You can also run: python utils/detect_vocab_size.py ${TEXT_TOKENIZER}"
        exit 1
    }
    
    echo "----------------------------------------"
    echo "⚠️  Please check the output above and manually set TEXT_VOCAB_SIZE in this script"
    echo "Then re-run this script with the correct vocabulary size"
    exit 0
fi

# Run tokenization
echo "Starting tokenization of LLaVA dataset..."
echo "Input: ${INPUT_PATH}"
echo "Output: ${OUTPUT_DIR}/${DATASET_NAME}"
echo "Device: ${DEVICE}"
echo "Batch size: ${BATCH_SIZE}"
echo "Number of shards: ${NUM_SHARDS}"
echo "Text vocabulary size: ${TEXT_VOCAB_SIZE}"
echo "Image vocabulary size: ${IMAGE_VOCAB_SIZE}"
echo "Total vocabulary size: $((TEXT_VOCAB_SIZE + IMAGE_VOCAB_SIZE))"
echo "Image token offset: ${TEXT_VOCAB_SIZE}"
echo "----------------------------------------"

python "${TOKENIZER_SCRIPT}" \
    --dataset "${DATASET_NAME}" \
    --input-path "${INPUT_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --tokenizer emu3 \
    --batch-size "${BATCH_SIZE}" \
    --num-shards "${NUM_SHARDS}" \
    --device "${DEVICE}" \
    --text-vocab-size "${TEXT_VOCAB_SIZE}" \
    --image-vocab-size "${IMAGE_VOCAB_SIZE}" \
    --create-shards

echo "----------------------------------------"
echo "Tokenization complete!"