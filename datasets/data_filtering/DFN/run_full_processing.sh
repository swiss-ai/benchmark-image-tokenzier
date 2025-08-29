#!/bin/bash

# Run full dataset processing with DFN scores


# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dfn_filter

# Dataset and output paths
DATASET_PATH="/capstor/store/cscs/swissai/infra01/vision-datasets/llava_pretrain/LLaVA-Pretrain/webdataset_output"
OUTPUT_PATH="/iopsstor/scratch/cscs/rqu/benchmark-image-tokenzier/datasets/data_filtering/DFN/dfn_scores_output"

echo "=================================================="
echo "DFN Score Processing for LLaVA Dataset"
echo "=================================================="
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_PATH"
echo "Start time: $(date)"
echo ""

# Count shards
NUM_SHARDS=$(find $DATASET_PATH -name "*.tar" | wc -l)
echo "Total shards to process: $NUM_SHARDS"
echo "Using $(nvidia-smi -L | wc -l) GPUs"
echo ""

# Run processing
python process_webdataset.py \
    --dataset_path "$DATASET_PATH" \
    --output_path "$OUTPUT_PATH" \
    --batch_size 32 \
    --num_workers 4

echo ""
echo "End time: $(date)"
echo "=================================================="

# Check results
if [ -f "$OUTPUT_PATH/dfn_scores_all.parquet" ]; then
    echo "Success! Results saved to $OUTPUT_PATH/dfn_scores_all.parquet"
    echo ""
    echo "Statistics:"
    cat "$OUTPUT_PATH/dfn_scores_statistics.json"
else
    echo "Error: Output file not found"
fi