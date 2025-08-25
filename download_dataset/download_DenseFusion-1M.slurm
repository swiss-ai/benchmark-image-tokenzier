#!/bin/bash
#SBATCH --account=a-infra01
#SBATCH --environment=emu3
#SBATCH --job-name=img2dataset
#SBATCH --output=/iopsstor/scratch/cscs/%u/benchmark-image-tokenzier/download_dataset/DenseFusion-1M/logs/download-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/benchmark-image-tokenzier/download_dataset/DenseFusion-1M/logs/download-%A.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72

# Create log directory
mkdir -p /iopsstor/scratch/cscs/$USER/benchmark-image-tokenzier/download_dataset/logs

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Configuration
INPUT_FILE="/capstor/store/cscs/swissai/infra01/vision-datasets/DenseFusion/DenseFusion-1M.jsonl"
OUTPUT_DIR="/capstor/store/cscs/swissai/infra01/vision-datasets/DenseFusion-1M"

echo "=========================================="
echo "DenseFusion-1M Download Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"
echo "Input: $INPUT_FILE"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# Run img2dataset with optimized settings for 288 CPUs
img2dataset \
    --url_list $INPUT_FILE \
    --input_format jsonl \
    --output_folder $OUTPUT_DIR \
    --output_format webdataset \
    --url_col url \
    --caption_col caption \
    --save_additional_columns '["image_id","image_path"]' \
    --min_image_size 512 \
    --processes_count 72 \
    --thread_count 256 \
    --retries 3 \
    --timeout 30 \
    --number_sample_per_shard 20000

# Check exit status
EXIT_CODE=$?

echo "=========================================="
echo "Job completed at: $(date)"
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Download completed successfully!"
    echo "Checking output..."
    ls -lah $OUTPUT_DIR | head -20
    echo "Total tar files created: $(ls $OUTPUT_DIR/*.tar | wc -l)"
else
    echo "Download failed with exit code: $EXIT_CODE"
fi

echo "=========================================="