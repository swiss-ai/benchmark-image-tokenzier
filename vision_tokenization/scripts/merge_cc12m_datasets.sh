#!/bin/bash
#SBATCH --account=a-infra01
#SBATCH --job-name=cc12m-merge
#SBATCH --environment=nemo
#SBATCH --nodes=1
#SBATCH --partition=debug
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=01:30:00
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/benchmark-image-tokenzier/vision_tokenization/logs/merge_cc12m_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/benchmark-image-tokenzier/vision_tokenization/logs/merge_cc12m_%j.err

###################### Merge CC12M Tokenized Datasets ######################

# Configuration
MEGATRON_PATH="/iopsstor/scratch/cscs/xyixuan/Megatron-LM"
CC12M_BASE="/capstor/store/cscs/swissai/infra01/vision-datasets/conceptual-12m"
TOKENIZED_DIR="${CC12M_BASE}/tokenized"
MERGED_DIR="${CC12M_BASE}/merged"

# Specify which ranges to merge (customize via environment variable)
# Format: "0-119 120-239 240-359" etc
RANGES="${RANGES:-360-479 480-599 600-719}"

# Output name for combined dataset
COMBINED_NAME="${COMBINED_NAME:-cc12m_merged}"

# Set Python path
export PYTHONPATH="${MEGATRON_PATH}:${PYTHONPATH}"

# Create directories
mkdir -p "${MERGED_DIR}"

echo "=================================================================================="
echo "CC12M Dataset Merging"
echo "=================================================================================="
echo "Tokenized Directory: ${TOKENIZED_DIR}"
echo "Merged Directory: ${MERGED_DIR}"
echo "Ranges to merge: ${RANGES}"
echo "Combined Output: ${COMBINED_NAME}"
echo "=================================================================================="

# Function to merge a single range directory
merge_range() {
    local range=$1
    local input="${TOKENIZED_DIR}/cc12m_${range}"
    local output="${MERGED_DIR}/cc12m_${range}"

    # Check if input exists
    if [ ! -d "${input}" ]; then
        echo "  ✗ cc12m_${range}: Directory not found"
        return 1
    fi

    # Check if already merged
    if [ -f "${output}.bin" ] && [ -f "${output}.idx" ]; then
        echo "  ✓ cc12m_${range}: Already merged"
        return 0
    fi

    # Check for shard files (per-shard outputs like 00360.bin)
    num_bin_files=$(ls "${input}"/*.bin 2>/dev/null | wc -l)

    if [ ${num_bin_files} -eq 0 ]; then
        echo "  ✗ cc12m_${range}: No shard files found"
        return 1
    fi

    echo "  ⚙ cc12m_${range}: Merging ${num_bin_files} shard files..."

    # Create temp directory with symlinks (like in merge_finevision_datasets.sh)
    temp_dir="/tmp/merge_cc12m_${range}_$$"
    mkdir -p "${temp_dir}"

    # Symlink all .bin and .idx files
    for f in "${input}"/*.bin "${input}"/*.idx; do
        [ -f "$f" ] && ln -s "$f" "${temp_dir}/$(basename $f)"
    done

    # Perform merge using temp directory
    python "${MEGATRON_PATH}/scripts/merge_datasets/merge_datasets.py" \
        --input "${temp_dir}" \
        --output-prefix "${output}"

    merge_status=$?

    # Clean up temp directory
    rm -rf "${temp_dir}"

    if [ ${merge_status} -eq 0 ]; then
        size=$(ls -lh "${output}.bin" 2>/dev/null | awk '{print $5}')

        # Get statistics from dataset_info.json if it exists
        if [ -f "${input}/dataset_info.json" ]; then
            stats=$(python -c "
import json
with open('${input}/dataset_info.json') as f:
    info = json.load(f)
    stats = info.get('statistics', {})
    print(f\"samples={stats.get('total_samples', 0):,}, tokens={stats.get('total_tokens', 0):,}\")
" 2>/dev/null || echo "")
            echo "  ✓ cc12m_${range}: Merged successfully (${size}, ${stats})"
        else
            echo "  ✓ cc12m_${range}: Merged successfully (${size})"
        fi
        return 0
    else
        echo "  ✗ cc12m_${range}: Merge failed"
        return 1
    fi
}

# Step 1: Merge individual range directories
echo ""
echo "Step 1: Merging Individual Range Directories"
echo "---------------------------------------------"

success_count=0
failed_ranges=""

for range in ${RANGES}; do
    merge_range "${range}"
    if [ $? -eq 0 ]; then
        ((success_count++))
    else
        failed_ranges="${failed_ranges} ${range}"
    fi
done

echo ""
echo "Merged ${success_count} out of $(echo ${RANGES} | wc -w) range directories"

if [ -n "${failed_ranges}" ]; then
    echo "Failed ranges:${failed_ranges}"
fi

if [ ${success_count} -eq 0 ]; then
    echo "ERROR: No ranges were successfully merged"
    exit 1
fi

# Step 2: Create combined dataset from all ranges
echo ""
echo "Step 2: Creating Combined Dataset"
echo "----------------------------------"

output_path="${CC12M_BASE}/${COMBINED_NAME}"

# Check if already exists
if [ -f "${output_path}.bin" ] && [ -f "${output_path}.idx" ]; then
    echo "Combined dataset already exists at ${output_path}"
    echo "Delete it first if you want to recreate it"
else
    echo "Merging all ranges into ${COMBINED_NAME}..."

    # Check if merged directory has files
    num_merged=$(ls "${MERGED_DIR}"/cc12m_*.bin 2>/dev/null | wc -l)

    if [ ${num_merged} -eq 0 ]; then
        echo "No merged range files found in ${MERGED_DIR}"
        exit 1
    fi

    python "${MEGATRON_PATH}/scripts/merge_datasets/merge_datasets.py" \
        --input "${MERGED_DIR}" \
        --output-prefix "${output_path}"

    if [ $? -eq 0 ]; then
        size=$(ls -lh "${output_path}.bin" | awk '{print $5}')

        # Count total samples and tokens
        total_stats=$(python -c "
import json
total_samples = 0
total_tokens = 0
ranges = '${RANGES}'.split()
for r in ranges:
    try:
        with open('${TOKENIZED_DIR}/cc12m_' + r + '/dataset_info.json') as f:
            info = json.load(f)
            stats = info.get('statistics', {})
            total_samples += stats.get('total_samples', 0)
            total_tokens += stats.get('total_tokens', 0)
    except:
        pass
if total_samples > 0:
    print(f'Total: {total_samples:,} samples, {total_tokens:,} tokens')
" 2>/dev/null || echo "")

        echo "✓ Combined dataset created successfully (${size})"
        [ -n "${total_stats}" ] && echo "  ${total_stats}"
    else
        echo "✗ Failed to create combined dataset"
        exit 1
    fi
fi

# Summary
echo ""
echo "=================================================================================="
echo "Summary"
echo "=================================================================================="
echo "Merged range datasets in ${MERGED_DIR}:"
ls -lh "${MERGED_DIR}"/cc12m_*.bin 2>/dev/null | while read line; do
    file=$(echo $line | awk '{print $9}')
    size=$(echo $line | awk '{print $5}')
    basename=$(basename $file .bin)
    echo "  - ${basename}: ${size}"
done

if [ -f "${output_path}.bin" ]; then
    echo ""
    echo "Combined dataset:"
    ls -lh "${output_path}.bin" | awk '{print "  - "'${COMBINED_NAME}'": "$5}'
fi

# Create a summary JSON file
echo ""
echo "Creating summary file..."
python -c "
import json
import os
from pathlib import Path

summary = {
    'ranges_processed': '${RANGES}'.split(),
    'output_path': '${output_path}',
    'merged_ranges': {},
    'combined_dataset': None
}

# Collect info for each range
for r in '${RANGES}'.split():
    range_dir = '${TOKENIZED_DIR}/cc12m_' + r
    merged_file = '${MERGED_DIR}/cc12m_' + r + '.bin'

    if os.path.exists(range_dir + '/dataset_info.json'):
        with open(range_dir + '/dataset_info.json') as f:
            info = json.load(f)
            summary['merged_ranges'][r] = {
                'samples': info['statistics']['total_samples'],
                'tokens': info['statistics']['total_tokens'],
                'merged': os.path.exists(merged_file)
            }

# Check combined dataset
if os.path.exists('${output_path}.bin'):
    size = os.path.getsize('${output_path}.bin')
    summary['combined_dataset'] = {
        'path': '${output_path}',
        'size_bytes': size,
        'size_gb': round(size / (1024**3), 2)
    }

# Save summary
with open('${MERGED_DIR}/merge_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print('Summary saved to ${MERGED_DIR}/merge_summary.json')
" 2>/dev/null || echo "Could not create summary file"

echo ""
echo "Completed at $(date)"
echo "===================================================================================="