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

# Specify which directories to merge (customize via environment variable)
# Can be simple ranges or include resolution suffixes
# Examples: "0-359" or "0-359_res_256*256_720*720"
CC12M_DIRS="${CC12M_DIRS:-0-359_res_256*256_720*720 720-1099_res_256*256_720*720}"

# Output name for combined dataset - default includes the directories being merged
if [ -z "${COMBINED_NAME}" ]; then
    # Extract just the ranges and check if all have same resolution
    ranges=""
    has_res=false
    same_res=true
    first_res=""

    for dir in ${CC12M_DIRS}; do
        # Get range part (before _res if present)
        range=$(echo "$dir" | sed 's/_res_.*//')
        ranges="${ranges}_${range}"

        # Check resolution
        if [[ "$dir" == *"_res_"* ]]; then
            has_res=true
            res=$(echo "$dir" | sed 's/.*_res_//')
            if [ -z "$first_res" ]; then
                first_res="$res"
            elif [ "$first_res" != "$res" ]; then
                same_res=false
            fi
        fi
    done

    # Build name
    if [ "$has_res" = true ]; then
        if [ "$same_res" = true ]; then
            # All have same resolution
            safe_res=$(echo "$first_res" | sed 's/\*/x/g')
            COMBINED_NAME="cc12m${ranges}_res_${safe_res}_merged"
        else
            # Mixed resolutions
            COMBINED_NAME="cc12m${ranges}_mixed_res_merged"
        fi
    else
        # No resolution specified
        COMBINED_NAME="cc12m${ranges}_merged"
    fi
fi

# Set Python path
export PYTHONPATH="${MEGATRON_PATH}:${PYTHONPATH}"

# Create directories
mkdir -p "${MERGED_DIR}"

echo "=================================================================================="
echo "CC12M Dataset Merging"
echo "=================================================================================="
echo "Tokenized Directory: ${TOKENIZED_DIR}"
echo "Merged Directory: ${MERGED_DIR}"
echo "Directories to merge: ${CC12M_DIRS}"
echo "Combined Output: ${COMBINED_NAME}"
echo "=================================================================================="

# Function to merge a single directory
merge_range() {
    local dir_name=$1
    local input="${TOKENIZED_DIR}/cc12m_${dir_name}"
    local output="${MERGED_DIR}/cc12m_${dir_name}"

    # Check if input exists
    if [ ! -d "${input}" ]; then
        echo "  ✗ cc12m_${dir_name}: Directory not found"
        return 1
    fi

    # Check if already merged
    if [ -f "${output}.bin" ] && [ -f "${output}.idx" ]; then
        echo "  ✓ cc12m_${dir_name}: Already merged"
        return 0
    fi

    # Check for shard files (per-shard outputs like 00360.bin)
    num_bin_files=$(ls "${input}"/*.bin 2>/dev/null | wc -l)

    if [ ${num_bin_files} -eq 0 ]; then
        echo "  ✗ cc12m_${dir_name}: No shard files found"
        return 1
    fi

    echo "  ⚙ cc12m_${dir_name}: Merging ${num_bin_files} shard files..."

    # Create temp directory with symlinks - sanitize dir_name for temp dir
    safe_name=$(echo "${dir_name}" | sed 's/[*\/]/_/g')
    temp_dir="/tmp/merge_cc12m_${safe_name}_$$"
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
            echo "  ✓ cc12m_${dir_name}: Merged successfully (${size}, ${stats})"
        else
            echo "  ✓ cc12m_${dir_name}: Merged successfully (${size})"
        fi
        return 0
    else
        echo "  ✗ cc12m_${dir_name}: Merge failed"
        return 1
    fi
}

# Step 1: Merge individual range directories
echo ""
echo "Step 1: Merging Individual Range Directories"
echo "---------------------------------------------"

success_count=0
failed_ranges=""

for dir in ${CC12M_DIRS}; do
    merge_range "${dir}"
    if [ $? -eq 0 ]; then
        ((success_count++))
    else
        failed_ranges="${failed_ranges} ${dir}"
    fi
done

echo ""
echo "Merged ${success_count} out of $(echo ${CC12M_DIRS} | wc -w) directories"

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

    # Create temp directory with symlinks to only the SPECIFIC cc12m files we want
    temp_merge_dir="/tmp/cc12m_final_merge_$$"
    mkdir -p "${temp_merge_dir}"

    # Symlink only the specific directories we're merging (from CC12M_DIRS)
    for dir in ${CC12M_DIRS}; do
        bin_file="${MERGED_DIR}/cc12m_${dir}.bin"
        idx_file="${MERGED_DIR}/cc12m_${dir}.idx"

        if [ -f "$bin_file" ] && [ -f "$idx_file" ]; then
            ln -s "$bin_file" "${temp_merge_dir}/$(basename $bin_file)"
            ln -s "$idx_file" "${temp_merge_dir}/$(basename $idx_file)"
        else
            echo "Warning: Missing files for cc12m_${dir}"
        fi
    done

    # Check if we have files to merge
    num_merged=$(ls "${temp_merge_dir}"/cc12m_*.bin 2>/dev/null | wc -l)

    if [ ${num_merged} -eq 0 ]; then
        echo "No merged range files found in ${MERGED_DIR}"
        rm -rf "${temp_merge_dir}"
        exit 1
    fi

    python "${MEGATRON_PATH}/scripts/merge_datasets/merge_datasets.py" \
        --input "${temp_merge_dir}" \
        --output-prefix "${output_path}"

    merge_status=$?

    # Clean up temp directory
    rm -rf "${temp_merge_dir}"

    if [ ${merge_status} -ne 0 ]; then
        echo "Final merge failed"
        exit ${merge_status}
    fi

    if [ $? -eq 0 ]; then
        size=$(ls -lh "${output_path}.bin" | awk '{print $5}')

        # Count total samples and tokens
        total_stats=$(python -c "
import json
total_samples = 0
total_tokens = 0
dirs = '${CC12M_DIRS}'.split()
for d in dirs:
    try:
        with open('${TOKENIZED_DIR}/cc12m_' + d + '/dataset_info.json') as f:
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