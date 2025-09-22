#!/bin/bash
#SBATCH --account=a-infra01
#SBATCH --job-name=merge-all
#SBATCH --environment=nemo
#SBATCH --nodes=1
#SBATCH --partition=debug
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=01:30:00
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/benchmark-image-tokenzier/vision_tokenization/logs/merge_all_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/benchmark-image-tokenzier/vision_tokenization/logs/merge_all_%j.err

###################### Merge Multiple Datasets from Different Sources ######################
# This script can merge specific folders/files from different datasets into one combined dataset

# Configuration
MEGATRON_PATH="/iopsstor/scratch/cscs/xyixuan/Megatron-LM"
BASE_DIR="/capstor/store/cscs/swissai/infra01/vision-datasets"

# Output configuration
OUTPUT_BASE="${OUTPUT_BASE:-${BASE_DIR}/merged}"
OUTPUT_NAME="${OUTPUT_NAME:-image_only_merged}"

# Define source datasets as "dataset_path:pattern" pairs
# You can customize this list by setting SOURCE_DATASETS environment variable
# Format: "path1:pattern1 path2:pattern2 ..."
# Examples:
#   - Full directory: "conceptual-12m/tokenized/cc12m_0-119:*"
#   - Specific files: "imagenet-w21/tokenized/imagenet-w21_range_0-200:imagenet*.bin"
#   - Already merged: "conceptual-12m/merged:cc12m_*.bin"

if [ -z "${SOURCE_DATASETS}" ]; then
    # Default: merge ImageNet and FineVision merged files
    SOURCE_DATASETS=(
        # ImageNet merged file (in base directory)
        "${BASE_DIR}:imagenet-w21_range_0-2048_res_65536_1048576_merged.bin"

        # FineVision merged file (in FineVision directory)
        "${BASE_DIR}/FineVision:finevision_merged.bin"
    )
else
    # Parse user-provided datasets
    IFS=' ' read -r -a SOURCE_DATASETS <<< "${SOURCE_DATASETS}"
fi

# Set Python path
export PYTHONPATH="${MEGATRON_PATH}:${PYTHONPATH}"

# Create output directory
mkdir -p "${OUTPUT_BASE}"

echo "=================================================================================="
echo "Multi-Dataset Merge Configuration"
echo "=================================================================================="
echo "Megatron Path: ${MEGATRON_PATH}"
echo "Output Directory: ${OUTPUT_BASE}"
echo "Output Name: ${OUTPUT_NAME}"
echo "Number of source datasets: ${#SOURCE_DATASETS[@]}"
echo "=================================================================================="
echo "Source Datasets:"
for dataset in "${SOURCE_DATASETS[@]}"; do
    echo "  - ${dataset}"
done
echo "=================================================================================="

# Function to collect files from a source
collect_source_files() {
    local source_spec=$1
    local temp_dir=$2
    local index=$3

    # Split source_spec into path and pattern
    IFS=':' read -r source_path pattern <<< "${source_spec}"

    echo "Processing source ${index}: ${source_path}"

    # Check if source exists
    if [ ! -d "${source_path}" ]; then
        echo "  ✗ Directory not found: ${source_path}"
        return 1
    fi

    # Find matching files
    local bin_files=()
    if [ "${pattern}" = "*" ]; then
        # Match all .bin files
        mapfile -t bin_files < <(find "${source_path}" -maxdepth 1 -name "*.bin" -type f 2>/dev/null | sort)
    else
        # Use pattern matching
        mapfile -t bin_files < <(find "${source_path}" -maxdepth 1 -name "${pattern}" -type f 2>/dev/null | sort)
    fi

    local num_files=${#bin_files[@]}

    if [ ${num_files} -eq 0 ]; then
        echo "  ✗ No matching files found with pattern: ${pattern}"
        return 1
    fi

    echo "  ✓ Found ${num_files} .bin files"

    # Calculate total tokens from bin file sizes (4 bytes per token)
    local total_bytes=0
    for bin_file in "${bin_files[@]}"; do
        local file_size=$(stat -c%s "${bin_file}" 2>/dev/null || stat -f%z "${bin_file}" 2>/dev/null)
        total_bytes=$((total_bytes + file_size))
    done
    local total_tokens=$((total_bytes / 4))
    echo "  ℹ Total tokens in source: $(printf "%'d" ${total_tokens})"

    # Create symlinks with unique names to avoid conflicts
    local file_count=0
    for bin_file in "${bin_files[@]}"; do
        local base_name=$(basename "${bin_file}" .bin)
        local idx_file="${bin_file%.bin}.idx"

        if [ ! -f "${idx_file}" ]; then
            echo "  ⚠ Missing .idx file for ${base_name}.bin, skipping..."
            continue
        fi

        # Create uniquely named symlinks (prefix with source index)
        ln -sf "${bin_file}" "${temp_dir}/src${index}_${file_count}_${base_name}.bin"
        ln -sf "${idx_file}" "${temp_dir}/src${index}_${file_count}_${base_name}.idx"
        ((file_count++))
    done

    echo "  ✓ Created ${file_count} file pairs in staging area"

    return 0
}

# Main merging process
main() {
    echo ""
    echo "Starting merge process..."
    echo "--------------------------------------------------------------------------------"

    # Create temporary staging directory
    TEMP_DIR="/tmp/merge_all_$$"
    mkdir -p "${TEMP_DIR}"

    # Collect all source files
    echo "Stage 1: Collecting source files"
    echo "--------------------------------------------------------------------------------"

    local success_count=0
    local total_sources=${#SOURCE_DATASETS[@]}

    for i in "${!SOURCE_DATASETS[@]}"; do
        if collect_source_files "${SOURCE_DATASETS[$i]}" "${TEMP_DIR}" "$((i+1))"; then
            ((success_count++))
        fi
        echo ""
    done

    if [ ${success_count} -eq 0 ]; then
        echo "ERROR: No valid source files found"
        rm -rf "${TEMP_DIR}"
        exit 1
    fi

    echo "Successfully collected files from ${success_count}/${total_sources} sources"
    echo ""

    # Count total files to merge
    local total_bin_files=$(ls "${TEMP_DIR}"/*.bin 2>/dev/null | wc -l)
    local total_idx_files=$(ls "${TEMP_DIR}"/*.idx 2>/dev/null | wc -l)

    # Calculate total input tokens
    local total_input_bytes=0
    for bin_file in "${TEMP_DIR}"/*.bin; do
        if [ -L "${bin_file}" ]; then
            # Follow symlink to get actual file size
            actual_file=$(readlink -f "${bin_file}")
            file_size=$(stat -c%s "${actual_file}" 2>/dev/null || stat -f%z "${actual_file}" 2>/dev/null)
            total_input_bytes=$((total_input_bytes + file_size))
        fi
    done
    local total_input_tokens=$((total_input_bytes / 4))

    echo "Stage 2: Merging datasets"
    echo "--------------------------------------------------------------------------------"
    echo "Total .bin files to merge: ${total_bin_files}"
    echo "Total .idx files to merge: ${total_idx_files}"
    echo "Total input tokens: $(printf "%'d" ${total_input_tokens})"
    echo "Total input size: $(printf "%.2f GB" $(echo "scale=2; ${total_input_bytes} / 1024 / 1024 / 1024" | bc))"

    if [ ${total_bin_files} -eq 0 ]; then
        echo "ERROR: No .bin files found in staging directory"
        rm -rf "${TEMP_DIR}"
        exit 1
    fi

    # Perform the merge
    OUTPUT_PATH="${OUTPUT_BASE}/${OUTPUT_NAME}"

    echo "Output will be saved to: ${OUTPUT_PATH}.bin and ${OUTPUT_PATH}.idx"
    echo ""

    echo "Running merge command..."
    python "${MEGATRON_PATH}/scripts/merge_datasets/merge_datasets.py" \
        --input "${TEMP_DIR}" \
        --output-prefix "${OUTPUT_PATH}"

    merge_status=$?

    # Calculate output statistics
    if [ ${merge_status} -eq 0 ]; then
        echo ""
        echo "Stage 3: Calculating final statistics"
        echo "--------------------------------------------------------------------------------"

        # Get output file sizes and calculate tokens
        output_bin="${OUTPUT_PATH}.bin"
        output_idx="${OUTPUT_PATH}.idx"

        if [ -f "${output_bin}" ]; then
            bin_size=$(stat -c%s "${output_bin}" 2>/dev/null || stat -f%z "${output_bin}" 2>/dev/null)
            output_tokens=$((bin_size / 4))

            echo "Output Statistics:"
            echo "  Output .bin file size: $(printf "%.2f GB" $(echo "scale=2; ${bin_size} / 1024 / 1024 / 1024" | bc))"
            echo "  Total tokens in output: $(printf "%'d" ${output_tokens})"

            # Calculate merge efficiency
            if [ ${total_input_tokens} -gt 0 ]; then
                efficiency=$(echo "scale=2; ${output_tokens} * 100 / ${total_input_tokens}" | bc)
                echo "  Merge efficiency: ${efficiency}% (output/input tokens)"
            fi

            # Create dataset info JSON with accurate token counts
            python3 << EOF
import json
import os
from datetime import datetime
from pathlib import Path

output_base = "${OUTPUT_BASE}"
output_name = "${OUTPUT_NAME}"
bin_size = ${bin_size}
output_tokens = ${output_tokens}
total_input_tokens = ${total_input_tokens}
temp_dir = "${TEMP_DIR}"

# Parse source datasets and calculate their tokens
source_stats = []
source_datasets = """${SOURCE_DATASETS[@]}""".split()

for source in source_datasets:
    if ':' in source:
        path, pattern = source.rsplit(':', 1)
        # Find the actual bin files
        import glob
        bin_files = glob.glob(os.path.join(path, pattern))

        total_source_tokens = 0
        for bin_file in bin_files:
            if os.path.exists(bin_file):
                file_size = os.path.getsize(bin_file)
                tokens = file_size // 4
                total_source_tokens += tokens

        if total_source_tokens > 0:
            source_stats.append({
                'path': path,
                'pattern': pattern,
                'files': len(bin_files),
                'tokens': total_source_tokens,
                'size_gb': round(sum(os.path.getsize(f) for f in bin_files if os.path.exists(f)) / (1024**3), 2)
            })

# Create final dataset info
final_info = {
    'timestamp': datetime.now().isoformat(),
    'dataset_name': output_name,
    'merge_configuration': {
        'sources': ${#SOURCE_DATASETS[@]},
        'source_paths': ["${SOURCE_DATASETS[@]}"],
        'output_path': os.path.join(output_base, output_name),
        'megatron_path': "${MEGATRON_PATH}"
    },
    'statistics': {
        'input_tokens': total_input_tokens,
        'output_tokens': output_tokens,
        'merge_efficiency_percent': round(output_tokens * 100 / total_input_tokens, 2) if total_input_tokens > 0 else 0,
        'sources_detail': source_stats
    },
    'output_files': {
        'bin_file': f"{output_name}.bin",
        'bin_size_bytes': bin_size,
        'bin_size_gb': round(bin_size / (1024**3), 2),
        'idx_file': f"{output_name}.idx",
        'total_tokens': output_tokens
    }
}

# Save dataset info
info_output = os.path.join(output_base, f"{output_name}_info.json")
with open(info_output, 'w') as f:
    json.dump(final_info, f, indent=2)

print(f"\n✓ Dataset info saved to {info_output}")
EOF

        else
            echo "ERROR: Output .bin file not found"
            merge_status=1
        fi
    fi

    # Clean up temp directory
    echo ""
    echo "Cleaning up temporary files..."
    rm -rf "${TEMP_DIR}"

    if [ ${merge_status} -eq 0 ]; then
        echo ""
        echo "=================================================================================="
        echo "✓ MERGE COMPLETED SUCCESSFULLY"
        echo "=================================================================================="
        echo "Output files:"
        ls -lh "${OUTPUT_PATH}".*
        echo "=================================================================================="
    else
        echo ""
        echo "=================================================================================="
        echo "✗ MERGE FAILED"
        echo "=================================================================================="
        exit ${merge_status}
    fi
}

# Run main function
main