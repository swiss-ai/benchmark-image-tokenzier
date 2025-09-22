#!/bin/bash
#SBATCH --account=a-infra01
#SBATCH --job-name=imagenet-merge
#SBATCH --environment=nemo
#SBATCH --nodes=1
#SBATCH --partition=debug
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=01:30:00
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/benchmark-image-tokenzier/vision_tokenization/logs/merge_imagenet_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/benchmark-image-tokenzier/vision_tokenization/logs/merge_imagenet_%j.err

###################### Merge ImageNet Tokenized Datasets ######################

# Configuration
MEGATRON_PATH="/iopsstor/scratch/cscs/xyixuan/Megatron-LM"
IMAGENET_BASE="/capstor/store/cscs/swissai/infra01/vision-datasets"
TOKENIZED_DIR="${IMAGENET_BASE}/tokenized"
MERGED_DIR="${IMAGENET_BASE}/merged"

# ImageNet dataset directories to merge
IMAGENET_DIRS="${IMAGENET_DIRS:-imagenet-w21_range_0-200_res_65536_1048576 imagenet-w21_range_200-2048_res_65536_1048576}"

# Output name for combined dataset - using the format: imagenet-w21_range_0-2048_res_65536_1048576_merged
COMBINED_NAME="${COMBINED_NAME:-imagenet-w21_range_0-2048_res_65536_1048576_merged}"

# Set Python path
export PYTHONPATH="${MEGATRON_PATH}:${PYTHONPATH}"

# Create directories
mkdir -p "${MERGED_DIR}"

echo "=================================================================================="
echo "ImageNet Dataset Merging"
echo "=================================================================================="
echo "Tokenized Directory: ${TOKENIZED_DIR}"
echo "Merged Directory: ${MERGED_DIR}"
echo "ImageNet Directories: ${IMAGENET_DIRS}"
echo "Combined Output: ${COMBINED_NAME}"
echo "=================================================================================="

# Function to merge a single ImageNet range directory
merge_imagenet_range() {
    local dir_name=$1
    local input="${TOKENIZED_DIR}/${dir_name}"
    local output="${MERGED_DIR}/${dir_name}"

    # Check if input exists
    if [ ! -d "${input}" ]; then
        echo "  ✗ ${dir_name}: Directory not found"
        return 1
    fi

    # Check if already merged
    if [ -f "${output}.bin" ] && [ -f "${output}.idx" ]; then
        echo "  ✓ ${dir_name}: Already merged"
        return 0
    fi

    # Check for shard files (imagenet_w21-train-*.bin pattern)
    num_bin_files=$(ls "${input}"/imagenet_w21-train-*.bin 2>/dev/null | wc -l)

    if [ ${num_bin_files} -eq 0 ]; then
        echo "  ✗ ${dir_name}: No shard files found"
        return 1
    fi

    echo "  ⚙ ${dir_name}: Merging ${num_bin_files} shard files..."

    # Create temp directory with symlinks (similar to CC12M approach)
    temp_dir="/tmp/merge_imagenet_${dir_name}_$$"
    mkdir -p "${temp_dir}"

    # Symlink all imagenet_w21-train-*.bin and .idx files
    for f in "${input}"/imagenet_w21-train-*.bin "${input}"/imagenet_w21-train-*.idx; do
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
        echo "  ✓ ${dir_name}: Merged successfully (${size})"
        return 0
    else
        echo "  ✗ ${dir_name}: Merge failed"
        return 1
    fi
}

# Step 1: Merge individual ImageNet range directories
echo ""
echo "Step 1: Merging Individual ImageNet Range Directories"
echo "------------------------------------------------------"

success_count=0
failed_dirs=""

for dir in ${IMAGENET_DIRS}; do
    merge_imagenet_range "${dir}"
    if [ $? -eq 0 ]; then
        ((success_count++))
    else
        failed_dirs="${failed_dirs} ${dir}"
    fi
done

echo ""
echo "Merged ${success_count} out of $(echo ${IMAGENET_DIRS} | wc -w) ImageNet directories"

if [ -n "${failed_dirs}" ]; then
    echo "Failed directories:${failed_dirs}"
fi

if [ ${success_count} -eq 0 ]; then
    echo "ERROR: No directories were successfully merged"
    exit 1
fi

# Step 2: Create combined dataset from all ImageNet ranges
echo ""
echo "Step 2: Creating Combined ImageNet Dataset"
echo "-------------------------------------------"

output_path="${IMAGENET_BASE}/${COMBINED_NAME}"

# Check if already exists
if [ -f "${output_path}.bin" ] && [ -f "${output_path}.idx" ]; then
    echo "Combined dataset already exists at ${output_path}"
    echo "Delete it first if you want to recreate it"
else
    echo "Merging all ImageNet ranges into ${COMBINED_NAME}..."

    # Check if merged directory has files
    num_merged=0
    for dir in ${IMAGENET_DIRS}; do
        if [ -f "${MERGED_DIR}/${dir}.bin" ]; then
            ((num_merged++))
        fi
    done

    if [ ${num_merged} -eq 0 ]; then
        echo "No merged ImageNet files found in ${MERGED_DIR}"
        exit 1
    fi

    # Create temp directory with symlinks to merged files only
    temp_combined_dir="/tmp/merge_imagenet_combined_$$"
    mkdir -p "${temp_combined_dir}"

    for dir in ${IMAGENET_DIRS}; do
        if [ -f "${MERGED_DIR}/${dir}.bin" ] && [ -f "${MERGED_DIR}/${dir}.idx" ]; then
            ln -s "${MERGED_DIR}/${dir}.bin" "${temp_combined_dir}/$(basename ${dir}).bin"
            ln -s "${MERGED_DIR}/${dir}.idx" "${temp_combined_dir}/$(basename ${dir}).idx"
        fi
    done

    python "${MEGATRON_PATH}/scripts/merge_datasets/merge_datasets.py" \
        --input "${temp_combined_dir}" \
        --output-prefix "${output_path}"

    merge_status=$?

    # Clean up temp directory
    rm -rf "${temp_combined_dir}"

    if [ ${merge_status} -eq 0 ]; then
        size=$(ls -lh "${output_path}.bin" | awk '{print $5}')
        echo "✓ Combined dataset created successfully (${size})"
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
echo "Merged ImageNet range datasets in ${MERGED_DIR}:"
for dir in ${IMAGENET_DIRS}; do
    if [ -f "${MERGED_DIR}/${dir}.bin" ]; then
        size=$(ls -lh "${MERGED_DIR}/${dir}.bin" 2>/dev/null | awk '{print $5}')
        echo "  - ${dir}: ${size}"
    fi
done

if [ -f "${output_path}.bin" ]; then
    echo ""
    echo "Combined dataset:"
    ls -lh "${output_path}.bin" | awk '{print "  - "'${COMBINED_NAME}'": "$5}'
fi

echo ""
echo "Completed at $(date)"
echo "===================================================================================="