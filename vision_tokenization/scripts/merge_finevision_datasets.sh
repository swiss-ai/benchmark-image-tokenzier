#!/bin/bash
#SBATCH --account=a-infra01
#SBATCH --job-name=emu3-merge
#SBATCH --environment=nemo
#SBATCH --nodes=1
#SBATCH --partition=debug
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=01:30:00
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/benchmark-image-tokenzier/vision_tokenization/logs/merge_datasets_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/benchmark-image-tokenzier/vision_tokenization/logs/merge_datasets_%j.err

###################### Merge FineVision Tokenized Datasets ######################

# Configuration
MEGATRON_PATH="/iopsstor/scratch/cscs/xyixuan/Megatron-LM"
FINEVISION_BASE="/capstor/store/cscs/swissai/infra01/vision-datasets/FineVision"
TOKENIZED_DIR="${FINEVISION_BASE}/tokenized"
MERGED_DIR="${FINEVISION_BASE}/merged"

# Datasets to merge (customize via environment variable)
DATASETS="${DATASETS:-densefusion_1m laion_gpt4v latexformulas objects365_qa}"

# Merge mode: "subsets" (only merge individual datasets) or "all" (merge subsets + create combined)
MERGE_MODE="${MERGE_MODE:-all}"

# Output name for combined dataset
COMBINED_NAME="${COMBINED_NAME:-finevision_merged}"

# Set Python path
export PYTHONPATH="${MEGATRON_PATH}:${PYTHONPATH}"

# Create directories
mkdir -p "${MERGED_DIR}"

echo "=================================================================================="
echo "FineVision Dataset Merging"
echo "=================================================================================="
echo "Tokenized Directory: ${TOKENIZED_DIR}"
echo "Merged Directory: ${MERGED_DIR}"
echo "Datasets: ${DATASETS}"
echo "Merge Mode: ${MERGE_MODE}"
[ "${MERGE_MODE}" == "all" ] && echo "Combined Output: ${COMBINED_NAME}"
echo "=================================================================================="

# Function to merge a single dataset's rank files
merge_dataset() {
    local name=$1
    local input="${TOKENIZED_DIR}/${name}"
    local output="${MERGED_DIR}/${name}"
    
    # Check if input exists
    if [ ! -d "${input}" ]; then
        echo "  ✗ ${name}: Directory not found"
        return 1
    fi
    
    # Check if already merged
    if [ -f "${output}.bin" ] && [ -f "${output}.idx" ]; then
        echo "  ✓ ${name}: Already merged"
        return 0
    fi
    
    # Check for rank files
    if ! ls "${input}"/rank_*.bin &>/dev/null; then
        echo "  ✗ ${name}: No rank files found"
        return 1
    fi
    
    # Create temp directory with symlinks to only rank files
    temp_dir="/tmp/merge_${name}_$$"
    mkdir -p "${temp_dir}"
    
    # Symlink only rank_*.bin and rank_*.idx files
    for f in "${input}"/rank_*.bin "${input}"/rank_*.idx; do
        [ -f "$f" ] && ln -s "$f" "${temp_dir}/$(basename $f)"
    done
    
    # Perform merge using temp directory
    echo "  ⚙ ${name}: Merging $(ls ${temp_dir}/rank_*.bin 2>/dev/null | wc -l) rank files..."
    python "${MEGATRON_PATH}/scripts/merge_datasets/merge_datasets.py" \
        --input "${temp_dir}" \
        --output-prefix "${output}"
    
    merge_status=$?
    
    # Clean up temp directory
    rm -rf "${temp_dir}"
    
    if [ ${merge_status} -eq 0 ]; then
        size=$(ls -lh "${output}.bin" | awk '{print $5}')
        echo "  ✓ ${name}: Merged successfully (${size})"
        return 0
    else
        echo "  ✗ ${name}: Merge failed"
        return 1
    fi
}

# Step 1: Merge individual datasets
echo ""
echo "Step 1: Merging Individual Datasets"
echo "------------------------------------"

success_count=0
for dataset in ${DATASETS}; do
    merge_dataset "${dataset}" && ((success_count++))
done

echo ""
echo "Merged ${success_count} out of $(echo ${DATASETS} | wc -w) datasets"

if [ ${success_count} -eq 0 ]; then
    echo "ERROR: No datasets were successfully merged"
    exit 1
fi

# Step 2: Create combined dataset (if requested)
if [ "${MERGE_MODE}" == "all" ]; then
    echo ""
    echo "Step 2: Creating Combined Dataset"
    echo "----------------------------------"
    
    output_path="${FINEVISION_BASE}/${COMBINED_NAME}"
    
    # Check if already exists
    if [ -f "${output_path}.bin" ] && [ -f "${output_path}.idx" ]; then
        echo "Combined dataset already exists at ${output_path}"
    else
        echo "Merging all subsets into ${COMBINED_NAME}..."
        python "${MEGATRON_PATH}/scripts/merge_datasets/merge_datasets.py" \
            --input "${MERGED_DIR}" \
            --output-prefix "${output_path}"
        
        if [ $? -eq 0 ]; then
            size=$(ls -lh "${output_path}.bin" | awk '{print $5}')
            echo "✓ Combined dataset created successfully (${size})"
        else
            echo "✗ Failed to create combined dataset"
            exit 1
        fi
    fi
fi

# Summary
echo ""
echo "=================================================================================="
echo "Summary"
echo "=================================================================================="
echo "Merged subsets in ${MERGED_DIR}:"
ls -lh "${MERGED_DIR}"/*.bin 2>/dev/null | awk '{print "  - "$9": "$5}'

if [ "${MERGE_MODE}" == "all" ] && [ -f "${FINEVISION_BASE}/${COMBINED_NAME}.bin" ]; then
    echo ""
    echo "Combined dataset:"
    ls -lh "${FINEVISION_BASE}/${COMBINED_NAME}.bin" | awk '{print "  - "$9": "$5}'
fi

echo ""
echo "Completed at $(date)"
echo "===================================================================================="