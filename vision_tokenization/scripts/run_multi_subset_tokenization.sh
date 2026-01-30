#!/bin/bash
set -euo pipefail

###################### Multi-Subset Tokenization Launcher ######################
# Submits one SLURM job per dataset config/subset name, each running the
# tokenization pipeline with the given base config and overridden config_name.
#
# Config names can be provided inline (comma-separated) or via a text file
# (one name per line, blank lines and #-comments are ignored).
#
# Usage:
#   # Inline config names
#   ./run_multi_subset_tokenization.sh \
#       --config /path/to/config.json \
#       --config-names "subset_a,subset_b,subset_c"
#
#   # Config names from file
#   ./run_multi_subset_tokenization.sh \
#       --config /path/to/config.json \
#       --config-names-file /path/to/subsets.txt
#
#   # With overrides
#   ./run_multi_subset_tokenization.sh \
#       --config /path/to/config.json \
#       --config-names-file /path/to/subsets.txt \
#       --num-shards 200 \
#       --nodes 4 \
#       --resume
#
#   # Dry run (print sbatch commands without submitting)
#   ./run_multi_subset_tokenization.sh \
#       --config /path/to/config.json \
#       --config-names "subset_a,subset_b" \
#       --dry-run
#
# Config names file format (subsets.txt):
#   # Lines starting with '#' are comments
#   subset_a
#   subset_b
#   subset_c
#
################################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/run_subset_tokenization.slurm"

# Defaults
CONFIG_PATH=""
CONFIG_NAMES=()
CONFIG_NAMES_FILE=""
NUM_SHARDS=""
NODES=""
RESUME=false
OFFLINE=false
TIME=""
PARTITION=""
ACCOUNT=""
DRY_RUN=false

usage() {
    cat <<'USAGE'
Usage: run_multi_subset_tokenization.sh [OPTIONS]

Required (one of):
  --config-names LIST         Comma-separated list of config/subset names
  --config-names-file PATH    Path to text file with one config name per line

Required:
  --config PATH               Path to base JSON config file

Optional:
  --num-shards N              Override num_shards for all subsets
  --nodes N                   Number of SLURM nodes per job (default: from SBATCH directive)
  --resume                    Resume from checkpoint (skip completed shards)
  --offline                   Enable HF datasets offline mode
  --time LIMIT                SLURM time limit, e.g. "12:00:00" (default: from SBATCH directive)
  --partition NAME            SLURM partition (default: from SBATCH directive)
  --account NAME              SLURM account (default: from SBATCH directive)
  --dry-run                   Print sbatch commands without submitting
  -h, --help                  Show this help message
USAGE
}

# ── Parse arguments ──────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --config-names)
            IFS=',' read -ra CONFIG_NAMES <<< "$2"
            shift 2
            ;;
        --config-names-file)
            CONFIG_NAMES_FILE="$2"
            shift 2
            ;;
        --num-shards)
            NUM_SHARDS="$2"
            shift 2
            ;;
        --nodes)
            NODES="$2"
            shift 2
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --offline)
            OFFLINE=true
            shift
            ;;
        --time)
            TIME="$2"
            shift 2
            ;;
        --partition)
            PARTITION="$2"
            shift 2
            ;;
        --account)
            ACCOUNT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

# ── Validate ─────────────────────────────────────────────────────────────────

if [ -z "${CONFIG_PATH}" ]; then
    echo "ERROR: --config is required"
    usage
    exit 1
fi

if [ ! -f "${CONFIG_PATH}" ]; then
    echo "ERROR: Config file not found: ${CONFIG_PATH}"
    exit 1
fi

# Read config names from file if provided
if [ -n "${CONFIG_NAMES_FILE}" ]; then
    if [ ! -f "${CONFIG_NAMES_FILE}" ]; then
        echo "ERROR: Config names file not found: ${CONFIG_NAMES_FILE}"
        exit 1
    fi
    while IFS= read -r line || [ -n "$line" ]; do
        # Trim whitespace
        line="$(echo "$line" | xargs)"
        # Skip empty lines and comments
        if [ -z "$line" ] || [[ "$line" == \#* ]]; then
            continue
        fi
        CONFIG_NAMES+=("$line")
    done < "${CONFIG_NAMES_FILE}"
fi

if [ ${#CONFIG_NAMES[@]} -eq 0 ]; then
    echo "ERROR: No config names provided. Use --config-names or --config-names-file"
    usage
    exit 1
fi

if [ ! -f "${SLURM_SCRIPT}" ]; then
    echo "ERROR: SLURM script not found: ${SLURM_SCRIPT}"
    echo "Expected at: ${SLURM_SCRIPT}"
    exit 1
fi

# ── Build common sbatch flags ────────────────────────────────────────────────

SBATCH_FLAGS=""

if [ -n "${NODES}" ]; then
    SBATCH_FLAGS="${SBATCH_FLAGS} --nodes=${NODES}"
fi

if [ -n "${TIME}" ]; then
    SBATCH_FLAGS="${SBATCH_FLAGS} --time=${TIME}"
fi

if [ -n "${PARTITION}" ]; then
    SBATCH_FLAGS="${SBATCH_FLAGS} --partition=${PARTITION}"
fi

if [ -n "${ACCOUNT}" ]; then
    SBATCH_FLAGS="${SBATCH_FLAGS} --account=${ACCOUNT}"
fi

# ── Submit jobs ──────────────────────────────────────────────────────────────

# Make config path absolute
CONFIG_PATH="$(cd "$(dirname "${CONFIG_PATH}")" && pwd)/$(basename "${CONFIG_PATH}")"

echo "=================================================================================="
echo "Multi-Subset Tokenization Launcher"
echo "=================================================================================="
echo "Base config:     ${CONFIG_PATH}"
echo "Config names:    ${CONFIG_NAMES[*]}"
echo "Num subsets:     ${#CONFIG_NAMES[@]}"
echo "Num shards:      ${NUM_SHARDS:-<from config>}"
echo "Nodes per job:   ${NODES:-<from SBATCH directive>}"
echo "Resume:          ${RESUME}"
echo "Offline:         ${OFFLINE}"
echo "Dry run:         ${DRY_RUN}"
echo "=================================================================================="
echo ""

SUBMITTED_JOBS=()

for CONFIG_NAME in "${CONFIG_NAMES[@]}"; do
    # Build export list for this job
    EXPORT_VARS="ALL,CONFIG_PATH=${CONFIG_PATH},CONFIG_NAME=${CONFIG_NAME}"

    if [ -n "${NUM_SHARDS}" ]; then
        EXPORT_VARS="${EXPORT_VARS},NUM_SHARDS_OVERRIDE=${NUM_SHARDS}"
    fi

    if [ "${RESUME}" = true ]; then
        EXPORT_VARS="${EXPORT_VARS},RESUME_MODE=true"
    fi

    if [ "${OFFLINE}" = true ]; then
        EXPORT_VARS="${EXPORT_VARS},OFFLINE_MODE=true"
    fi

    JOB_NAME="tok-${CONFIG_NAME}"

    SBATCH_CMD="sbatch --job-name=${JOB_NAME} --export=${EXPORT_VARS}${SBATCH_FLAGS:+ ${SBATCH_FLAGS}} ${SLURM_SCRIPT}"

    if [ "${DRY_RUN}" = true ]; then
        echo "[DRY RUN] ${SBATCH_CMD}"
    else
        echo "Submitting: ${CONFIG_NAME}"
        OUTPUT=$(eval "${SBATCH_CMD}")
        JOB_ID=$(echo "${OUTPUT}" | awk '{print $NF}')
        echo "  -> ${OUTPUT}"
        SUBMITTED_JOBS+=("${JOB_ID}:${CONFIG_NAME}")
    fi
done

echo ""
echo "=================================================================================="
if [ "${DRY_RUN}" = true ]; then
    echo "DRY RUN complete. ${#CONFIG_NAMES[@]} jobs would be submitted."
else
    echo "Submitted ${#SUBMITTED_JOBS[@]} jobs:"
    for entry in "${SUBMITTED_JOBS[@]}"; do
        JOB_ID="${entry%%:*}"
        NAME="${entry#*:}"
        echo "  Job ${JOB_ID} -> ${NAME}"
    done
    echo ""
    # Build comma-separated job ID list for sacct
    JOB_IDS=""
    for entry in "${SUBMITTED_JOBS[@]}"; do
        JOB_IDS="${JOB_IDS:+${JOB_IDS},}${entry%%:*}"
    done
    echo "Monitor with: squeue -u ${USER}"
    echo "Or:           sacct -j ${JOB_IDS}"
fi
echo "=================================================================================="