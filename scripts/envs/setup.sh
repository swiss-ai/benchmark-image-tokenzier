# Shared job setup: PYTHONPATH plus the locked packages the container does not ship.
#
# Usage, from inside the srun body:
#     source /path/to/scripts/envs/setup.sh          # tokenization jobs
#     source /path/to/scripts/envs/setup.sh ops      # merge/postprocess jobs
#
# Regenerate the locks after editing a .in file, inside the container:
#     uv pip compile requirements.in -o requirements.lock \
#         --python /opt/venv/bin/python3 --generate-hashes

_envs_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
_repo_dir=$(cd "${_envs_dir}/../.." && pwd)

export PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/apertus/Megatron-LM:${_repo_dir}:${PYTHONPATH:-}

uv pip install --python /opt/venv/bin/python3 --quiet --require-hashes \
    -r "${_envs_dir}/requirements${1:+-$1}.lock"
