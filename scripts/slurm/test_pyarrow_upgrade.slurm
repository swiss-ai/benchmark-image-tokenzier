#!/bin/bash
# One-off test: in the NeMo 25.11 container's /opt/venv, upgrade pyarrow and
# verify that the failing dailymed interleaved2 parquet now reads cleanly.
#
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5
#SBATCH --job-name=test-pyarrow-up
#SBATCH --environment=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/scripts/envs/nemo_25_11.toml
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/test_pyarrow_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/test_pyarrow_%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=8
#SBATCH --partition=normal

set -euo pipefail

srun -u bash -c '
  set -e
  echo "=== container python + pyarrow BEFORE upgrade ==="
  /opt/venv/bin/python3 -c "import sys, pyarrow; print(\"python:\", sys.version.split()[0]); print(\"pyarrow:\", pyarrow.__version__)"

  echo
  echo "=== try reading the failing dailymed shard with current pyarrow ==="
  /opt/venv/bin/python3 -c "
import pyarrow.parquet as pq
try:
    t = pq.read_table(\"/capstor/store/cscs/swissai/infra01/medical-datasets/raw/dailymed_spl/interleaved2/part-00000.parquet\")
    print(\"  → BEFORE: OK,\", t.num_rows, \"rows,\", t.column_names)
except Exception as e:
    print(\"  → BEFORE: FAIL:\", type(e).__name__, str(e)[:160])
"

  echo
  echo "=== upgrade pyarrow ==="
  uv pip install --python /opt/venv/bin/python3 --upgrade "pyarrow>=22"

  echo
  echo "=== container python + pyarrow AFTER upgrade ==="
  /opt/venv/bin/python3 -c "import pyarrow; print(\"pyarrow:\", pyarrow.__version__)"

  echo
  echo "=== retry reading the failing dailymed shard with column filter ==="
  /opt/venv/bin/python3 -c "
import pyarrow.parquet as pq
pf = pq.ParquetFile(\"/capstor/store/cscs/swissai/infra01/medical-datasets/raw/dailymed_spl/interleaved2/part-00000.parquet\")
t = pf.read_row_group(0, columns=[\"id\", \"segments_json\", \"images_bytes\"])
print(\"  → AFTER: OK,\", t.num_rows, \"rows,\", t.column_names)
print(\"  → images_bytes type:\", t.schema.field(\"images_bytes\").type)
print(\"  → first row n_images:\", len(t[\"images_bytes\"][0]))
"

  echo
  echo "=== sanity check that other deps still import ==="
  /opt/venv/bin/python3 -c "
import pyarrow, torch, transformers
print(\"  pyarrow:\", pyarrow.__version__, \"torch:\", torch.__version__, \"transformers:\", transformers.__version__)
"

  echo
  echo "DONE — pyarrow upgrade verified."
'
