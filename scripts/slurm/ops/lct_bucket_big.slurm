#!/bin/bash
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --job-name=lct-bucket
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/lct_bucket_%x_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/lct_bucket_%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --exclusive
#SBATCH --no-requeue
# Build medium + long buckets for one BIG lct dataset (sensenova / nemotron).
# Pass the dataset via:  sbatch --export=ALL,DS=<name> lct_bucket_big.slurm
set -e
: "${DS:?must pass DS=<dataset> via --export}"
export LD_LIBRARY_PATH=/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse:${LD_LIBRARY_PATH:-}
DRV=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/scripts/build_lct_bucket.sh
echo "=== lct bucketing for $DS (medium then long) on $(hostname) ==="
srun -u bash -c "bash $DRV $DS medium && bash $DRV $DS long"
echo "=== done $DS ==="
