#!/bin/bash
#SBATCH --job-name=eiger_mb
#SBATCH --account=sm94
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/eiger_mb/%x_%j.log
#SBATCH --error=/iopsstor/scratch/cscs/%u/eiger_mb/%x_%j.err
#SBATCH --environment=my_env

# Stop the script if a command fails or if an undefined variable is used
set -eo pipefail

chmod +x /users/${USER}/eiger/script/run_cudf_microbenchmarks.sh
mkdir -p /iopsstor/scratch/cscs/${USER}/eiger_mb

CMD="/users/${USER}/eiger/script/run_cudf_microbenchmarks.sh cscs build 0"
srun $CMD