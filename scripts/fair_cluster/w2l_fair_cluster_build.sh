#!/bin/bash

## SLURM script for building wav2letter
#SBATCH --job-name=wav2letter-build
## %j is the job id, %u is the user id
#SBATCH --output=/checkpoint/%u/jobs/wav2letter/build/wav2letter-build-%j.out
#SBATCH --error=/checkpoint/%u/jobs/wav2letter/build/wav2letter-build-%j.err
#SBATCH --partition=priority
#SBATCH --comment="Quick building wav2letter"
# Only one node and gpu needed for build
#SBATCH --nodes=1
# Use 1 attached Volta to more-easily auto-set nvcc flags
#SBATCH --gres=gpu:volta:1
#SBATCH --ntasks-per-node=1
# Build on 32 cores
#SBATCH --cpus-per-task 32
#SBATCH --time=12:00:00

### Set env vars
# Start clean
module purge
# Load Dependencies
module load cuda/9.2
module load cudnn/v7.1-cuda.9.2
module load NCCL/2.2.13-1-cuda.9.2
module load mkl/2018.0.128
module load openmpi/3.0.0/gcc.6.3.0
module load kenlm/110617/gcc.6.3.0

### Build
# Assumes dependencies have been installed in $HOME/usr. This is REQUIRED
# any dependencies which would otherwise install in locations like /usr
# aren't accessible from learnfair machines. Any dynamic libraries
# loaded at runtime must be in `/private/...` or another directory
# mounted by/accessible from learnfair machines.
export CMAKE_PREFIX_PATH="$HOME/usr"
cd "$HOME/wav2letter/build/" && cmake .. && make -j32
