#!/bin/bash

## SLURM script for running distributed training with wav2letter
#SBATCH --job-name=wav2letter-train
## %j is the job id, %u is the user id
#SBATCH --output=/checkpoint/%u/jobs/wav2letter/train/wav2letter-train-%j.out
#SBATCH --error=/checkpoint/%u/jobs/wav2letter/train/wav2letter-train-%j.err
## partition name
#SBATCH --partition=scavenge
## number of nodes
#SBATCH --nodes=1
# Example for 8 GPUs
#SBATCH --gres=gpu:volta:8
## number of tasks per node
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task 8
#SBATCH --time=12:00:00
# NOTE: if using fewer than 8 GPUs, the number of cpus-per-task MUST
# be equal to the number of GPUs


# Run job with mpi
# A few notes:
#   - Not using -np flag in MPI flag still signals the correct number of jobs
#   - Keep num_nodes at 1, even if using multi-node, because MPI will do
#     inter-node communication on the cluste
mpirun ~/wav2letter/build/Train train \
        -datadir /private/home/vineelkpratap/speech/pristine/ \
        -train wsj2/nov93dev \
        -valid wsj2/nov93dev \
        -tokensdir /checkpoint/jacobkahn/wav2letter/examples/dict \
        -tokens letters.lst \
        -archdir /checkpoint/jacobkahn/wav2letter/archfiles \
        -arch conv_20lyr_100mil \
        -lr 0.0001 \
        -target ltr \
        -nthread 16 \
        -criterion ctc \
        -batchsize 4 \
        -sqnorm \
        -mfsc \
        -onorm target \
        --logtostderr=1 \
        -rundir /checkpoint/jacobkahn/wav2letter/experiments/speech \
        -runname "w2l-train-${SLURM_JOB_ID}"  \
        --enable_distributed \
        --num_nodes=1 \
        -lsm
