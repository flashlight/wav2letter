#!/bin/bash

## SLURM script for running decoding with wav2letter
#SBATCH --job-name=wav2letter-decode
## %j is the job id, %u is the user id
#SBATCH --output=/checkpoint/%u/jobs/wav2letter/decode/wav2letter-decode-%j.out
#SBATCH --error=/checkpoint/%u/jobs/wav2letter/decode/wav2letter-decode-%j.err
## partition name
#SBATCH --partition=scavenge
## number of nodes
# We only need one node, no gpus
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:0
## number of tasks per node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task 80
#SBATCH --time=12:00:00

# Runs on cpu
~/wav2letter/build/Decoder \
    -nthread_decoder 80 \
    -show \
    -criterion ctc \
    -letters /private/home/jacobkahn/letter-dictionary.txt \
    -words /private/home/jacobkahn/w2l-dict-openseq2seq-modified.txt \
    -emission_dir /private/home/qiantong/dev_clean_emissions \
    --logtostderr=1 \
    -lm /private/home/qiantong/OpenSeq2Seq/language_model/4-gram.binary \
    -smearing max \
    -beamsize 100 \
    -lmweight 3 \
    -silweight -0.2 \
    -wordscore 2.5 \
    -beamscore 20 \
