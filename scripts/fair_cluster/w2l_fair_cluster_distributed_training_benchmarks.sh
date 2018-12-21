#!/bin/bash

# Runs wav2letter in many different GPU/node configurations for benchmarking
# distributed training/performance across many GPUs/nodes in the FAIR Cluster.
# - Runs with 2^k GPUs for k in [1, 6]
# - Assumes a SLURM-ready training script is in ~/wav2letter-train.sh
# - Since single-GPU training requres a w2l flag changes, do manually

# For a single node, run in 2, 4, 8 configurations
single_node_gpu_quantities=("2" "4" "8")

# Number of node configurations besides 1
node_quantities=("2" "4" "8")

# For SLURM sbatch, any variables passed from the command line
# will take priority over variables defined by SBATCH
# Flags that will not be overriden: partition, time, ntasks-per-node
# see wav2letter-slurm.sh for defaults

# Prefixes
log_path_prefix=/checkpoint/%u/jobs/wav2letter/train-benchmark

# Custom text for run name
run_name_details=my-special-model

# Run single node jobs
# Iterate over num gpus
for s in "${single_node_gpu_quantities[@]}"
do
    name="w2l-${run_name_details}-1-node-${s}-gpu"
    echo "Starting job $name"
    sbatch \
        --job-name="${name}" \
        --output="${log_path_prefix}/${name}-%j.out" \
        --error="${log_path_prefix}/${name}-%j.err" \
        --nodes=1 \
        --gres="gpu:volta:${s}" \
        --ntasks-per-node="${s}" \
        --priority=priority \
        --comment="Running benchmark - must avoid preemption " \
        $HOME/wav2letter/scripts/fair_cluster/wav2letter-train.sh
done

# Run multi-node jobs. Always use 8 GPUs per node
for s in "${node_quantities[@]}"
do
    name="w2l-${run_name_details}-${s}-node-8-gpu"
    echo "Starting job $name"
    sbatch \
        --job-name="${name}" \
        --output="${log_path_prefix}/${name}-%j.out" \
        --error="${log_path_prefix}/${name}-%j.err" \
        --nodes="${s}" \
        --gres="gpu:volta:8" \
        --ntasks-per-node="8" \
        --partition=priority \
        --comment="Running benchmark - must avoid preemption " \
        $HOME/wav2letter/scripts/fair_cluster/wav2letter-train.sh
done
