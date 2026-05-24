#!/bin/bash
#SBATCH --job-name=hpc-llm-train
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# Single-node training template (8 GPUs on one node).
# Usage: sbatch scripts/slurm/single_node.sh

set -euo pipefail
mkdir -p logs

# Environment
module load cuda/12.1 || true
source ~/venvs/hpc-llm/bin/activate

# NCCL tuning
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=8

# Launch with torchrun (single-node, standalone rendezvous)
torchrun \
    --standalone \
    --nproc_per_node=8 \
    scripts/train_fsdp.py \
        --model gpt2-large \
        --batch-size 4 \
        --seq-len 1024 \
        --steps 1000 \
        --gradient-checkpointing
