#!/bin/bash
#SBATCH --job-name=hpc-llm-multinode
#SBATCH --nodes=4
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# Multi-node training template (4 nodes x 8 GPUs = 32 GPUs).
# Usage: sbatch scripts/slurm/multi_node.sh

set -euo pipefail
mkdir -p logs

module load cuda/12.1 || true
source ~/venvs/hpc-llm/bin/activate

export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=8

# Rendezvous: pick the first allocated node as the master
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=( $nodes )
master_node=${nodes_array[0]}
master_addr=$(srun --nodes=1 --ntasks=1 -w "$master_node" hostname --ip-address | awk '{print $1}')

export MASTER_ADDR=$master_addr
export MASTER_PORT=29500

echo "Master node: $master_node ($master_addr:$MASTER_PORT)"
echo "Total nodes: ${#nodes_array[@]}"

srun --label torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=8 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    scripts/train_deepspeed.py \
        --model gpt2-xl \
        --deepspeed-config configs/zero3.json
