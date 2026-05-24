#!/bin/bash
#SBATCH --job-name=vllm-serve
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# vLLM serving job (Week 3).
# Usage: sbatch scripts/slurm/inference_vllm.sh

set -euo pipefail
mkdir -p logs

module load cuda/12.1 || true
source ~/venvs/hpc-llm/bin/activate

MODEL=${MODEL:-meta-llama/Llama-2-7b-hf}
TP=${TP:-2}
PORT=${PORT:-8000}

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --tensor-parallel-size "$TP" \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --host 0.0.0.0 \
    --port "$PORT"
