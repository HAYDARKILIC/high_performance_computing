"""
Minimal FSDP training launcher demonstrating the patterns from Week 4.

Launch on a single 8-GPU node with::

    torchrun --standalone --nproc_per_node=8 scripts/train_fsdp.py \\
        --model gpt2 --batch-size 4 --steps 100

The script is intentionally small: a real training run would add a data
loader, checkpointing, and evaluation. The goal here is to exercise the
FSDP wrapping policy, mixed precision, and gradient checkpointing on a
real Hugging Face model.
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
from torch.optim import AdamW

from src.distributed.fsdp_utils import enable_gradient_checkpointing, wrap_with_fsdp
from src.utils.profiling import bench, nvtx_range


def setup_distributed() -> tuple[int, int, int]:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt2")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--gradient-checkpointing", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    if args.gradient_checkpointing:
        enable_gradient_checkpointing(model)
    model = wrap_with_fsdp(model)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    vocab = model.config.vocab_size if hasattr(model.config, "vocab_size") else 50257
    batch = torch.randint(0, vocab, (args.batch_size, args.seq_len), device=device)

    model.train()
    for step in range(args.steps):
        with nvtx_range(f"step_{step}"):
            optimizer.zero_grad(set_to_none=True)
            out = model(batch, labels=batch)
            out.loss.backward()
            optimizer.step()
        if rank == 0 and step % 5 == 0:
            print(f"step {step:4d}  loss = {out.loss.item():.4f}")

    def step_fn() -> None:
        optimizer.zero_grad(set_to_none=True)
        out = model(batch, labels=batch)
        out.loss.backward()
        optimizer.step()

    ms = bench(step_fn, warmup=2, iters=5)
    if rank == 0:
        print(f"\nMean step time: {ms:.2f} ms over {world_size} GPUs")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
