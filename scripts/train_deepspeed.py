"""
DeepSpeed ZeRO-3 training launcher to accompany Week 4.

Run with::

    deepspeed --num_gpus=8 scripts/train_deepspeed.py \\
        --model gpt2 --deepspeed-config configs/zero3.json

The full ZeRO-3 configuration lives in ``configs/zero3.json``; this script
just plumbs it into a Hugging Face model.
"""

from __future__ import annotations

import argparse

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt2")
    p.add_argument("--deepspeed-config", default="configs/zero3.json")
    p.add_argument("--local_rank", type=int, default=0)  # injected by deepspeed launcher
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--steps", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import deepspeed
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    engine, _, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        config=args.deepspeed_config,
    )

    device = torch.device(f"cuda:{args.local_rank}")
    vocab = model.config.vocab_size if hasattr(model.config, "vocab_size") else 50257
    batch = torch.randint(0, vocab, (args.batch_size, args.seq_len), device=device)

    engine.train()
    for step in range(args.steps):
        out = engine(batch, labels=batch)
        engine.backward(out.loss)
        engine.step()
        if engine.global_rank == 0 and step % 5 == 0:
            print(f"step {step:4d}  loss = {out.loss.item():.4f}")


if __name__ == "__main__":
    main()
