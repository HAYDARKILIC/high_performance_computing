"""
QLoRA fine-tuning script accompanying Week 6.

Trains a 4-bit NF4-quantized base model with LoRA adapters. Memory
footprint is small enough to fit a 7B model on a single 24 GB GPU.

Example::

    python scripts/train_qlora.py --model meta-llama/Llama-2-7b-hf \\
        --dataset tatsu-lab/alpaca --max-steps 200
"""

from __future__ import annotations

import argparse

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-2-7b-hf")
    p.add_argument("--dataset", default="tatsu-lab/alpaca")
    p.add_argument("--output-dir", default="./outputs/qlora")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    ds = load_dataset(args.dataset, split="train")

    def tokenize(example):
        text = example.get("text") or example.get("instruction", "")
        return tokenizer(text, truncation=True, max_length=1024)

    ds = ds.map(tokenize, remove_columns=ds.column_names)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        bf16=True,
        optim="paged_adamw_32bit",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="tensorboard",
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=ds, tokenizer=tokenizer)
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
