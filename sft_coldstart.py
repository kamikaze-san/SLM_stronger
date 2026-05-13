#!/usr/bin/env python3
"""SFT cold start for Qwen3-1.7B using LoRA.

Trains the student model on teacher traces to initialise it near the
teacher distribution before OPD. Loss is computed only on the assistant
response (<think>...</think><answer>...</answer>), not on the prompt.

~500 steps, LoRA rank=32 alpha=64, BF16, AdamW lr=1e-5, batch=8.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


IGNORE_INDEX = -100


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def format_conversation(example: dict, tokenizer: Any) -> str:
    """Format as Qwen3 chat template with system + user + assistant."""
    messages = [
        {"role": "system", "content": example["instruction"]},
        {"role": "user", "content": example["input"]},
        {"role": "assistant", "content": example["output"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def tokenize_with_loss_mask(
    example: dict,
    tokenizer: Any,
    max_length: int,
) -> dict:
    """Tokenize and set labels=-100 for prompt tokens (no loss on input)."""
    messages_prompt = [
        {"role": "system", "content": example["instruction"]},
        {"role": "user", "content": example["input"]},
    ]
    messages_full = messages_prompt + [
        {"role": "assistant", "content": example["output"]}
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages_prompt, tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        messages_full, tokenize=False
    )

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(
        full_text,
        add_special_tokens=False,
        max_length=max_length,
        truncation=True,
    )["input_ids"]

    labels = [IGNORE_INDEX] * len(prompt_ids) + full_ids[len(prompt_ids):]
    # Truncate labels to match input_ids length
    labels = labels[: len(full_ids)]

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def load_model_and_tokenizer(model_name: str) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    return model, tokenizer


def apply_lora(model: Any, rank: int, alpha: int) -> Any:
    # Target the attention and MLP projection layers
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"]
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--train-data", type=Path, default=Path("data/sft_dataset_nothink/train.jsonl"))
    parser.add_argument("--val-data", type=Path, default=Path("data/sft_dataset_nothink/val.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/sft_coldstart"))
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Per-device batch size. Effective batch = batch_size * grad_accum.")
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model_name}")
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    model = apply_lora(model, args.lora_rank, args.lora_alpha)

    print("Tokenizing datasets...")
    train_raw = load_jsonl(args.train_data)
    val_raw = load_jsonl(args.val_data)

    # Normalize ground_truth to string to avoid pyarrow mixed-type errors
    for row in train_raw + val_raw:
        row["ground_truth"] = str(row.get("ground_truth", ""))

    def tokenize(example: dict) -> dict:
        return tokenize_with_loss_mask(example, tokenizer, args.max_length)

    train_dataset = Dataset.from_list(train_raw).map(
        tokenize, remove_columns=list(train_raw[0].keys()), num_proc=4
    )
    val_dataset = Dataset.from_list(val_raw).map(
        tokenize, remove_columns=list(val_raw[0].keys()), num_proc=4
    )

    print(f"Train examples: {len(train_dataset)}")
    print(f"Val examples:   {len(val_dataset)}")

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX,
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        bf16=True,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    print("Starting SFT cold start training...")
    trainer.train()

    print(f"Saving final model to {args.output_dir}/final")
    model.save_pretrained(str(args.output_dir / "final"))
    tokenizer.save_pretrained(str(args.output_dir / "final"))
    print("Done.")


if __name__ == "__main__":
    main()
