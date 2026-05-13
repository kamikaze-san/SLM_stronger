#!/usr/bin/env python3
"""On-Policy Distillation (OPD) for Qwen3-1.7B.

Per-token reverse KL loss applied only on incorrect student rollouts.
Teacher (Qwen3-14B) is frozen throughout.

Loop per step:
  1. Sample ZPD question
  2. Student generates rollout (no grad)
  3. Check correctness — skip if correct
  4. Teacher forward pass on rollout (frozen, no grad)
  5. Student forward pass with gradients
  6. Reverse KL on generated tokens only
  7. Accumulate gradients, optimizer step every --grad-accum rollouts
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = "Think through this step by step."


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ── Answer extraction ──────────────────────────────────────────────────────────

def strip_think_block(text: str) -> str:
    m = re.search(r"</think>(.*)", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else text


def _answer_tag(text: str) -> str | None:
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None


def _answer_line(text: str) -> str | None:
    m = re.search(r"(?:^|\n)\s*(?:\*\*)?Answer:?\**\s*(.+)", text, re.IGNORECASE)
    return m.group(1).strip() if m else None


def extract_gsm8k(response: str) -> str | None:
    for src in [_answer_tag(response), _answer_line(strip_think_block(response)), strip_think_block(response)]:
        if src is None:
            continue
        nums = re.findall(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?",
                          src.replace("$", " ").replace("\\$", " "))
        if nums:
            return nums[-1].replace(",", "")
    return None


def extract_mmlu(response: str) -> str | None:
    for src in [_answer_tag(response), _answer_line(strip_think_block(response)), strip_think_block(response)]:
        if src is None:
            continue
        m = re.search(r"\b([ABCD])\b", src, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return None


def extract_strategyqa(response: str) -> str | None:
    for src in [_answer_tag(response), _answer_line(strip_think_block(response)), strip_think_block(response)]:
        if src is None:
            continue
        m = re.search(r"\b(yes|no)\b", src, re.IGNORECASE)
        if m:
            return m.group(1).lower()
    return None


def check_correct(response: str, example: dict) -> bool:
    bench = example["benchmark"]
    gt = str(example["ground_truth"])
    if bench == "gsm8k":
        pred = extract_gsm8k(response)
        if pred is None:
            return False
        try:
            return math.isclose(float(pred), float(gt.replace(",", "")), abs_tol=1e-6)
        except ValueError:
            return False
    elif bench == "mmlu":
        pred = extract_mmlu(response)
        return pred is not None and pred.lower() == gt.lower().strip()
    elif bench == "strategyqa":
        pred = extract_strategyqa(response)
        return pred is not None and pred.lower() == gt.lower().strip()
    return False


# ── Model loading ──────────────────────────────────────────────────────────────

def get_eos_ids(tokenizer: Any) -> list[int]:
    ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id else []
    for token in ("<|end|>", "<|endoftext|>", "<|im_end|>"):
        tid = tokenizer.convert_tokens_to_ids(token)
        if tid is not None and tid != tokenizer.unk_token_id and tid not in ids:
            ids.append(tid)
    return ids


def load_student(model_name: str, sft_adapter: Path | None, rank: int, alpha: int) -> Any:
    base = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    if sft_adapter and sft_adapter.exists():
        print(f"Merging SFT LoRA from {sft_adapter} into base (in memory)...")
        base = PeftModel.from_pretrained(base, str(sft_adapter))
        base = base.merge_and_unload()

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(base, config)
    model.cuda()
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()
    return model


def load_teacher(model_name: str) -> Any:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ── Prompt builder ─────────────────────────────────────────────────────────────

def build_prompt(tokenizer: Any, question: str, no_think: bool) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if no_think:
        try:
            return tokenizer.apply_chat_template(messages, **kwargs, enable_thinking=False)
        except TypeError:
            pass
    return tokenizer.apply_chat_template(messages, **kwargs)


# ── Core OPD computation ───────────────────────────────────────────────────────

def generate_rollouts_batch(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    max_new_tokens: int,
    eos_ids: list[int],
) -> list[tuple[torch.Tensor, int]]:
    """Generate a batch of rollouts. Returns list of (full_ids_cpu_1d, prompt_len)."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=eos_ids,
        )
    return [(outputs[i].cpu(), prompt_len) for i in range(len(prompts))]


def reverse_kl_loss(
    student: Any,
    teacher: Any,
    full_ids: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    """Reverse KL(student || teacher) averaged over generated tokens."""
    student_device = next(student.parameters()).device
    teacher_device = next(teacher.parameters()).device

    input_ids = full_ids.unsqueeze(0)
    seq_len = full_ids.shape[0]

    # logits[i] predicts token[i+1], generated tokens start at prompt_len
    gen_start = prompt_len - 1
    gen_end = seq_len - 1

    if gen_start >= gen_end:
        return torch.tensor(0.0, device=student_device, requires_grad=True)

    with torch.no_grad():
        t_logits = teacher(input_ids.to(teacher_device)).logits[0, gen_start:gen_end].to(student_device)

    s_logits = student(input_ids.to(student_device)).logits[0, gen_start:gen_end]

    s_log_p = F.log_softmax(s_logits, dim=-1)
    t_log_p = F.log_softmax(t_logits, dim=-1)

    # KL(student || teacher) = sum_x student(x) * [log student(x) - log teacher(x)]
    kl = (s_log_p.exp() * (s_log_p - t_log_p)).sum(-1).mean()
    return kl


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--sft-adapter", type=Path, default=Path("checkpoints/sft_coldstart/final"),
                        help="SFT LoRA adapter merged into student in memory before OPD.")
    parser.add_argument("--teacher", default="Qwen/Qwen3-14B")
    parser.add_argument("--train-data", type=Path, default=Path("data/zpd_filtered/all_zpd.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/opd"))
    parser.add_argument("--max-steps", type=int, default=2000,
                        help="Number of optimizer steps (not total iterations).")
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="Gradient accumulation over incorrect rollouts before optimizer step.")
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--gen-batch-size", type=int, default=4,
                        help="Rollouts to generate in one forward pass. Reduce if OOM during generation.")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-think", action="store_true", default=True)
    return parser.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    questions = load_jsonl(args.train_data)
    print(f"Loaded {len(questions)} ZPD questions")

    print(f"\nLoading student: {args.student}")
    tokenizer = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    student = load_student(args.student, args.sft_adapter, args.lora_rank, args.lora_alpha)
    eos_ids = get_eos_ids(tokenizer)

    print(f"\nLoading teacher: {args.teacher}")
    teacher = load_teacher(args.teacher)
    print("Teacher frozen.")

    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    log_path = args.output_dir / "opd_log.jsonl"
    window = {"skipped": 0, "trained": 0, "loss_sum": 0.0}

    optimizer.zero_grad()
    accum_count = 0
    opt_step = 0

    # Resume from latest checkpoint if available
    ckpt_dirs = sorted(args.output_dir.glob("step_*"),
                       key=lambda p: int(p.name.split("_")[1]))
    if ckpt_dirs:
        latest = ckpt_dirs[-1]
        state_file = latest / "trainer_state.pt"
        if state_file.exists():
            print(f"Resuming from {latest} ...")
            state = torch.load(state_file, map_location="cpu")
            opt_step = state["opt_step"]
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            torch.set_rng_state(state["rng"])
            # Reload LoRA weights from checkpoint
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file
            adapter_path = latest / "adapter_model.safetensors"
            if adapter_path.exists():
                adapter_weights = load_file(str(adapter_path))
                set_peft_model_state_dict(student, adapter_weights)
            print(f"Resumed at step {opt_step}")

    pbar = tqdm(total=args.max_steps, initial=opt_step, desc="OPD")

    _diag_done = False
    while opt_step < args.max_steps:
        # Sample a batch of questions and generate all rollouts at once
        batch_qs = [random.choice(questions) for _ in range(args.gen_batch_size)]
        prompts = [build_prompt(tokenizer, q["question"], args.no_think) for q in batch_qs]

        _t0 = time.perf_counter()
        try:
            rollouts = generate_rollouts_batch(
                student, tokenizer, prompts, args.max_new_tokens, eos_ids
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            # Fallback: generate one at a time
            rollouts = []
            for prompt in prompts:
                try:
                    r = generate_rollouts_batch(student, tokenizer, [prompt], args.max_new_tokens, eos_ids)
                    rollouts.extend(r)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
        _gen_t = time.perf_counter() - _t0

        _fwd_bwd_t = 0.0
        for q, (full_ids, prompt_len) in zip(batch_qs, rollouts):
            if full_ids.shape[0] <= prompt_len:
                continue

            response = tokenizer.decode(full_ids[prompt_len:], skip_special_tokens=True)

            if check_correct(response, q):
                window["skipped"] += 1
                continue

            _t0 = time.perf_counter()
            loss = reverse_kl_loss(student, teacher, full_ids, prompt_len)
            (loss / args.grad_accum).backward()
            _fwd_bwd_t += time.perf_counter() - _t0

            accum_count += 1
            window["trained"] += 1
            window["loss_sum"] += loss.item()

            if accum_count >= args.grad_accum:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in student.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                accum_count = 0
                opt_step += 1
                pbar.update(1)

                if opt_step % args.logging_steps == 0:
                    avg_loss = window["loss_sum"] / max(1, window["trained"])
                    total_seen = window["skipped"] + window["trained"]
                    skip_rate = window["skipped"] / max(1, total_seen)
                    log_row = {
                        "step": opt_step,
                        "loss": round(avg_loss, 4),
                        "skip_rate": round(skip_rate, 3),
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                    append_jsonl(log_path, log_row)
                    pbar.set_postfix(loss=f"{avg_loss:.4f}", skip=f"{skip_rate:.1%}")
                    window = {"skipped": 0, "trained": 0, "loss_sum": 0.0}

                if opt_step % args.save_steps == 0:
                    ckpt = args.output_dir / f"step_{opt_step}"
                    student.save_pretrained(str(ckpt))
                    tokenizer.save_pretrained(str(ckpt))
                    torch.save({
                        "opt_step": opt_step,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "rng": torch.get_rng_state(),
                    }, ckpt / "trainer_state.pt")
                    print(f"\nSaved checkpoint: {ckpt}")

                if opt_step >= args.max_steps:
                    break

        if not _diag_done:
            total_tokens = sum(f.shape[0] - p for f, p in rollouts)
            print(f"[diag] batch={args.gen_batch_size}  tokens={total_tokens}  gen={_gen_t:.1f}s  fwd+bwd={_fwd_bwd_t:.1f}s  gen_tok/s={total_tokens/_gen_t:.0f}")
            _diag_done = True

    pbar.close()
    final_path = args.output_dir / "final"
    student.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"\nOPD complete. Final adapter saved to {final_path}")


if __name__ == "__main__":
    main()
