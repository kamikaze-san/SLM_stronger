#!/usr/bin/env python3
"""On-Policy Distillation (OPD) for Qwen3-1.7B with vLLM generation.

Per-token reverse KL loss applied only on incorrect student rollouts.
Teacher (Qwen3-14B) is frozen in PyTorch throughout.
Student rollouts generated via vLLM (fast). Student weights synced to vLLM
every --sync-steps optimizer steps.

Memory layout (~48GB):
  Teacher  (PyTorch, frozen BF16): ~28GB
  Student  (PyTorch, LoRA train):   ~4GB
  Optimizer states:                 ~6GB
  vLLM     (student generation):    ~8GB
  Scratch:                          ~2GB
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from peft.utils import set_peft_model_state_dict
from safetensors.torch import load_file
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams


SYSTEM_PROMPT = "Think through this step by step."
SYNC_WEIGHTS_PATH = Path("/tmp/opd_sync_weights")
LORA_TEMP_PATH = Path("/tmp/opd_lora_sync")


# ── Data ───────────────────────────────────────────────────────────────────────

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


# ── Prompt ─────────────────────────────────────────────────────────────────────

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


def get_eos_ids(tokenizer: Any) -> list[int]:
    ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id else []
    for token in ("<|end|>", "<|endoftext|>", "<|im_end|>"):
        tid = tokenizer.convert_tokens_to_ids(token)
        if tid is not None and tid != tokenizer.unk_token_id and tid not in ids:
            ids.append(tid)
    return ids


# ── Model loading ──────────────────────────────────────────────────────────────

def load_student(args: argparse.Namespace, resume_ckpt: Path | None) -> Any:
    base = AutoModelForCausalLM.from_pretrained(
        args.student, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    if resume_ckpt is None and args.sft_adapter and args.sft_adapter.exists():
        print(f"Merging SFT LoRA from {args.sft_adapter} ...")
        base = PeftModel.from_pretrained(base, str(args.sft_adapter)).merge_and_unload()

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(base, config)

    if resume_ckpt is not None:
        adapter_path = resume_ckpt / "adapter_model.safetensors"
        if adapter_path.exists():
            print(f"Loading OPD adapter from {resume_ckpt} ...")
            set_peft_model_state_dict(model, load_file(str(adapter_path)))

    model.cuda()
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()
    return model


def load_teacher(model_name: str) -> Any:
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def save_merged_for_vllm(student: Any, args: argparse.Namespace, dest: Path) -> bool:
    """Merge current student LoRA into base and save to dest. Returns True on success."""
    try:
        student.save_pretrained(str(LORA_TEMP_PATH))

        temp_base = AutoModelForCausalLM.from_pretrained(
            args.student, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        temp_peft = PeftModel.from_pretrained(temp_base, str(LORA_TEMP_PATH))
        merged = temp_peft.merge_and_unload()

        tmp_dest = dest.parent / (dest.name + "_tmp")
        merged.save_pretrained(str(tmp_dest))
        del temp_base, temp_peft, merged
        torch.cuda.empty_cache()

        if dest.exists():
            shutil.rmtree(str(dest))
        shutil.move(str(tmp_dest), str(dest))
        return True
    except Exception as e:
        print(f"[sync] save_merged_for_vllm failed: {e}")
        torch.cuda.empty_cache()
        return False


def load_vllm(args: argparse.Namespace, weights_path: Path) -> LLM:
    return LLM(
        model=str(weights_path),
        dtype="bfloat16",
        gpu_memory_utilization=args.vllm_gpu_mem,
        trust_remote_code=True,
        enforce_eager=True,
    )


def sync_vllm(
    student: Any,
    vllm_model: LLM | None,
    args: argparse.Namespace,
    step: int,
) -> LLM | None:
    """Sync student weights into vLLM. Returns updated vllm_model (or old if OOM)."""
    print(f"\n[sync] Step {step}: syncing student → vLLM ...")
    t0 = time.perf_counter()

    # Delete vLLM first to free memory before creating merged copy
    if vllm_model is not None:
        del vllm_model
        torch.cuda.empty_cache()

    saved = save_merged_for_vllm(student, args, SYNC_WEIGHTS_PATH)
    if not saved:
        print(f"[sync] Merge failed, skipping reload.")
        # Try to reload from last known good weights
        if SYNC_WEIGHTS_PATH.exists():
            try:
                return load_vllm(args, SYNC_WEIGHTS_PATH)
            except Exception:
                torch.cuda.empty_cache()
                return None
        return None

    try:
        new_vllm = load_vllm(args, SYNC_WEIGHTS_PATH)
        print(f"[sync] Done in {time.perf_counter()-t0:.1f}s")
        return new_vllm
    except torch.cuda.OutOfMemoryError:
        print(f"[WARN] Step {step}: vLLM reload OOM. Trying lower gpu_memory_utilization ...")
        torch.cuda.empty_cache()
        try:
            reduced = max(0.15, args.vllm_gpu_mem - 0.1)
            return LLM(
                model=str(SYNC_WEIGHTS_PATH),
                dtype="bfloat16",
                gpu_memory_utilization=reduced,
                trust_remote_code=True,
                enforce_eager=True,
            )
        except Exception as e:
            print(f"[WARN] vLLM reload failed: {e}. Generation disabled until next sync.")
            torch.cuda.empty_cache()
            return None


# ── KL loss ────────────────────────────────────────────────────────────────────

def reverse_kl_loss(
    student: Any,
    teacher: Any,
    full_ids: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    """Reverse KL(student || teacher) averaged over generated tokens only."""
    student_device = next(student.parameters()).device
    teacher_device = next(teacher.parameters()).device

    input_ids = full_ids.unsqueeze(0)
    seq_len = full_ids.shape[0]
    gen_start = prompt_len - 1
    gen_end = seq_len - 1

    if gen_start >= gen_end:
        return torch.tensor(0.0, device=student_device, requires_grad=True)

    with torch.no_grad():
        t_logits = teacher(input_ids.to(teacher_device)).logits[0, gen_start:gen_end].to(student_device)

    s_logits = student(input_ids.to(student_device)).logits[0, gen_start:gen_end]

    s_log_p = F.log_softmax(s_logits, dim=-1)
    t_log_p = F.log_softmax(t_logits, dim=-1)
    kl = (s_log_p.exp() * (s_log_p - t_log_p)).sum(-1).mean()
    return kl


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--sft-adapter", type=Path, default=Path("checkpoints/sft_coldstart/final"))
    parser.add_argument("--teacher", default="Qwen/Qwen3-14B")
    parser.add_argument("--train-data", type=Path, default=Path("data/zpd_filtered/all_zpd.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/opd"))
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--gen-batch-size", type=int, default=8,
                        help="Rollouts per vLLM call. vLLM handles batching efficiently.")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--sync-steps", type=int, default=100,
                        help="Sync student weights to vLLM every N optimizer steps.")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--vllm-gpu-mem", type=float, default=0.3,
                        help="gpu_memory_utilization for vLLM. Reduce if OOM.")
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

    # Find latest OPD checkpoint for resuming
    ckpt_dirs = sorted(args.output_dir.glob("step_*"),
                       key=lambda p: int(p.name.split("_")[1]))
    resume_ckpt = ckpt_dirs[-1] if ckpt_dirs else None
    if resume_ckpt:
        print(f"Found checkpoint: {resume_ckpt}")

    print(f"\nLoading tokenizer: {args.student}")
    tokenizer = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    eos_ids = get_eos_ids(tokenizer)

    print(f"Loading student: {args.student}")
    student = load_student(args, resume_ckpt)

    # Save initial merged weights for vLLM before loading teacher
    # (teacher will consume ~28GB; doing this first keeps peak lower)
    print("\nPreparing initial vLLM weights ...")
    SYNC_WEIGHTS_PATH.mkdir(parents=True, exist_ok=True)
    if not (SYNC_WEIGHTS_PATH / "config.json").exists():
        save_merged_for_vllm(student, args, SYNC_WEIGHTS_PATH)
    # Also save tokenizer for vLLM
    tokenizer.save_pretrained(str(SYNC_WEIGHTS_PATH))

    print(f"\nLoading teacher: {args.teacher}")
    teacher = load_teacher(args.teacher)
    print("Teacher frozen.")

    print(f"\nLoading vLLM (gpu_memory_utilization={args.vllm_gpu_mem}) ...")
    vllm_model = load_vllm(args, SYNC_WEIGHTS_PATH)
    print("vLLM ready.")

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=args.max_new_tokens,
        stop_token_ids=eos_ids,
        skip_special_tokens=False,
    )

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

    opt_step = 0
    accum_count = 0
    if resume_ckpt:
        state_file = resume_ckpt / "trainer_state.pt"
        if state_file.exists():
            state = torch.load(state_file, map_location="cpu")
            opt_step = state["opt_step"]
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            torch.set_rng_state(state["rng"])
            print(f"Resumed at step {opt_step}")

    log_path = args.output_dir / "opd_log.jsonl"
    window = {"skipped": 0, "trained": 0, "loss_sum": 0.0}
    optimizer.zero_grad()
    pbar = tqdm(total=args.max_steps, initial=opt_step, desc="OPD")

    while opt_step < args.max_steps:
        # Sample batch of questions
        batch_qs = [random.choice(questions) for _ in range(args.gen_batch_size)]
        prompts = [build_prompt(tokenizer, q["question"], args.no_think) for q in batch_qs]

        # Generate rollouts via vLLM (fast)
        if vllm_model is None:
            # vLLM unavailable (OOM during last sync), force a sync attempt
            vllm_model = sync_vllm(student, None, args, opt_step)
            if vllm_model is None:
                continue

        t_gen = time.perf_counter()
        outputs = vllm_model.generate(prompts, sampling_params)
        gen_time = time.perf_counter() - t_gen
        total_gen_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)

        # Process each rollout
        for q, output in zip(batch_qs, outputs):
            prompt_ids = list(output.prompt_token_ids)
            gen_ids = list(output.outputs[0].token_ids)

            if not gen_ids:
                continue

            response = tokenizer.decode(gen_ids, skip_special_tokens=True)

            if check_correct(response, q):
                window["skipped"] += 1
                continue

            full_ids = torch.tensor(prompt_ids + gen_ids)
            prompt_len = len(prompt_ids)

            try:
                loss = reverse_kl_loss(student, teacher, full_ids, prompt_len)
                (loss / args.grad_accum).backward()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                accum_count = 0
                print("[OOM] KL step skipped, cache cleared.")
                continue

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
                    tok_per_sec = total_gen_tokens / max(gen_time, 1e-6)
                    log_row = {
                        "step": opt_step,
                        "loss": round(avg_loss, 4),
                        "skip_rate": round(skip_rate, 3),
                        "lr": optimizer.param_groups[0]["lr"],
                        "gen_tok_per_sec": round(tok_per_sec, 1),
                    }
                    append_jsonl(log_path, log_row)
                    pbar.set_postfix(loss=f"{avg_loss:.4f}", skip=f"{skip_rate:.1%}",
                                     tok_s=f"{tok_per_sec:.0f}")
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

                if opt_step % args.sync_steps == 0:
                    vllm_model = sync_vllm(student, vllm_model, args, opt_step)

                if opt_step >= args.max_steps:
                    break

    pbar.close()
    final_path = args.output_dir / "final"
    student.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"\nOPD complete. Final adapter saved to {final_path}")


if __name__ == "__main__":
    main()
