#!/usr/bin/env python3
"""GRPO (Group Relative Policy Optimization) for Qwen3-1.7B.

For each question, sample K rollouts. Reward = 1 if the final answer is
correct, else 0. Advantage = group-normalised reward, so correct rollouts get
positive gradient and incorrect ones get negative gradient, relative to how
hard that particular question is for the current policy.

Contrast with the other two runs in this repo:
  opd_train.py  reverse KL, incorrect rollouts only  -> negative signal only
  rft_train.py  NLL,        correct   rollouts only  -> positive signal only
  grpo_train.py policy grad, all rollouts, K per q   -> both, relative

No teacher needed. Student rollouts generated via vLLM, weights synced every
--sync-steps optimizer steps.

Memory layout (~20GB):
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
SYNC_WEIGHTS_PATH = Path("/tmp/grpo_sync_weights")
LORA_TEMP_PATH = Path("/tmp/grpo_lora_sync")


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

def build_user_prompt(example: dict) -> str:
    """Must match eval_baseline.py's make_*_examples() exactly.

    Previously training passed the bare question while eval added a format
    instruction, so RL optimised a prompt distribution that never occurs at
    eval time.
    """
    q = example["question"]
    bench = example["benchmark"]
    if bench == "gsm8k":
        return (f"Question: {q}\n\n"
                "Show your reasoning, then on the last line write only: Answer: [number]")
    if bench == "strategyqa":
        return (f"Question: {q}\n\n"
                "Think about what facts you need to answer this. Work through the reasoning, "
                "then commit to a final answer. Do not hedge — give your best judgment. "
                "On the last line write only: Answer: Yes or Answer: No")
    if bench == "mmlu":
        # zpd_filter.py stored only the bare stem; the A/B/C/D options were
        # never persisted, so these are unanswerable. Excluded by default via
        # --exclude-benchmarks. Reaching here means that filter was disabled.
        raise ValueError(
            "MMLU training items have no answer choices (see zpd_filter.py). "
            "Keep --exclude-benchmarks mmlu, or rebuild the options first."
        )
    raise ValueError(f"unknown benchmark: {bench}")


def build_prompt(tokenizer: Any, example: dict, no_think: bool) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(example)},
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

def use_sft(args: argparse.Namespace) -> bool:
    return (not args.no_sft) and args.sft_adapter is not None and args.sft_adapter.exists()


def load_base_backbone(args: argparse.Namespace) -> Any:
    """Base model with SFT folded in (if used). The backbone all adapters sit on."""
    model = AutoModelForCausalLM.from_pretrained(
        args.student, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    if use_sft(args):
        model = PeftModel.from_pretrained(model, str(args.sft_adapter)).merge_and_unload()
    return model


def load_student(args: argparse.Namespace, resume_ckpt: Path | None) -> Any:
    # NOTE: the SFT merge must happen on resume too. The old code guarded this
    # with `resume_ckpt is None`, so resumed runs silently trained on bare base.
    base = load_base_backbone(args)

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
            print(f"Loading GRPO adapter from {resume_ckpt} ...")
            set_peft_model_state_dict(model, load_file(str(adapter_path)))

    model.cuda()
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()
    return model


def save_merged_for_vllm(student: Any, args: argparse.Namespace, dest: Path) -> bool:
    """Write base(+SFT)+current adapter to dest, for vLLM to generate from.

    Raises on failure. It previously swallowed the exception and returned False,
    which left a stale snapshot in place — so vLLM kept generating from step-0
    weights for entire runs while training silently continued.
    """
    if shutil.disk_usage(str(dest.parent)).free < 12 * 1024**3:
        raise RuntimeError(
            f"[sync] refusing to write: <12GB free on {dest.parent}. "
            "Staging a merged copy needs ~2x the model size; a failed write here "
            "silently freezes the generation policy."
        )

    student.save_pretrained(str(LORA_TEMP_PATH))

    # Must rebuild the SAME backbone the student trains on. The old code loaded
    # a pristine base here, so SFT was dropped from every generation snapshot.
    temp_base = load_base_backbone(args)
    temp_peft = PeftModel.from_pretrained(temp_base, str(LORA_TEMP_PATH))
    merged = temp_peft.merge_and_unload()

    tmp_dest = dest.parent / (dest.name + "_tmp")
    if tmp_dest.exists():
        shutil.rmtree(str(tmp_dest))
    merged.save_pretrained(str(tmp_dest))
    del temp_base, temp_peft, merged
    torch.cuda.empty_cache()

    if not (tmp_dest / "config.json").exists():
        raise RuntimeError(f"[sync] merged save produced no config.json in {tmp_dest}")

    if dest.exists():
        shutil.rmtree(str(dest))
    shutil.move(str(tmp_dest), str(dest))
    return True


def load_vllm(args: argparse.Namespace, weights_path: Path) -> LLM:
    return LLM(
        model=str(weights_path),
        dtype="bfloat16",
        gpu_memory_utilization=args.vllm_gpu_mem,
        trust_remote_code=True,
        enforce_eager=True,
        swap_space=0,
    )


def sync_vllm(
    student: Any,
    vllm_model: LLM | None,
    args: argparse.Namespace,
    step: int,
) -> LLM | None:
    print(f"\n[sync] Step {step}: syncing student → vLLM ...")
    t0 = time.perf_counter()

    if vllm_model is not None:
        del vllm_model
        torch.cuda.empty_cache()

    save_merged_for_vllm(student, args, SYNC_WEIGHTS_PATH)   # raises on failure

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
                swap_space=0,
            )
        except Exception as e:
            print(f"[WARN] vLLM reload failed: {e}. Generation disabled until next sync.")
            torch.cuda.empty_cache()
            return None


# ── GRPO loss ──────────────────────────────────────────────────────────────────

def group_advantages(rewards: list[float]) -> list[float] | None:
    """Group-normalised advantages. None if the group carries no signal."""
    n = len(rewards)
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n
    std = math.sqrt(var)
    if std < 1e-6:          # all correct or all wrong -> nothing to learn from
        return None
    return [(r - mean) / (std + 1e-4) for r in rewards]


def policy_gradient_loss(
    student: Any,
    full_ids: torch.Tensor,
    prompt_len: int,
    advantage: float,
) -> torch.Tensor:
    """-A * mean log pi(generated tokens).

    One optimizer step per generation batch means the policy is unchanged while
    the batch is consumed, so the PPO importance ratio is exactly 1 and the
    clipped surrogate reduces to this. Length-normalised so long rollouts do not
    dominate.
    """
    device = next(student.parameters()).device
    input_ids = full_ids.unsqueeze(0).to(device)

    seq_len = full_ids.shape[0]
    gen_start = prompt_len - 1
    gen_end = seq_len - 1
    if gen_start >= gen_end:
        return torch.zeros((), device=device, requires_grad=True)

    logits = student(input_ids).logits[0, gen_start:gen_end]
    targets = full_ids[prompt_len:].to(device)

    log_probs = F.log_softmax(logits, dim=-1)
    token_logp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return -advantage * token_logp.mean()


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--sft-adapter", type=Path, default=Path("checkpoints/sft_coldstart/final"))
    parser.add_argument("--no-sft", action="store_true",
                        help="Train directly on base, ignoring --sft-adapter (experiment A).")
    parser.add_argument("--exclude-benchmarks", nargs="*", default=["mmlu"],
                        help="Benchmarks dropped from TRAINING. mmlu is excluded by default "
                             "because zpd_filter.py did not persist the A/B/C/D options. "
                             "Pass an empty list to disable.")
    parser.add_argument("--train-data", type=Path, default=Path("data/zpd_filtered/all_zpd.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/grpo"))
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Optimizer steps. Each consumes questions-per-step * num-rollouts rollouts.")
    parser.add_argument("--questions-per-step", type=int, default=8)
    parser.add_argument("--num-rollouts", type=int, default=8,
                        help="K rollouts per question. Must be >1 for group advantages to exist.")
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--sync-steps", type=int, default=20,
                        help="Lower = closer to on-policy, but each sync reloads vLLM (~30-60s).")
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--warmup-steps", type=int, default=25)
    parser.add_argument("--vllm-gpu-mem", type=float, default=0.45)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Higher than eval on purpose: exploration is what surfaces rare successes.")
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
    n_all = len(questions)
    if args.exclude_benchmarks:
        questions = [q for q in questions if q["benchmark"] not in args.exclude_benchmarks]
        print(f"Excluded {n_all - len(questions)} items from {args.exclude_benchmarks}")
    if not questions:
        raise SystemExit("No training questions left after --exclude-benchmarks.")
    print(f"Loaded {len(questions)} ZPD questions (of {n_all})")
    print(f"Backbone: base{'+SFT' if use_sft(args) else ' only (--no-sft)'}")
    print(f"Each step: {args.questions_per_step} questions x {args.num_rollouts} rollouts "
          f"= {args.questions_per_step * args.num_rollouts} generations")

    ckpt_dirs = sorted(args.output_dir.glob("step_*"),
                       key=lambda p: int(p.name.split("_")[1]))
    resume_ckpt = ckpt_dirs[-1] if ckpt_dirs else None
    if resume_ckpt:
        print(f"Resuming from checkpoint: {resume_ckpt}")

    print(f"\nLoading tokenizer: {args.student}")
    tokenizer = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    eos_ids = get_eos_ids(tokenizer)

    print(f"Loading student: {args.student}")
    student = load_student(args, resume_ckpt)

    # Always write fresh. The old `if not config.json exists` guard meant a
    # restarted run silently adopted the previous run's snapshot.
    print("\nPreparing initial vLLM weights ...")
    SYNC_WEIGHTS_PATH.mkdir(parents=True, exist_ok=True)
    save_merged_for_vllm(student, args, SYNC_WEIGHTS_PATH)
    tokenizer.save_pretrained(str(SYNC_WEIGHTS_PATH))

    print(f"\nLoading vLLM (gpu_memory_utilization={args.vllm_gpu_mem}) ...")
    vllm_model = load_vllm(args, SYNC_WEIGHTS_PATH)
    print("vLLM ready. No teacher loaded.")

    # n=1 with duplicated prompts rather than n=K: vLLM V0's multi-sample
    # sampler hits an illegal memory access on this setup.
    sampling_params = SamplingParams(
        n=1,
        temperature=args.temperature,
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
    if resume_ckpt:
        state_file = resume_ckpt / "trainer_state.pt"
        if state_file.exists():
            state = torch.load(state_file, map_location="cpu")
            opt_step = state["opt_step"]
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            torch.set_rng_state(state["rng"])
            print(f"Resumed at step {opt_step}")

    log_path = args.output_dir / "grpo_log.jsonl"
    window = {"reward_sum": 0.0, "n_rollouts": 0, "groups": 0,
              "dead_groups": 0, "loss_sum": 0.0, "n_backward": 0}
    pbar = tqdm(total=args.max_steps, initial=opt_step, desc="GRPO")

    while opt_step < args.max_steps:
        batch_qs = [random.choice(questions) for _ in range(args.questions_per_step)]
        prompts: list[str] = []
        for q in batch_qs:
            prompts.extend([build_prompt(tokenizer, q, args.no_think)] * args.num_rollouts)

        if vllm_model is None:
            vllm_model = sync_vllm(student, None, args, opt_step)
            if vllm_model is None:
                continue

        t_gen = time.perf_counter()
        outputs = vllm_model.generate(prompts, sampling_params)
        gen_time = time.perf_counter() - t_gen
        total_gen_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)

        optimizer.zero_grad()
        n_backward = 0

        for gi, q in enumerate(batch_qs):
            group = outputs[gi * args.num_rollouts : (gi + 1) * args.num_rollouts]
            if not group:
                continue
            prompt_ids = list(group[0].prompt_token_ids)
            completions = [o.outputs[0] for o in group if len(o.outputs[0].token_ids) > 0]
            if len(completions) < 2:
                continue

            rewards = [
                1.0 if check_correct(tokenizer.decode(c.token_ids, skip_special_tokens=True), q) else 0.0
                for c in completions
            ]
            window["reward_sum"] += sum(rewards)
            window["n_rollouts"] += len(rewards)
            window["groups"] += 1

            advantages = group_advantages(rewards)
            if advantages is None:
                window["dead_groups"] += 1
                continue

            # Normalise so a step's gradient magnitude does not depend on how
            # many groups happened to carry signal.
            denom = args.questions_per_step * len(completions)

            for c, adv in zip(completions, advantages):
                full_ids = torch.tensor(prompt_ids + list(c.token_ids))
                try:
                    loss = policy_gradient_loss(student, full_ids, len(prompt_ids), adv)
                    (loss / denom).backward()
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print("[OOM] rollout skipped, cache cleared.")
                    continue
                window["loss_sum"] += loss.item()
                n_backward += 1

        if n_backward == 0:
            # Every group was all-correct or all-wrong; nothing to update on.
            continue

        window["n_backward"] += n_backward
        torch.nn.utils.clip_grad_norm_(
            [p for p in student.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        opt_step += 1
        pbar.update(1)

        if opt_step % args.logging_steps == 0:
            mean_reward = window["reward_sum"] / max(1, window["n_rollouts"])
            dead_rate = window["dead_groups"] / max(1, window["groups"])
            log_row = {
                "step": opt_step,
                "mean_reward": round(mean_reward, 4),
                "dead_group_rate": round(dead_rate, 3),
                "loss": round(window["loss_sum"] / max(1, window["n_backward"]), 4),
                "lr": optimizer.param_groups[0]["lr"],
                "gen_tok_per_sec": round(total_gen_tokens / max(gen_time, 1e-6), 1),
            }
            append_jsonl(log_path, log_row)
            pbar.set_postfix(reward=f"{mean_reward:.3f}", dead=f"{dead_rate:.1%}")
            window = {"reward_sum": 0.0, "n_rollouts": 0, "groups": 0,
                      "dead_groups": 0, "loss_sum": 0.0, "n_backward": 0}

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

    pbar.close()
    final_path = args.output_dir / "final"
    student.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"\nGRPO complete. Final adapter saved to {final_path}")
    print(f"Copy the merged snapshot out of /tmp now:\n"
          f"  cp -r {SYNC_WEIGHTS_PATH} checkpoints/grpo_merged")


if __name__ == "__main__":
    main()
