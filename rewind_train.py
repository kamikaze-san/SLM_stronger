#!/usr/bin/env python3
"""Rewind-GRPO: rescue dead groups by restarting from a partial solution.

Vanilla GRPO learns nothing from a question where all K rollouts fail — the
rewards are uniform, the advantages are all zero, and the group is discarded.
That was ~60% of groups in our runs.

For those groups this takes the longest failed attempt, cuts it at
--rewind-frac of its length, and regenerates K continuations from that partial
work. The early part of a failed attempt is usually still correct, so finishing
from there is easier than starting blank: measured on 100 all-wrong GSM8K
questions, a fixed 40% cut yields at least one correct finish 27% of the time.
Those groups then train exactly like any other GRPO group.

Note the probe and the training rollouts are the SAME generations — if any
continuation is correct we already have the group, so there is no separate
measurement cost. Extra cost is K generations per dead group attempted.

Only GSM8K groups are rewound. StrategyQA rollouts are short, have no
multi-step chain to cut into, and its ZPD labels are 74% "yes" (see
project memory) so its groups are not worth rescuing.

Everything else — data, loss, sync, logging — is imported unchanged from
grpo_train so this is a single-variable comparison against that baseline.

Usage (matches the GRPO baseline exactly apart from rewind):
  VLLM_USE_V1=0 python rewind_train.py --no-sft --lr 5e-6 \
      --max-steps 1000 --save-steps 200 --kl-coef 0
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from vllm import SamplingParams

import grpo_train as G

# Separate /tmp paths so a rewind run never collides with a GRPO run.
G.SYNC_WEIGHTS_PATH = Path("/tmp/rewind_sync_weights")
G.LORA_TEMP_PATH = Path("/tmp/rewind_lora_sync")


def snap_to_whitespace(text: str, idx: int) -> int:
    """Move a cut point back to the nearest whitespace, so we never split a number."""
    if idx <= 0 or idx >= len(text):
        return max(0, min(idx, len(text)))
    j = idx
    while j > 0 and not text[j - 1].isspace():
        j -= 1
    return j if j > 0 else idx


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # identical to grpo_train's surface so the baseline can be reproduced
    p.add_argument("--student", default="Qwen/Qwen3-1.7B")
    p.add_argument("--sft-adapter", type=Path, default=Path("checkpoints/sft_coldstart/final"))
    p.add_argument("--no-sft", action="store_true")
    p.add_argument("--exclude-benchmarks", nargs="*", default=["mmlu"])
    p.add_argument("--train-data", type=Path, default=Path("data/zpd_filtered/all_zpd.jsonl"))
    p.add_argument("--output-dir", type=Path, default=Path("checkpoints/rewind"))
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--questions-per-step", type=int, default=8)
    p.add_argument("--num-rollouts", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--sync-steps", type=int, default=20)
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--warmup-steps", type=int, default=25)
    p.add_argument("--vllm-gpu-mem", type=float, default=0.45)
    p.add_argument("--kl-coef", type=float, default=0.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-think", action="store_true", default=True)
    # rewind-specific
    p.add_argument("--rewind-frac", type=float, default=0.4,
                   help="Fraction of the failed attempt to keep. 0.4 was the best "
                        "single fixed cut in probe_rewind.py (27%% rescue rate).")
    p.add_argument("--rewind-max-failures", type=int, default=3,
                   help="Stop attempting rewind on a question after this many "
                        "consecutive rescue failures (saves K generations per visit).")
    p.add_argument("--no-rewind", action="store_true",
                   help="Disable rewind entirely — reproduces the GRPO baseline.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    questions = G.load_jsonl(args.train_data)
    n_all = len(questions)
    if args.exclude_benchmarks:
        questions = [q for q in questions if q["benchmark"] not in args.exclude_benchmarks]
        print(f"Excluded {n_all - len(questions)} items from {args.exclude_benchmarks}")
    print(f"Loaded {len(questions)} ZPD questions (of {n_all})")
    print(f"Backbone: base{'+SFT' if G.use_sft(args) else ' only (--no-sft)'}")
    print(f"Rewind: {'DISABLED' if args.no_rewind else f'cut at {args.rewind_frac:.0%} of failed attempt, GSM8K only'}")

    ckpt_dirs = sorted(args.output_dir.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
    resume_ckpt = ckpt_dirs[-1] if ckpt_dirs else None
    if resume_ckpt:
        print(f"Resuming from {resume_ckpt}")

    tokenizer = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    eos_ids = G.get_eos_ids(tokenizer)

    student = G.load_student(args, resume_ckpt)

    print("\nPreparing initial vLLM weights ...")
    G.SYNC_WEIGHTS_PATH.mkdir(parents=True, exist_ok=True)
    G.save_merged_for_vllm(student, tokenizer, args, G.SYNC_WEIGHTS_PATH)

    print(f"\nLoading vLLM (gpu_memory_utilization={args.vllm_gpu_mem}) ...")
    vllm_model = G.load_vllm(args, G.SYNC_WEIGHTS_PATH)

    sampling_params = SamplingParams(
        n=1, temperature=args.temperature, top_p=0.95,
        max_tokens=args.max_new_tokens, stop_token_ids=eos_ids,
        skip_special_tokens=False,
    )

    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    opt_step = 0
    if resume_ckpt and (resume_ckpt / "trainer_state.pt").exists():
        state = torch.load(resume_ckpt / "trainer_state.pt", map_location="cpu")
        opt_step = state["opt_step"]
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["rng"])
        print(f"Resumed at step {opt_step}")

    log_path = args.output_dir / "rewind_log.jsonl"
    fresh = {"reward_sum": 0.0, "n_rollouts": 0, "groups": 0, "dead_groups": 0,
             "loss_sum": 0.0, "n_backward": 0, "kl_sum": 0.0,
             "rewind_tried": 0, "rewind_rescued": 0, "trained_groups": 0}
    window = dict(fresh)
    rescue_fail: dict[str, int] = {}      # question_id -> consecutive rescue failures
    t_start = time.perf_counter()
    cum_rollouts = 0
    pbar = tqdm(total=args.max_steps, initial=opt_step, desc="REWIND")

    while opt_step < args.max_steps:
        batch_qs = [random.choice(questions) for _ in range(args.questions_per_step)]
        prompts: list[str] = []
        for q in batch_qs:
            prompts.extend([G.build_prompt(tokenizer, q, args.no_think)] * args.num_rollouts)

        if vllm_model is None:
            vllm_model = G.sync_vllm(student, tokenizer, None, args, opt_step)
            if vllm_model is None:
                continue

        t_gen = time.perf_counter()
        outputs = vllm_model.generate(prompts, sampling_params)
        gen_time = time.perf_counter() - t_gen
        total_gen_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        cum_rollouts += len(outputs)

        # ── build the list of trainable groups ────────────────────────────────
        # each entry: (prompt_ids, [(token_ids, reward)], question)
        trainable: list[tuple[list[int], list[Any], list[float]]] = []
        rewind_jobs: list[tuple[int, dict, str]] = []   # (group_idx, question, prefix)

        for gi, q in enumerate(batch_qs):
            group = outputs[gi * args.num_rollouts: (gi + 1) * args.num_rollouts]
            comps = [o.outputs[0] for o in group if len(o.outputs[0].token_ids) > 0]
            if len(comps) < 2:
                continue
            texts = [tokenizer.decode(c.token_ids, skip_special_tokens=True) for c in comps]
            rewards = [1.0 if G.check_correct(t, q) else 0.0 for t in texts]
            window["reward_sum"] += sum(rewards)
            window["n_rollouts"] += len(rewards)
            window["groups"] += 1

            if G.group_advantages(rewards) is not None:
                trainable.append((list(group[0].prompt_token_ids),
                                  [c.token_ids for c in comps], rewards))
                continue

            # dead group
            window["dead_groups"] += 1
            if args.no_rewind or q["benchmark"] != "gsm8k" or any(rewards):
                continue                        # all-correct dead groups: nothing to gain
            if rescue_fail.get(str(q["question_id"]), 0) >= args.rewind_max_failures:
                continue                        # repeatedly unrescuable, stop paying for it
            longest = max(texts, key=len)
            if len(longest) < 40:
                continue
            cut = snap_to_whitespace(longest, int(args.rewind_frac * len(longest)))
            if cut <= 0:
                continue
            rewind_jobs.append((gi, q, longest[:cut]))

        # ── rewind pass: regenerate from partial work ─────────────────────────
        if rewind_jobs:
            rw_prompts: list[str] = []
            for _, q, prefix in rewind_jobs:
                base = G.build_prompt(tokenizer, q, args.no_think)
                rw_prompts.extend([base + prefix] * args.num_rollouts)
            rw_out = vllm_model.generate(rw_prompts, sampling_params)
            cum_rollouts += len(rw_out)
            total_gen_tokens += sum(len(o.outputs[0].token_ids) for o in rw_out)

            for j, (_, q, prefix) in enumerate(rewind_jobs):
                grp = rw_out[j * args.num_rollouts: (j + 1) * args.num_rollouts]
                comps = [o.outputs[0] for o in grp if len(o.outputs[0].token_ids) > 0]
                if len(comps) < 2:
                    continue
                window["rewind_tried"] += 1
                # score prefix+continuation: the answer may be produced by either part
                rewards = [
                    1.0 if G.check_correct(
                        prefix + tokenizer.decode(c.token_ids, skip_special_tokens=True), q
                    ) else 0.0
                    for c in comps
                ]
                qid = str(q["question_id"])
                if G.group_advantages(rewards) is None:
                    rescue_fail[qid] = rescue_fail.get(qid, 0) + 1
                    continue
                rescue_fail[qid] = 0
                window["rewind_rescued"] += 1
                # prefix is CONTEXT, not action: it lives in prompt_token_ids, so
                # gradient flows only through the continuation. Correct on-policy.
                trainable.append((list(grp[0].prompt_token_ids),
                                  [c.token_ids for c in comps], rewards))

        # ── update ────────────────────────────────────────────────────────────
        if not trainable:
            continue
        optimizer.zero_grad()
        n_backward = 0
        denom = sum(len(t[1]) for t in trainable)

        for prompt_ids, tok_lists, rewards in trainable:
            advantages = G.group_advantages(rewards)
            if advantages is None:
                continue
            window["trained_groups"] += 1
            for toks, adv in zip(tok_lists, advantages):
                full_ids = torch.tensor(prompt_ids + list(toks))
                try:
                    loss, kl_ref = G.policy_gradient_loss(
                        student, full_ids, len(prompt_ids), adv, args.kl_coef)
                    (loss / denom).backward()
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print("[OOM] rollout skipped, cache cleared.")
                    continue
                window["loss_sum"] += loss.item()
                window["kl_sum"] += kl_ref
                n_backward += 1

        if n_backward == 0:
            continue
        window["n_backward"] += n_backward
        torch.nn.utils.clip_grad_norm_(
            [p for p in student.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        opt_step += 1
        pbar.update(1)

        if opt_step % args.logging_steps == 0:
            mean_reward = window["reward_sum"] / max(1, window["n_rollouts"])
            dead_rate = window["dead_groups"] / max(1, window["groups"])
            tried, rescued = window["rewind_tried"], window["rewind_rescued"]
            log_row = {
                "step": opt_step,
                "mean_reward": round(mean_reward, 4),
                "dead_group_rate": round(dead_rate, 3),
                "rewind_tried": tried,
                "rewind_rescued": rescued,
                "rescue_rate": round(rescued / tried, 3) if tried else None,
                "trained_groups": window["trained_groups"],
                "loss": round(window["loss_sum"] / max(1, window["n_backward"]), 4),
                "kl_to_ref": round(window["kl_sum"] / max(1, window["n_backward"]), 5),
                "lr": optimizer.param_groups[0]["lr"],
                "gen_tok_per_sec": round(total_gen_tokens / max(gen_time, 1e-6), 1),
                "elapsed_s": round(time.perf_counter() - t_start, 1),
                "cum_rollouts": cum_rollouts,
                "blacklisted": sum(1 for v in rescue_fail.values()
                                   if v >= args.rewind_max_failures),
            }
            G.append_jsonl(log_path, log_row)
            pbar.set_postfix(reward=f"{mean_reward:.3f}", dead=f"{dead_rate:.1%}",
                             resc=f"{rescued}/{tried}")
            window = dict(fresh)

        if opt_step % args.save_steps == 0:
            ckpt = args.output_dir / f"step_{opt_step}"
            student.save_pretrained(str(ckpt))
            tokenizer.save_pretrained(str(ckpt))
            torch.save({"opt_step": opt_step, "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "rng": torch.get_rng_state()},
                       ckpt / "trainer_state.pt")
            with (args.output_dir / "rescue_fail.json").open("w") as f:
                json.dump(rescue_fail, f)
            print(f"\nSaved checkpoint: {ckpt}")

        if opt_step % args.sync_steps == 0:
            vllm_model = G.sync_vllm(student, tokenizer, vllm_model, args, opt_step)

    pbar.close()
    final_path = args.output_dir / "final"
    student.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"\nRewind-GRPO complete. Final adapter -> {final_path}")
    print(f"Copy the merged snapshot out of /tmp now:\n"
          f"  cp -r {G.SYNC_WEIGHTS_PATH} checkpoints/rewind_merged")


if __name__ == "__main__":
    main()
