#!/usr/bin/env python3
"""Feasibility probe for rewind-GRPO. No training, no teacher.

Answers three questions before any trainer gets built:

  1. Are V-curves clean? Rewinding assumes value collapses sharply at the point
     the reasoning breaks. If V decays gradually or non-monotonically, "the error
     step" is not well defined and the whole premise is shaky.
  2. What fraction of dead-group questions are salvageable? If V(bare question)
     is already 0 at this sample count, the question is simply too hard and
     belongs in the discard pile (the c=0 case in difficulty-filtering work).
  3. Can the student localise its own errors? Compares an LLM-judge localiser
     (student, shown the correct answer) against MC ground truth. The metric
     that matters is not index agreement but USABILITY: of the prefixes the
     judge picks, what fraction actually have V > 0?

V(prefix) is a Monte-Carlo value estimate: sample N continuations from the
current policy and count how many reach the correct answer. Probing is done over
TOKEN positions, not "steps" — no sentence segmentation, so messy formatting
(two steps on one line) is a non-issue. Boundaries snap to whitespace so we
never cut mid-number.

Usage:
  VLLM_USE_V1=0 python probe_rewind.py --model checkpoints/grpo1000_kl_merged \\
      --n-questions 100 --probes 6 --n-continuations 8
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics as stats
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


SYSTEM_PROMPT = "Think through this step by step."


# ── Data / extraction (kept identical to grpo_train.py so rewards match) ───────

def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


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


def is_correct(response: str, gt: str) -> bool:
    pred = extract_gsm8k(response)
    if pred is None:
        return False
    try:
        return math.isclose(float(pred), float(str(gt).replace(",", "")), abs_tol=1e-6)
    except ValueError:
        return False


def build_user_prompt(question: str) -> str:
    """Must match grpo_train.py / eval_baseline.py exactly."""
    return (f"Question: {question}\n\n"
            "Show your reasoning, then on the last line write only: Answer: [number]")


def chat_prompt(tokenizer: Any, user: str, prefill: str = "") -> str:
    """Chat-templated prompt, optionally with an assistant prefill to continue from.

    The prefill is how 'rewinding' works: the model continues an assistant turn
    that already contains the good prefix. No special machinery needed.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    try:
        text = tokenizer.apply_chat_template(messages, **kwargs, enable_thinking=False)
    except TypeError:
        text = tokenizer.apply_chat_template(messages, **kwargs)
    return text + prefill


def snap_to_whitespace(text: str, idx: int) -> int:
    """Move a cut point back to the nearest whitespace so we never split a number."""
    if idx <= 0 or idx >= len(text):
        return max(0, min(idx, len(text)))
    j = idx
    while j > 0 and not text[j - 1].isspace():
        j -= 1
    return j if j > 0 else idx


# ── MC value estimation ────────────────────────────────────────────────────────

def probe_values(
    llm: LLM,
    tokenizer: Any,
    question: str,
    gt: str,
    rollout: str,
    cut_points: list[int],
    n_cont: int,
    sp: SamplingParams,
) -> list[float]:
    """V for each prefix length in cut_points, batched into one vLLM call."""
    user = build_user_prompt(question)
    prompts: list[str] = []
    for cp in cut_points:
        prefix = rollout[:cp]
        prompts.extend([chat_prompt(tokenizer, user, prefix)] * n_cont)

    outs = llm.generate(prompts, sp)
    values = []
    for i, cp in enumerate(cut_points):
        chunk = outs[i * n_cont:(i + 1) * n_cont]
        prefix = rollout[:cp]
        # score prefix+continuation, since the answer may be completed by either
        ok = sum(is_correct(prefix + o.outputs[0].text, gt) for o in chunk)
        values.append(ok / max(1, len(chunk)))
    return values


def curve_shape(values: list[float]) -> str:
    """Classify a V-curve: does it collapse cleanly, or drift?"""
    if all(v == 0 for v in values):
        return "all-zero"
    if all(v > 0 for v in values):
        return "never-collapses"
    # find first zero after a positive
    first_pos = next(i for i, v in enumerate(values) if v > 0)
    after = values[first_pos:]
    zeros = [i for i, v in enumerate(after) if v == 0]
    if not zeros:
        return "never-collapses"
    first_zero = zeros[0]
    # clean = once it hits zero it stays zero, and the drop is large
    stays = all(v == 0 for v in after[first_zero:])
    drop = after[first_zero - 1] - after[first_zero] if first_zero > 0 else after[0]
    if stays and drop >= 0.25:
        return "clean-collapse"
    if stays:
        return "soft-collapse"
    return "non-monotone"


# ── Student-as-judge localiser ─────────────────────────────────────────────────

JUDGE_TEMPLATE = """A student solved this problem incorrectly.

Problem: {question}

Student's attempt:
{numbered}

The correct final answer is {gt}.

Exactly one of the numbered lines above is where the reasoning FIRST goes wrong.
Everything before that line is fine. Reply with only that line number, nothing else.
Line number:"""


def judge_localise(
    llm: LLM,
    tokenizer: Any,
    question: str,
    gt: str,
    rollout: str,
    sp_judge: SamplingParams,
) -> int | None:
    """Ask the student which line first goes wrong. Returns a char offset."""
    lines = [l for l in rollout.split("\n")]
    numbered = "\n".join(f"{i+1}: {l}" for i, l in enumerate(lines))
    prompt = chat_prompt(tokenizer, JUDGE_TEMPLATE.format(
        question=question, numbered=numbered, gt=gt))
    out = llm.generate([prompt], sp_judge)[0].outputs[0].text
    m = re.search(r"\d+", out)
    if not m:
        return None
    line_no = int(m.group()) - 1
    if not (0 <= line_no < len(lines)):
        return None
    # char offset of the START of that line = keep everything before it
    return len("\n".join(lines[:line_no])) + (1 if line_no > 0 else 0)


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="checkpoints/grpo1000_kl_merged",
                   help="Merged model dir (base+adapter), as vLLM needs full weights.")
    p.add_argument("--train-data", type=Path, default=Path("data/zpd_filtered/gsm8k_zpd.jsonl"))
    p.add_argument("--n-questions", type=int, default=100,
                   help="Dead-group questions to analyse (sampled from those found).")
    p.add_argument("--screen-rollouts", type=int, default=8,
                   help="Rollouts per question when screening for dead groups.")
    p.add_argument("--probes", type=int, default=6, help="Cut points per rollout.")
    p.add_argument("--n-continuations", type=int, default=8, help="N for each V estimate.")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--vllm-gpu-mem", type=float, default=0.55)
    p.add_argument("--judge", action="store_true", default=True,
                   help="Also run the student-as-judge localiser and score its usability.")
    p.add_argument("--out", type=Path, default=Path("results/rewind_probe.json"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    questions = [q for q in load_jsonl(args.train_data) if q["benchmark"] == "gsm8k"]
    print(f"Loaded {len(questions)} GSM8K ZPD questions")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(model=args.model, dtype="bfloat16", trust_remote_code=True,
              enforce_eager=True, swap_space=0,
              gpu_memory_utilization=args.vllm_gpu_mem)

    sp = SamplingParams(n=1, temperature=args.temperature, top_p=0.95,
                        max_tokens=args.max_new_tokens)
    sp_judge = SamplingParams(n=1, temperature=0.0, max_tokens=16)

    # ── Stage 1: screen for dead groups ───────────────────────────────────────
    print(f"\nScreening for dead groups ({args.screen_rollouts} rollouts/question) ...")
    random.shuffle(questions)
    screen_pool = questions[: max(args.n_questions * 4, args.n_questions)]
    prompts = []
    for q in screen_pool:
        prompts.extend([chat_prompt(tokenizer, build_user_prompt(q["question"]))]
                       * args.screen_rollouts)
    outs = llm.generate(prompts, sp)

    dead: list[tuple[dict, str]] = []
    n_correct_any = 0
    for i, q in enumerate(screen_pool):
        chunk = outs[i * args.screen_rollouts:(i + 1) * args.screen_rollouts]
        texts = [o.outputs[0].text for o in chunk]
        flags = [is_correct(t, q["ground_truth"]) for t in texts]
        if any(flags):
            n_correct_any += 1
        else:
            dead.append((q, max(texts, key=len)))   # analyse the longest failure

    print(f"  {len(screen_pool)} questions screened: "
          f"{n_correct_any} had >=1 correct (GRPO handles these), "
          f"{len(dead)} all-wrong (rewind targets these) "
          f"= {100*len(dead)/max(1,len(screen_pool)):.1f}% dead")
    dead = dead[: args.n_questions]
    if not dead:
        raise SystemExit("No dead groups found — nothing for rewind to do.")

    # ── Stage 2: MC value curves ──────────────────────────────────────────────
    print(f"\nProbing V-curves on {len(dead)} all-wrong rollouts "
          f"({args.probes} cut points x {args.n_continuations} continuations) ...")
    records = []
    for q, rollout in tqdm(dead, desc="probe"):
        L = len(rollout)
        if L < 40:
            continue
        raw_cuts = [round(L * f / args.probes) for f in range(args.probes)]
        cuts = sorted({snap_to_whitespace(rollout, c) for c in raw_cuts})
        values = probe_values(llm, tokenizer, q["question"], q["ground_truth"],
                              rollout, cuts, args.n_continuations, sp)

        usable = [c for c, v in zip(cuts, values) if v > 0]
        rec = {
            "question_id": q["question_id"],
            "rollout_len": L,
            "cuts": cuts,
            "values": values,
            "shape": curve_shape(values),
            "best_cut": max(usable) if usable else None,
            "best_v": max(values),
            "v_at_zero_prefix": values[0] if cuts and cuts[0] == 0 else None,
        }

        if args.judge:
            jcut = judge_localise(llm, tokenizer, q["question"], q["ground_truth"],
                                  rollout, sp_judge)
            rec["judge_cut"] = jcut
            if jcut is not None:
                jv = probe_values(llm, tokenizer, q["question"], q["ground_truth"],
                                  rollout, [snap_to_whitespace(rollout, jcut)],
                                  args.n_continuations, sp)[0]
                rec["judge_v"] = jv
        records.append(rec)

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("1. ARE V-CURVES CLEAN?")
    shapes: dict[str, int] = {}
    for r in records:
        shapes[r["shape"]] = shapes.get(r["shape"], 0) + 1
    for s, n in sorted(shapes.items(), key=lambda kv: -kv[1]):
        print(f"   {s:<18} {n:>4}  ({100*n/len(records):5.1f}%)")
    print("   clean-collapse => rewind premise holds; non-monotone/soft => shaky")

    print("\n2. ARE DEAD-GROUP QUESTIONS SALVAGEABLE?")
    salvageable = [r for r in records if r["best_cut"] is not None]
    print(f"   {len(salvageable)}/{len(records)} ({100*len(salvageable)/len(records):.1f}%) "
          f"have some prefix with V>0 -> rewind can create signal")
    print(f"   {len(records)-len(salvageable)} have V=0 everywhere -> too hard, discard")
    if salvageable:
        fr = [r["best_cut"] / r["rollout_len"] for r in salvageable]
        bv = [r["best_v"] for r in salvageable]
        print(f"   best cut sits at {100*stats.mean(fr):.0f}% of the rollout on average")
        print(f"   V at that cut: mean {stats.mean(bv):.3f}, median {stats.median(bv):.3f}")

    if args.judge:
        print("\n3. CAN THE STUDENT LOCALISE ITS OWN ERRORS?")
        judged = [r for r in records if r.get("judge_cut") is not None and "judge_v" in r]
        print(f"   parsed a line number for {len(judged)}/{len(records)}")
        if judged:
            usable = [r for r in judged if r["judge_v"] > 0]
            print(f"   USABILITY: {len(usable)}/{len(judged)} "
                  f"({100*len(usable)/len(judged):.1f}%) of judge-picked prefixes have V>0")
            print("   (>=70% => judge is a viable cheap localiser; <=40% => use MC)")
            both = [r for r in judged if r["best_cut"] is not None]
            if both:
                d = [abs(r["judge_cut"] - r["best_cut"]) / r["rollout_len"] for r in both]
                print(f"   mean |judge - MC| = {100*stats.mean(d):.0f}% of rollout length")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump({"args": vars(args) | {"train_data": str(args.train_data),
                                         "out": str(args.out)},
                   "records": records}, f, indent=2)
    print(f"\nPer-question detail -> {args.out}")


if __name__ == "__main__":
    main()
