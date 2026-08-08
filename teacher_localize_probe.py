#!/usr/bin/env python3
"""Does a teacher localise the student's first mistake better than a blind cut?

Rewind-GRPO currently cuts a failed attempt at a fixed 40% of its length and
regenerates from there. That rescues 27% of dead groups in probing (18.8% in
actual training). A 6-point MC grid search finds a usable prefix 50% of the
time, so better localisation is worth up to ~1.85x more rescues.

This measures whether an 8B teacher, shown the reference solution, picks better
cut points than the blind 40% heuristic. Both are scored the same way:

    usability = fraction of chosen prefixes with V > 0

where V is the Monte-Carlo success rate — sample N continuations from the prefix
and count how many reach the correct answer. A prefix with V > 0 is one rewind
can actually train from; a prefix with V = 0 produces another dead group.

Decision thresholds (vs the 27% fixed-cut baseline, re-measured here on the
same questions so the comparison is paired):
    >=45%   clearly better -> build teacher-localised rewind
    30-40%  marginal -> probably not worth +36% wall clock
    ~27%    localisation is not the bottleneck -> stop, keep the 40% cut

Also reports the teacher's UNHINTED solve rate, which is how often the
competence gate would fire if you later add KL at the located position (the KL
target must be answer-free, so it needs a teacher that can actually solve it).

Both models are resident: teacher ~16.4GB + vLLM at --vllm-gpu-mem 0.2 (~9.5GB).

Usage:
  VLLM_USE_V1=0 python teacher_localize_probe.py --n-questions 100
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as stats
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

import probe_rewind as P


# ── reference solutions ────────────────────────────────────────────────────────

def clean_solution(raw: str) -> str:
    """Strip <<8*18=144>> calculator annotations and the '#### N' terminator."""
    body = raw.split("####")[0]
    return re.sub(r"<<[^>]*>>", "", body).strip()


# ── student attempt -> numbered lines the teacher can point at ─────────────────

# ── locating the teacher's quoted span in the original text ───────────────────
#
# Line numbering was the wrong approach. Measured on real failed GSM8K rollouts:
# 13-118 non-empty lines (median 54), ~5 pure '---' rules each, plus bare '$$'
# delimiters, markdown headers and inline LaTeX. Any line/unit scheme imposes
# structure the data does not reliably have, and picking 1-of-54 is a hard task.
#
# Instead the teacher QUOTES the offending text verbatim and we locate it by
# string match. That assumes nothing about formatting. The failure mode —
# paraphrasing instead of quoting — is measurable and handled by a three-tier
# match, and every record reports which tier fired.

def _norm_map(text: str) -> tuple[str, list[int]]:
    """Whitespace-collapsed text plus, for each output char, its original index."""
    out: list[str] = []
    idx: list[int] = []
    prev_space = False
    for i, ch in enumerate(text):
        if ch.isspace():
            if not prev_space and out:
                out.append(" ")
                idx.append(i)
            prev_space = True
        else:
            out.append(ch)
            idx.append(i)
            prev_space = False
    return "".join(out), idx


MAX_QUOTE_CHARS = 400          # a step is short; anything longer is commentary


def extract_quote(raw: str) -> str:
    """The <quote> block, if the teacher produced one and it is plausibly a span."""
    tagged = re.findall(r"<quote>(.*?)</quote>", raw, re.DOTALL)
    if not tagged:
        return ""
    q = tagged[-1].strip()
    return q if len(q) <= MAX_QUOTE_CHARS else ""


def longest_shared_run(rollout: str, reply: str, min_chars: int = 20) -> int | None:
    """Offset of the longest verbatim run the reply shares with the rollout.

    Last-resort locator, and the one that survives a badly-behaved teacher. If it
    ignores the tags, rambles, or paraphrases around a quoted fragment, whatever
    it *did* copy verbatim still points at a position. Only fails when nothing
    substantial was copied at all.
    """
    import difflib
    sm = difflib.SequenceMatcher(None, rollout, reply, autojunk=False)
    m = sm.find_longest_match(0, len(rollout), 0, len(reply))
    return m.a if m.size >= min_chars else None


def locate_teacher_cut(rollout: str, raw: str) -> tuple[int | None, str]:
    """Turn a teacher reply into a cut offset. Reports which route succeeded."""
    quote = extract_quote(raw)
    if quote:
        off, tier = locate_span(rollout, quote)
        if off is not None:
            return off, tier
    off = longest_shared_run(rollout, raw)
    if off is not None:
        return off, "inline-verbatim" if quote else "no-tags-inline"
    return None, "failed"


def locate_span(rollout: str, quote: str, min_ratio: float = 0.4,
                min_chars: int = 20) -> tuple[int | None, str]:
    """Char offset of `quote` in `rollout`, plus which match tier succeeded."""
    quote = quote.strip()
    if len(quote) < 8:
        return None, "too-short"

    i = rollout.find(quote)                                    # tier 1: exact
    if i >= 0:
        return i, "exact"

    nr, nmap = _norm_map(rollout)                              # tier 2: whitespace-insensitive
    nq, _ = _norm_map(quote)
    j = nr.find(nq)
    if j >= 0:
        return nmap[j], "normalized"
    j = nr.lower().find(nq.lower())
    if j >= 0:
        return nmap[j], "normalized-ci"

    import difflib                                             # tier 3: fuzzy
    m = difflib.SequenceMatcher(None, nr.lower(), nq.lower()).find_longest_match(
        0, len(nr), 0, len(nq))
    # Absolute floor as well as a ratio: a 20+ char verbatim run is a strong
    # locator even inside a paraphrase, while unrelated text rarely reaches it.
    if m.size >= max(min_chars, int(min_ratio * len(nq))):
        return nmap[m.a], "fuzzy"
    return None, "failed"


CRITIQUE_TEMPLATE = """A student solved this problem incorrectly. You are shown a correct reference solution.

Problem: {question}

Reference solution:
{solution}
Correct final answer: {gt}

Student's attempt:
{attempt}

Find the FIRST place where the student's reasoning goes wrong. Everything before it \
should be correct reasoning.

Copy that text VERBATIM from the student's attempt — character for character, exactly \
as it appears — and put it between <quote> and </quote> tags. Quote one sentence or one \
step, not the whole attempt. Do not quote the student's final answer line unless the \
error is genuinely there.

Reply with the tags only, for example:
<quote>the exact text goes here</quote>"""


SOLVE_TEMPLATE = """Question: {question}

Show your reasoning, then on the last line write only: Answer: [number]"""


# ── batched teacher generation ─────────────────────────────────────────────────

@torch.no_grad()
def teacher_generate(
    model: Any, tok: Any, prompts: list[str], max_new: int, batch: int, think: bool
) -> list[str]:
    outs: list[str] = []
    for i in tqdm(range(0, len(prompts), batch), desc="teacher", leave=False):
        chunk = prompts[i:i + batch]
        texts = []
        for p in chunk:
            msgs = [{"role": "user", "content": p}]
            kw: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
            if not think:
                try:
                    texts.append(tok.apply_chat_template(msgs, **kw, enable_thinking=False))
                    continue
                except TypeError:
                    pass
            texts.append(tok.apply_chat_template(msgs, **kw))
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        for j in range(len(chunk)):
            outs.append(tok.decode(gen[j][enc["input_ids"].shape[1]:],
                                   skip_special_tokens=True))
    return outs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--student", default="checkpoints/grpo1000_nokl_merged",
                   help="Merged student for vLLM. Falls back to base if absent.")
    p.add_argument("--student-fallback", default="Qwen/Qwen3-1.7B")
    p.add_argument("--teacher", default="Qwen/Qwen3-8B")
    p.add_argument("--train-data", type=Path, default=Path("data/zpd_filtered/gsm8k_zpd.jsonl"))
    p.add_argument("--n-questions", type=int, default=100)
    p.add_argument("--screen-rollouts", type=int, default=8)
    p.add_argument("--n-continuations", type=int, default=8)
    p.add_argument("--fixed-frac", type=float, default=0.4,
                   help="The incumbent blind cut, re-measured here for a paired comparison.")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--teacher-batch", type=int, default=8)
    p.add_argument("--teacher-think", action="store_true",
                   help="OFF by default, and leave it off. With thinking on, Qwen3-8B spent "
                        "the whole token budget reasoning and never emitted the <quote> tags "
                        "at all, so the reply was pure internal monologue. Use the teacher as "
                        "an instruct model.")
    p.add_argument("--skip-solve-rate", action="store_true",
                   help="Skip measuring the teacher's unhinted solve rate (saves ~10min).")
    p.add_argument("--vllm-gpu-mem", type=float, default=0.2)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--out", type=Path, default=Path("results/teacher_localize_probe.json"))
    p.add_argument("--debug-dump", type=int, default=0,
                   help="Write full detail (raw teacher reply, extracted quote, text either "
                        "side of the cut, V) for the first N questions. Use with a small "
                        "--n-questions to sanity-check before a full run.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import random
    random.seed(args.seed)

    questions = [q for q in P.load_jsonl(args.train_data) if q["benchmark"] == "gsm8k"]
    print(f"Loaded {len(questions)} GSM8K ZPD questions")

    print("Loading GSM8K train split for reference solutions ...")
    ds = load_dataset("gsm8k", "main", split="train")
    sol_of = {str(q["question_id"]): clean_solution(ds[int(q["question_id"])]["answer"])
              for q in questions}

    # ── teacher first, so vLLM profiles the memory that is actually left ──────
    print(f"\nLoading teacher: {args.teacher}")
    t_tok = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    if t_tok.pad_token is None:
        t_tok.pad_token = t_tok.eos_token
    t_tok.padding_side = "left"
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    teacher.eval()
    dmap = getattr(teacher, "hf_device_map", {})
    offloaded = [k for k, v in dmap.items() if v in ("cpu", "disk")]
    if offloaded:
        print(f"[WARN] {len(offloaded)} teacher modules offloaded to CPU/disk — will be slow. "
              f"Lower --vllm-gpu-mem or free GPU memory.")
    else:
        print("Teacher fully on GPU.")

    student_path = args.student if Path(args.student).exists() else args.student_fallback
    print(f"\nLoading student into vLLM: {student_path}")
    s_tok = AutoTokenizer.from_pretrained(student_path, trust_remote_code=True)
    llm = LLM(model=student_path, dtype="bfloat16", trust_remote_code=True,
              enforce_eager=True, swap_space=0, gpu_memory_utilization=args.vllm_gpu_mem)
    sp = SamplingParams(n=1, temperature=args.temperature, top_p=0.95,
                        max_tokens=args.max_new_tokens)

    # ── stage 1: find dead groups ─────────────────────────────────────────────
    print(f"\nScreening for dead groups ({args.screen_rollouts} rollouts/question) ...")
    random.shuffle(questions)
    pool = questions[: args.n_questions * 4]
    prompts = []
    for q in pool:
        prompts.extend([P.chat_prompt(s_tok, P.build_user_prompt(q["question"]))]
                       * args.screen_rollouts)
    outs = llm.generate(prompts, sp)

    dead: list[tuple[dict, str]] = []
    for i, q in enumerate(pool):
        chunk = outs[i * args.screen_rollouts:(i + 1) * args.screen_rollouts]
        texts = [o.outputs[0].text for o in chunk]
        if not any(P.is_correct(t, q["ground_truth"]) for t in texts):
            dead.append((q, max(texts, key=len)))
    print(f"  {len(dead)}/{len(pool)} all-wrong ({100*len(dead)/len(pool):.1f}%)")
    dead = [(q, r) for q, r in dead if len(r) >= 40][: args.n_questions]
    if not dead:
        raise SystemExit("No usable dead groups found.")
    print(f"  analysing {len(dead)}")

    # ── stage 2: teacher picks a cut ──────────────────────────────────────────
    print("\nTeacher localising first error (reference solution provided) ...")
    crit_prompts = [
        CRITIQUE_TEMPLATE.format(
            question=q["question"], solution=sol_of[str(q["question_id"])],
            gt=q["ground_truth"], attempt=rollout)
        for q, rollout in dead
    ]
    crit_out = teacher_generate(teacher, t_tok, crit_prompts,
                                max_new=3072 if args.teacher_think else 256,
                                batch=args.teacher_batch, think=args.teacher_think)

    # ── stage 3: score both cuts with MC ──────────────────────────────────────
    print("\nScoring teacher cut vs fixed cut with MC ...")
    records = []
    debug: list[dict] = []
    tiers: dict[str, int] = {}
    for (q, rollout), raw in tqdm(list(zip(dead, crit_out)), desc="score"):
        quote = extract_quote(raw)
        t_cut, tier = locate_teacher_cut(rollout, raw)
        tiers[tier] = tiers.get(tier, 0) + 1
        if t_cut is not None:
            t_cut = P.snap_to_whitespace(rollout, t_cut)
        f_cut = P.snap_to_whitespace(rollout, int(args.fixed_frac * len(rollout)))

        cuts = sorted({c for c in (t_cut, f_cut) if c is not None and c > 0})
        vals = P.probe_values(llm, s_tok, q["question"], q["ground_truth"],
                              rollout, cuts, args.n_continuations, sp) if cuts else []
        v_of = dict(zip(cuts, vals))
        records.append({
            "question_id": q["question_id"],
            "rollout_len": len(rollout),
            "quote": quote[:200],
            "match_tier": tier,
            "teacher_cut": t_cut,
            "teacher_v": v_of.get(t_cut) if t_cut is not None else None,
            "fixed_cut": f_cut,
            "fixed_v": v_of.get(f_cut),
        })

        if len(debug) < args.debug_dump:
            debug.append({
                "q": q, "rollout": rollout, "solution": sol_of[str(q["question_id"])],
                "raw": raw, "quote": quote, "tier": tier,
                "t_cut": t_cut, "t_v": v_of.get(t_cut) if t_cut is not None else None,
                "f_cut": f_cut, "f_v": v_of.get(f_cut),
            })

    # ── stage 4: teacher solve rate (competence gate estimate) ────────────────
    solve_rate = None
    if not args.skip_solve_rate:
        print("\nMeasuring teacher UNHINTED solve rate (for the KL competence gate) ...")
        sp_prompts = [SOLVE_TEMPLATE.format(question=q["question"]) for q, _ in dead]
        sv = teacher_generate(teacher, t_tok, sp_prompts, max_new=512,
                              batch=args.teacher_batch, think=False)
        ok = sum(P.is_correct(t, q["ground_truth"]) for (q, _), t in zip(dead, sv))
        solve_rate = ok / len(dead)

    # ── report ────────────────────────────────────────────────────────────────
    n = len(records)
    t_scored = [r for r in records if r["teacher_cut"] is not None and r["teacher_v"] is not None]
    t_ok = [r for r in t_scored if r["teacher_v"] > 0]
    f_ok = [r for r in records if (r["fixed_v"] or 0) > 0]

    print("\n" + "=" * 70)
    print(f"USABILITY  (fraction of chosen prefixes with V>0, n={n})")
    print(f"  fixed {args.fixed_frac:.0%} cut : {len(f_ok):>3}/{n} = {100*len(f_ok)/n:5.1f}%   <- incumbent")
    if t_scored:
        print(f"  teacher cut    : {len(t_ok):>3}/{len(t_scored)} = "
              f"{100*len(t_ok)/len(t_scored):5.1f}%")
    print(f"\nQUOTE MATCHING (how the teacher's verbatim span was located)")
    for t, c in sorted(tiers.items(), key=lambda kv: -kv[1]):
        print(f"  {t:<14} {c:>3}/{n}  ({100*c/n:5.1f}%)")
    print("  exact/normalized = teacher quoted faithfully; fuzzy = paraphrased; "
          "failed = unusable, fell back")

    both = [r for r in t_scored if r["fixed_v"] is not None]
    if both:
        win = sum(1 for r in both if r["teacher_v"] > r["fixed_v"])
        lose = sum(1 for r in both if r["teacher_v"] < r["fixed_v"])
        print(f"\nPAIRED on the same rollouts: teacher better on {win}, worse on {lose}, "
              f"tied on {len(both)-win-lose}")
        tv = [r["teacher_v"] for r in both]; fv = [r["fixed_v"] for r in both]
        print(f"  mean V: teacher {stats.mean(tv):.3f} vs fixed {stats.mean(fv):.3f}")
        frac = sorted(r["teacher_cut"]/r["rollout_len"] for r in both)
        print(f"  teacher cuts at {100*stats.mean(frac):.0f}% of rollout on average "
              f"(fixed is {args.fixed_frac:.0%})")
        q = lambda f: frac[min(len(frac)-1, int(f*len(frac)))]
        print(f"  cut fraction spread: p10 {100*q(.1):.0f}%  p50 {100*q(.5):.0f}%  "
              f"p90 {100*q(.9):.0f}%")
        late = sum(1 for f in frac if f > 0.85)
        if late:
            print(f"  [!] {late} cuts land past 85% of the rollout — little left to regenerate")

    print("\nSAMPLE QUOTES (first 5, to eyeball whether it is pointing sensibly)")
    for r in records[:5]:
        frac = f"{100*r['teacher_cut']/r['rollout_len']:.0f}%" if r["teacher_cut"] else "  -"
        print(f"  [{r['match_tier']:<12} @{frac:>4} V={r['teacher_v']}] {r['quote'][:90]!r}")

    if solve_rate is not None:
        print(f"\nTEACHER SOLVE RATE on these dead questions: {100*solve_rate:.1f}%")
        print("  = how often a KL competence gate would ALLOW the KL term in run C")

    print("\nDECISION: >=45% build teacher-localised rewind | 30-40% marginal | "
          "~27% localisation is not the bottleneck")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump({"solve_rate": solve_rate, "match_tiers": tiers,
                   "records": records}, f, indent=2)
    print(f"\nPer-question detail -> {args.out}")

    if debug:
        dpath = args.out.with_name("teacher_localize_debug.txt")
        with dpath.open("w", encoding="utf-8") as f:
            for k, d in enumerate(debug, 1):
                f.write("=" * 78 + f"\nQUESTION {k}  (qid {d['q']['question_id']}, "
                        f"ground truth {d['q']['ground_truth']})\n" + "=" * 78 + "\n")
                f.write(f"\n--- PROBLEM ---\n{d['q']['question']}\n")
                f.write(f"\n--- REFERENCE SOLUTION (given to teacher) ---\n{d['solution']}\n")
                f.write(f"\n--- STUDENT ATTEMPT (all of it, as the teacher saw it) ---\n"
                        f"{d['rollout']}\n")
                f.write(f"\n--- TEACHER RAW REPLY ---\n{d['raw']}\n")
                f.write(f"\n--- EXTRACTED QUOTE ---\n{d['quote']!r}\n")
                f.write(f"\n--- LOCATION: tier={d['tier']} cut={d['t_cut']} "
                        f"({100*d['t_cut']/len(d['rollout']):.0f}% of rollout)\n"
                        if d["t_cut"] is not None else
                        f"\n--- LOCATION: tier={d['tier']} — NOT LOCATED, would fall back "
                        f"to the fixed cut\n")
                if d["t_cut"] is not None:
                    f.write(f"\n>>> PREFIX KEPT (last 300 chars before the cut) <<<\n"
                            f"...{d['rollout'][max(0, d['t_cut']-300):d['t_cut']]}\n")
                    f.write(f"\n>>> DISCARDED (first 300 chars after the cut) <<<\n"
                            f"{d['rollout'][d['t_cut']:d['t_cut']+300]}...\n")
                f.write(f"\n--- V: teacher cut={d['t_v']}   fixed 40% cut={d['f_v']}\n\n")
        print(f"Full debug dump      -> {dpath}")


if __name__ == "__main__":
    main()
