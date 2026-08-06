#!/usr/bin/env python3
"""Per-question transition analysis between two eval runs.

Counts and characterises what a training run LEARNED (was wrong, now right) and
what it FORGOT (was right, now wrong) — the detail that a single accuracy delta
hides. A run that is +0.8pp net may have rewritten 10% of its answers.

Also checks several ways an apparent gain can be an artifact:
  - extraction failures (format regression, not reasoning loss)
  - answer-distribution bias (e.g. drifting toward "yes" on StrategyQA, or
    toward one letter on MMLU) which can raise accuracy without reasoning
  - response-length inflation, a common RL side effect

Usage:
  python analyze_transitions.py --base results/qwen3-1.7b-actual \
                                --new  results/qwen3-1.7b-grpo \
                                --label GRPO --examples 5
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from math import sqrt
from pathlib import Path
from typing import Any

BENCHMARKS = ["gsm8k", "mmlu", "strategyqa"]


def load_per_example(results_dir: Path, bench: str) -> dict[str, dict] | None:
    path = results_dir / f"{bench}_baseline_results.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(r["question_id"]): r for r in data["per_example"]}


def pct(n: int, d: int) -> str:
    return f"{100*n/d:5.1f}%" if d else "  n/a"


def transition_summary(base: dict, new: dict, label: str) -> tuple[list, list]:
    ids = sorted(set(base) & set(new))
    learned = [i for i in ids if not base[i]["correct"] and new[i]["correct"]]
    forgot = [i for i in ids if base[i]["correct"] and not new[i]["correct"]]
    kept_ok = [i for i in ids if base[i]["correct"] and new[i]["correct"]]
    kept_bad = [i for i in ids if not base[i]["correct"] and not new[i]["correct"]]

    n = len(ids)
    z = (len(learned) - len(forgot)) / sqrt(len(learned) + len(forgot)) if (learned or forgot) else 0.0
    churn = len(learned) + len(forgot)

    print(f"  n={n}")
    print(f"  base acc {pct(len(kept_ok)+len(forgot), n)}   {label} acc {pct(len(kept_ok)+len(learned), n)}")
    print(f"  learned   {len(learned):>5}  ({pct(len(learned), n)})")
    print(f"  forgotten {len(forgot):>5}  ({pct(len(forgot), n)})")
    print(f"  net       {len(learned)-len(forgot):>+5}   McNemar z={z:+.2f}")
    print(f"  churn     {churn:>5}  ({pct(churn, n)} of answers changed correctness)")
    if churn:
        print(f"  churn ratio: {len(learned)/max(1,len(forgot)):.2f} learned per forgotten")
    return learned, forgot


def extraction_check(base: dict, new: dict, forgot: list[str], label: str) -> None:
    """Regressions caused by unparseable output are a format bug, not lost reasoning."""
    ef_new = sum(1 for i in forgot if new[i].get("extraction_failed"))
    ef_base_all = sum(1 for r in base.values() if r.get("extraction_failed"))
    ef_new_all = sum(1 for r in new.values() if r.get("extraction_failed"))
    print(f"  extraction failures: base {ef_base_all} -> {label} {ef_new_all}")
    if forgot:
        print(f"  of {len(forgot)} forgotten, {ef_new} ({pct(ef_new, len(forgot))}) failed extraction"
              f"{'  <-- format regression, not reasoning' if ef_new > 0.2*len(forgot) else ''}")


def answer_bias(base: dict, new: dict, bench: str, label: str) -> None:
    """A binary/multiple-choice task can gain accuracy purely by shifting priors."""
    if bench not in ("strategyqa", "mmlu"):
        return
    def dist(d: dict) -> Counter:
        return Counter(str(r.get("extracted_answer")).lower() for r in d.values())
    gt = Counter(str(r["ground_truth"]).lower() for r in new.values())
    db, dn = dist(base), dist(new)
    keys = sorted(set(gt) | set(db) | set(dn), key=lambda k: -gt.get(k, 0))[:5]
    print(f"  answer distribution (truth / base / {label}):")
    for k in keys:
        tot = len(new)
        print(f"    {k:<8} {pct(gt.get(k,0), tot)} / {pct(db.get(k,0), tot)} / {pct(dn.get(k,0), tot)}")


def length_check(base: dict, new: dict, label: str) -> None:
    """response char length is trustworthy; tokens_generated is NOT (counts padding)."""
    lb = [len(r.get("response", "")) for r in base.values()]
    ln = [len(r.get("response", "")) for r in new.values()]
    mb, mn = sum(lb)/max(1,len(lb)), sum(ln)/max(1,len(ln))
    delta = 100*(mn-mb)/mb if mb else 0
    print(f"  mean response chars: base {mb:.0f} -> {label} {mn:.0f}  ({delta:+.1f}%)")


def subject_breakdown(base: dict, new: dict, learned: list, forgot: list, label: str) -> None:
    """MMLU only: is interference concentrated in particular subjects?"""
    if not any("subject" in r for r in new.values()):
        return
    per: dict[str, list[int]] = {}
    for i in learned:
        per.setdefault(new[i].get("subject", "?"), [0, 0])[0] += 1
    for i in forgot:
        per.setdefault(new[i].get("subject", "?"), [0, 0])[1] += 1
    rows = sorted(per.items(), key=lambda kv: kv[1][1] - kv[1][0], reverse=True)
    print(f"  worst-hit subjects (forgot - learned):")
    for subj, (lr, fg) in rows[:8]:
        print(f"    {subj:<40} learned {lr:>3}  forgot {fg:>3}  net {lr-fg:>+4}")
    print(f"  most-improved subjects:")
    for subj, (lr, fg) in rows[-5:][::-1]:
        print(f"    {subj:<40} learned {lr:>3}  forgot {fg:>3}  net {lr-fg:>+4}")


def dump_examples(base: dict, new: dict, ids: list[str], title: str, n: int, out) -> None:
    out.write(f"\n{'='*78}\n{title} ({len(ids)} total, showing {min(n,len(ids))})\n{'='*78}\n")
    for i in ids[:n]:
        out.write(f"\n--- question_id={i}  ground_truth={new[i]['ground_truth']}\n")
        if "subject" in new[i]:
            out.write(f"subject: {new[i]['subject']}\n")
        out.write(f"Q: {new[i]['question'][:400]}\n")
        out.write(f"\n[base]  extracted={base[i].get('extracted_answer')!r}\n")
        out.write(f"{base[i].get('response','')[-600:]}\n")
        out.write(f"\n[new]   extracted={new[i].get('extracted_answer')!r}\n")
        out.write(f"{new[i].get('response','')[-600:]}\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", type=Path, default=Path("results/qwen3-1.7b-actual"))
    p.add_argument("--new", type=Path, required=True)
    p.add_argument("--label", default="new")
    p.add_argument("--examples", type=int, default=5,
                   help="Examples of each transition to dump to the text report.")
    p.add_argument("--out", type=Path, default=None,
                   help="Where to write example dumps (default: <new>/transitions.txt)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = args.out or (args.new / "transitions.txt")
    ids_path = args.new / "transitions.json"
    all_ids: dict[str, dict[str, list[str]]] = {}

    with out_path.open("w", encoding="utf-8") as out:
        out.write(f"Transition report: {args.base} -> {args.new} ({args.label})\n")
        for bench in BENCHMARKS:
            base = load_per_example(args.base, bench)
            new = load_per_example(args.new, bench)
            if base is None or new is None:
                print(f"\n### {bench.upper()}: missing results, skipped")
                continue

            print(f"\n### {bench.upper()}")
            learned, forgot = transition_summary(base, new, args.label)
            extraction_check(base, new, forgot, args.label)
            length_check(base, new, args.label)
            answer_bias(base, new, bench, args.label)
            subject_breakdown(base, new, learned, forgot, args.label)

            all_ids[bench] = {"learned": learned, "forgotten": forgot}
            dump_examples(base, new, learned, f"{bench.upper()} LEARNED", args.examples, out)
            dump_examples(base, new, forgot, f"{bench.upper()} FORGOTTEN", args.examples, out)

    with ids_path.open("w", encoding="utf-8") as f:
        json.dump(all_ids, f, indent=2)

    print(f"\nExample transcripts -> {out_path}")
    print(f"Full id lists        -> {ids_path}")
    print("\nNote: tokens_generated / latency in these result files are unreliable")
    print("(padding counted as output); only accuracy and response text are trustworthy.")


if __name__ == "__main__":
    main()
