#!/usr/bin/env python3
"""Assemble SFT cold-start dataset from teacher traces.

Reads correct teacher traces, formats them for verl SFT training, and
writes train/val splits. Output format per example:
  {
    "instruction": "Think through this step by step.",
    "input": "<question>",
    "output": "<think>...</think><answer>...</answer>"
  }
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def answers_match(teacher: str, gt: str, benchmark: str) -> bool:
    try:
        if benchmark == "gsm8k":
            return math.isclose(float(str(teacher).replace(",", "")), float(gt), abs_tol=1e-6)
        return str(teacher).lower().strip() == str(gt).lower().strip()
    except Exception:
        return False


def format_example(trace: dict) -> dict:
    return {
        "instruction": "Think through this step by step.",
        "input": trace["question"],
        "output": trace["output"],
        "benchmark": trace["benchmark"],
        "subject": trace.get("subject", ""),
        "ground_truth": trace["ground_truth"],
    }


def print_dataset_stats(name: str, examples: list[dict]) -> None:
    counts = Counter(e["benchmark"] for e in examples)
    lengths = [len(e["output"].split()) for e in examples]
    s = sorted(lengths)
    n = len(s)
    print(f"\n{name} ({n} examples):")
    for bench, count in sorted(counts.items()):
        print(f"  {bench}: {count}")
    print(f"  Output length — mean: {sum(s)/n:.0f} words, p50: {s[n//2]}, p95: {s[int(n*0.95)]}, max: {s[-1]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, default=Path("data/teacher_traces/traces_correct.jsonl"),
                        help="Correct teacher traces JSONL (output of sanity check filter).")
    parser.add_argument("--output-dir", type=Path, default=Path("data/sft_dataset"))
    parser.add_argument("--val-fraction", type=float, default=0.05,
                        help="Fraction held out for validation.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    raw = load_jsonl(args.traces)
    print(f"Loaded {len(raw)} traces from {args.traces}")

    # Filter to correct traces only (in case raw file isn't pre-filtered)
    correct = [t for t in raw if answers_match(t["teacher_answer"], t["ground_truth"], t["benchmark"])]
    dropped = len(raw) - len(correct)
    if dropped:
        print(f"Dropped {dropped} incorrect traces (teacher answer != ground truth)")

    examples = [format_example(t) for t in correct]
    random.shuffle(examples)

    n_val = max(1, int(len(examples) * args.val_fraction))
    val = examples[:n_val]
    train = examples[n_val:]

    print_dataset_stats("Train", train)
    print_dataset_stats("Val", val)

    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "val.jsonl", val)

    # Also write a combined file for reference
    write_jsonl(args.output_dir / "all.jsonl", examples)

    print(f"\nWrote:")
    print(f"  {args.output_dir}/train.jsonl  ({len(train)} examples)")
    print(f"  {args.output_dir}/val.jsonl    ({len(val)} examples)")
    print(f"  {args.output_dir}/all.jsonl    ({len(examples)} examples)")


if __name__ == "__main__":
    main()
