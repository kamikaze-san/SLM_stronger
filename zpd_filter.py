#!/usr/bin/env python3
"""ZPD filtering: keep questions the student got wrong on training splits.

Reads eval results from the training-split run and outputs filtered JSONL
files ready for teacher trace generation.

Wrong answer = keep (medium difficulty proxy, worth training on).
Right answer = skip (already mastered, no learning signal).

MMLU knowledge subdomains are excluded regardless — not fixable via reasoning training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MMLU_KNOWLEDGE_SUBDOMAINS = {
    "global_facts", "virology", "professional_law", "anatomy",
    "prehistory", "world_religions", "high_school_geography",
    "professional_medicine", "clinical_knowledge", "medical_genetics",
    "nutrition", "international_law", "jurisprudence", "us_foreign_policy",
    "high_school_government_and_politics", "public_relations",
    "high_school_european_history", "high_school_us_history",
    "high_school_world_history", "human_sexuality", "sociology",
    "high_school_macroeconomics", "high_school_microeconomics",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(rows):,} examples → {path}")


def filter_gsm8k(data: dict) -> list[dict]:
    kept = []
    for ex in data["per_example"]:
        if not ex["correct"]:
            kept.append({
                "question_id": ex["question_id"],
                "benchmark": "gsm8k",
                "question": ex["question"],
                "ground_truth": ex["ground_truth"],
            })
    return kept


def filter_mmlu(data: dict) -> list[dict]:
    kept = []
    skipped_knowledge = 0
    for ex in data["per_example"]:
        subject = ex.get("subject", "")
        if subject in MMLU_KNOWLEDGE_SUBDOMAINS:
            skipped_knowledge += 1
            continue
        if not ex["correct"]:
            kept.append({
                "question_id": ex["question_id"],
                "benchmark": "mmlu",
                "subject": subject,
                "question": ex["question"],
                "ground_truth": ex["ground_truth"],
            })
    print(f"  Skipped {skipped_knowledge:,} knowledge subdomain examples")
    return kept


def filter_strategyqa(data: dict) -> list[dict]:
    kept = []
    for ex in data["per_example"]:
        if not ex["correct"]:
            kept.append({
                "question_id": ex["question_id"],
                "benchmark": "strategyqa",
                "question": ex["question"],
                "ground_truth": ex["ground_truth"],
            })
    return kept


def print_stats(name: str, total: int, kept: int) -> None:
    pct = 100 * kept / total if total else 0
    print(f"  {name}: {total:,} total → {kept:,} kept ({pct:.1f}% failed)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/qwen3-1.7b-train"),
                        help="Directory containing training-split eval results.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/zpd_filtered"),
                        help="Where to write filtered JSONL files.")
    parser.add_argument("--combined", action="store_true", default=True,
                        help="Also write a single combined JSONL across all benchmarks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    filters = {
        "gsm8k": filter_gsm8k,
        "mmlu": filter_mmlu,
        "strategyqa": filter_strategyqa,
    }

    all_kept: list[dict] = []

    for benchmark, filter_fn in filters.items():
        path = args.results_dir / f"{benchmark}_baseline_results.json"
        if not path.exists():
            print(f"[SKIP] {benchmark}: {path} not found")
            continue

        print(f"\n{benchmark.upper()}")
        data = load_json(path)
        kept = filter_fn(data)
        total = len(data["per_example"])
        print_stats(benchmark, total, len(kept))

        out_path = args.output_dir / f"{benchmark}_zpd.jsonl"
        write_jsonl(out_path, kept)
        all_kept.extend(kept)

    if args.combined and all_kept:
        combined_path = args.output_dir / "all_zpd.jsonl"
        write_jsonl(combined_path, all_kept)

    print(f"\nTotal ZPD examples across all benchmarks: {len(all_kept):,}")
    print("Ready for teacher trace generation.")


if __name__ == "__main__":
    main()
