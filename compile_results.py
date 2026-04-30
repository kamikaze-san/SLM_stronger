#!/usr/bin/env python3
"""Compile baseline benchmark outputs into a plain-text summary report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_model_name(results_dir: Path) -> str:
    for fname in ("gsm8k_baseline_results.json", "mmlu_baseline_results.json", "strategyqa_baseline_results.json"):
        path = results_dir / fname
        if path.exists():
            data = load_json(path)
            name = data.get("metadata", {}).get("model_name")
            if name:
                return name
    return results_dir.name


def compile_report(results_dir: Path, model_name: str | None = None) -> str:
    gsm8k = load_json(results_dir / "gsm8k_baseline_results.json")
    mmlu = load_json(results_dir / "mmlu_baseline_results.json")
    strategyqa = load_json(results_dir / "strategyqa_baseline_results.json")
    strategyqa_meta = strategyqa.get("metadata", {})
    analysis_path = results_dir / "mmlu_subdomain_analysis.csv"
    if not analysis_path.exists():
        raise FileNotFoundError(f"Missing {analysis_path}. Run analyze_mmlu.py first.")
    mmlu_analysis = pd.read_csv(analysis_path)

    if model_name is None:
        model_name = infer_model_name(results_dir)

    bottom = mmlu_analysis.sort_values("accuracy").head(10)
    top = mmlu_analysis.sort_values("accuracy", ascending=False).head(10)
    low = mmlu_analysis[mmlu_analysis["accuracy"] < 0.65]
    reasoning_low = low[low["category"] == "reasoning"]["subdomain"].tolist()
    knowledge_low = low[low["category"] == "knowledge"]["subdomain"].tolist()

    lines = [
        f"=== {model_name.upper()} BASELINE RESULTS ===",
        "",
        f"GSM8K Test Accuracy:      {pct(gsm8k['overall_accuracy']):>6}   (target: >88.6% pre-training, need +5% post)",
        f"MMLU Test Accuracy:       {pct(mmlu['overall_accuracy']):>6}   (target: >67.3% pre-training, need +5% post)",
        f"StrategyQA Test Accuracy: {pct(strategyqa['overall_accuracy']):>6}   (target: unknown, need +5% post)",
        f"StrategyQA Eval Split:     {strategyqa_meta.get('split', 'unknown')}   "
        f"(available: {strategyqa_meta.get('available_splits', 'unknown')})",
        "",
        "=== MMLU SUBDOMAIN BREAKDOWN ===",
        "Bottom 10 subdomains (most room for improvement):",
    ]
    for idx, row in enumerate(bottom.itertuples(index=False), start=1):
        lines.append(f"{idx}. {row.subdomain}: {pct(row.accuracy)}")

    lines.extend(["", "Top 10 subdomains (already strong):"])
    for idx, row in enumerate(top.itertuples(index=False), start=1):
        lines.append(f"{idx}. {row.subdomain}: {pct(row.accuracy)}")

    lines.extend(
        [
            "",
            f"REASONING subdomains below 65%: {reasoning_low}",
            f"KNOWLEDGE subdomains below 65%: {knowledge_low}",
            "",
            "=== WHAT THIS MEANS FOR TRAINING ===",
            f"Primary MMLU training target subdomains: {reasoning_low}",
            f"Expected to be hard to improve via RL: {knowledge_low}",
            "",
            "=== EXTRACTION FAILURE RATES ===",
            f"GSM8K:      {pct(gsm8k['extraction_failure_rate'])}",
            f"MMLU:       {pct(mmlu['extraction_failure_rate'])}",
            f"StrategyQA: {pct(strategyqa['extraction_failure_rate'])}",
            "(High failure rate = prompting issue, fix before training)",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--model-name", default=None, help="Override model name shown in report header.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.results_dir / "baseline_summary.txt"
    report = compile_report(args.results_dir, model_name=args.model_name)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
