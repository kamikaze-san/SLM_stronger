#!/usr/bin/env python3
"""Analyze MMLU baseline results by subdomain and category."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


MMLU_CATEGORIES = {
    "formal_logic": "reasoning",
    "abstract_algebra": "reasoning",
    "college_mathematics": "reasoning",
    "college_physics": "reasoning",
    "college_chemistry": "reasoning",
    "elementary_mathematics": "reasoning",
    "high_school_mathematics": "reasoning",
    "high_school_physics": "reasoning",
    "high_school_chemistry": "reasoning",
    "high_school_statistics": "reasoning",
    "econometrics": "reasoning",
    "college_computer_science": "reasoning",
    "high_school_computer_science": "reasoning",
    "machine_learning": "reasoning",
    "clinical_knowledge": "knowledge",
    "medical_genetics": "knowledge",
    "anatomy": "knowledge",
    "professional_medicine": "knowledge",
    "college_medicine": "knowledge",
    "nutrition": "knowledge",
    "virology": "knowledge",
    "professional_law": "knowledge",
    "international_law": "knowledge",
    "jurisprudence": "knowledge",
    "us_foreign_policy": "knowledge",
    "high_school_geography": "knowledge",
    "high_school_government_and_politics": "knowledge",
    "public_relations": "knowledge",
    "prehistory": "knowledge",
    "world_religions": "knowledge",
    "human_sexuality": "knowledge",
    "sociology": "knowledge",
    "high_school_macroeconomics": "knowledge",
    "high_school_microeconomics": "knowledge",
    "high_school_european_history": "knowledge",
    "high_school_us_history": "knowledge",
    "high_school_world_history": "knowledge",
    "global_facts": "knowledge",
    "moral_scenarios": "mixed",
    "moral_philosophy": "mixed",
    "philosophy": "mixed",
    "logical_fallacies": "mixed",
    "college_biology": "mixed",
    "high_school_biology": "mixed",
    "professional_psychology": "mixed",
    "human_aging": "mixed",
    "management": "mixed",
    "marketing": "mixed",
    "business_ethics": "mixed",
    "miscellaneous": "mixed",
    "professional_accounting": "mixed",
    "electrical_engineering": "mixed",
    "conceptual_physics": "mixed",
    "astronomy": "mixed",
    "security_studies": "mixed",
    "computer_security": "mixed",
    "high_school_psychology": "mixed",
}


def load_category_overrides(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        overrides = json.load(f)
    overrides = {key: value for key, value in overrides.items() if not key.startswith("_")}
    allowed = {"reasoning", "knowledge", "mixed"}
    bad = {k: v for k, v in overrides.items() if v not in allowed}
    if bad:
        raise ValueError(f"Invalid category overrides in {path}: {bad}")
    return overrides


def write_override_template(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "_comment": "Override MMLU category priors here. Delete this comment key before use.",
                "example_subdomain": "reasoning",
            },
            f,
            indent=2,
        )
        f.write("\n")


def analyze(input_path: Path, output_dir: Path, overrides_path: Path) -> pd.DataFrame:
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rows = pd.DataFrame(data["per_example"])
    grouped = (
        rows.groupby("subject", as_index=False)
        .agg(n_questions=("correct", "size"), accuracy=("correct", "mean"))
        .rename(columns={"subject": "subdomain"})
    )
    overrides = load_category_overrides(overrides_path)
    categories = {**MMLU_CATEGORIES, **overrides}

    grouped["gap_to_70pct"] = (0.70 - grouped["accuracy"]).clip(lower=0)
    grouped["category"] = grouped["subdomain"].map(categories).fillna("uncategorized")
    grouped["below_65pct"] = grouped["accuracy"] < 0.65
    grouped = grouped.sort_values(["accuracy", "subdomain"], ascending=[True, True])

    csv_path = output_dir / "mmlu_subdomain_analysis.csv"
    png_path = output_dir / "mmlu_subdomain_breakdown.png"
    grouped.to_csv(csv_path, index=False)
    plot_breakdown(grouped, png_path)
    write_override_template(overrides_path)
    return grouped


def plot_breakdown(df: pd.DataFrame, output_path: Path) -> None:
    palette = {
        "reasoning": "#2E86AB",
        "knowledge": "#C44536",
        "mixed": "#5B8E7D",
        "uncategorized": "#777777",
    }
    plt.figure(figsize=(18, 8))
    sns.barplot(data=df, x="subdomain", y="accuracy", hue="category", dodge=False, palette=palette)
    plt.axhline(0.65, color="#444444", linestyle="--", linewidth=1, label="65%")
    plt.axhline(0.70, color="#111111", linestyle=":", linewidth=1, label="70%")
    plt.ylim(0, 1)
    plt.xlabel("MMLU subdomain")
    plt.ylabel("Accuracy")
    plt.title("MMLU Subdomain Breakdown")
    plt.xticks(rotation=80, ha="right", fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/mmlu_baseline_results.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--category-overrides", type=Path, default=Path("mmlu_category_overrides.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = analyze(args.input, args.output_dir, args.category_overrides)
    print(df.to_string(index=False, formatters={"accuracy": "{:.3f}".format, "gap_to_70pct": "{:.3f}".format}))


if __name__ == "__main__":
    main()
