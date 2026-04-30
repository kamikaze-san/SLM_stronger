#!/usr/bin/env python3
"""Diagnostic analysis of baseline results."""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path("results")

random.seed(42)


def load(name: str) -> list[dict]:
    path = RESULTS_DIR / f"{name}_baseline_results.json"
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data["per_example"]


def mean(vals: list) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def count_steps(text: str) -> int:
    return len(re.findall(r"step \d+|^\d+\.", text.lower(), re.MULTILINE))


# ─── GSM8K ────────────────────────────────────────────────────────────────────

print("=" * 60)
print("GSM8K")
print("=" * 60)

gsm = load("gsm8k")
failed = [x for x in gsm if not x["correct"]]
correct = [x for x in gsm if x["correct"]]

failed_len = [len(x["response"].split()) for x in failed]
correct_len = [len(x["response"].split()) for x in correct]
print(f"Failed  — mean response length: {mean(failed_len):.0f} words")
print(f"Correct — mean response length: {mean(correct_len):.0f} words")

failed_steps = [count_steps(x["response"]) for x in failed]
correct_steps = [count_steps(x["response"]) for x in correct]
print(f"Failed  — mean reasoning steps: {mean(failed_steps):.1f}")
print(f"Correct — mean reasoning steps: {mean(correct_steps):.1f}")

extraction_failed = [x for x in failed if x.get("extracted_answer") is None]
wrong_answer = [x for x in failed if x.get("extracted_answer") is not None]
print(f"Failed due to extraction error: {len(extraction_failed)}")
print(f"Failed due to wrong answer:     {len(wrong_answer)}")

print("\n--- Sample failed GSM8K ---")
for x in random.sample(failed, min(10, len(failed))):
    print(f"Q: {x['question'][:100]}...")
    print(f"Response: {x['response'][:200]}...")
    print(f"Expected: {x['ground_truth']}, Got: {x['extracted_answer']}")
    print()


# ─── MMLU ─────────────────────────────────────────────────────────────────────

print("=" * 60)
print("MMLU")
print("=" * 60)

mmlu = load("mmlu")
by_subject: dict[str, list] = defaultdict(list)
for x in mmlu:
    by_subject[x["subject"]].append(x)

reasoning_subdomains = [
    "abstract_algebra", "college_mathematics", "formal_logic",
    "econometrics", "machine_learning", "college_physics",
    "college_chemistry", "college_computer_science",
    "high_school_mathematics", "high_school_physics",
]

print("--- Reasoning subdomain breakdown ---")
total_reasoning_failed = 0
for sub in reasoning_subdomains:
    examples = by_subject.get(sub, [])
    if not examples:
        continue
    sub_failed = [x for x in examples if not x["correct"]]
    total_reasoning_failed += len(sub_failed)
    print(f"  {sub}: {len(examples)} total, {len(sub_failed)} failed ({len(sub_failed)/len(examples)*100:.1f}%)")

overall_acc = sum(1 for x in mmlu if x["correct"]) / len(mmlu) * 100
projected = overall_acc + (total_reasoning_failed * 0.3 / (len(mmlu) / 100))
print(f"\nTotal reasoning failures: {total_reasoning_failed}")
print(f"If reasoning subdomains hit 80%: MMLU ~{projected:.1f}%")

failed_all = [x for x in mmlu if not x["correct"]]
correct_all = [x for x in mmlu if x["correct"]]
print(f"\nFailed  — mean response: {mean([len(x['response'].split()) for x in failed_all]):.0f} words")
print(f"Correct — mean response: {mean([len(x['response'].split()) for x in correct_all]):.0f} words")

# Correct answer mentioned in reasoning but wrong letter extracted
def answer_in_response(response: str, correct_answer: str) -> bool:
    think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    text = think_match.group(1) if think_match else response
    return correct_answer.lower() in text.lower()

mentioned_but_wrong = [
    x for x in failed_all if answer_in_response(x["response"], x["ground_truth"])
]
print(f"\nFailed but correct answer appeared in reasoning: {len(mentioned_but_wrong)}")
print(f"  → Free recovery potential: {len(mentioned_but_wrong)/len(mmlu)*100:.1f}% MMLU accuracy")

print("\n--- Sample failures by reasoning subdomain ---")
for sub in ["abstract_algebra", "formal_logic", "econometrics"]:
    sub_failed = [x for x in by_subject.get(sub, []) if not x["correct"]]
    if sub_failed:
        x = random.choice(sub_failed)
        print(f"\n{sub}:")
        print(f"  Q: {x['question'][:150]}...")
        print(f"  Response: {x['response'][:300]}...")
        print(f"  Expected: {x['ground_truth']}, Got: {x['extracted_answer']}")


# ─── StrategyQA ───────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("StrategyQA")
print("=" * 60)

sqa = load("strategyqa")
failed_sqa = [x for x in sqa if not x["correct"]]
correct_sqa = [x for x in sqa if x["correct"]]

gt_yes = sum(1 for x in sqa if str(x["ground_truth"]).lower() in ("yes", "true", "1"))
gt_no = len(sqa) - gt_yes
model_yes = sum(1 for x in sqa if str(x.get("extracted_answer", "")).lower() == "yes")
model_no = sum(1 for x in sqa if str(x.get("extracted_answer", "")).lower() == "no")

print(f"Ground truth: {gt_yes} Yes, {gt_no} No")
print(f"Model output: {model_yes} Yes, {model_no} No")
print(f"Model yes rate: {model_yes/len(sqa)*100:.1f}%  (ground truth yes rate: {gt_yes/len(sqa)*100:.1f}%)")

yes_qs = [x for x in sqa if str(x["ground_truth"]).lower() in ("yes", "true", "1")]
no_qs  = [x for x in sqa if str(x["ground_truth"]).lower() in ("no", "false", "0")]
yes_acc = sum(1 for x in yes_qs if x["correct"]) / len(yes_qs) if yes_qs else 0
no_acc  = sum(1 for x in no_qs  if x["correct"]) / len(no_qs)  if no_qs  else 0
print(f"Accuracy on Yes questions: {yes_acc*100:.1f}%")
print(f"Accuracy on No  questions: {no_acc*100:.1f}%")

print(f"\nFailed  — mean response: {mean([len(x['response'].split()) for x in failed_sqa]):.0f} words")
print(f"Correct — mean response: {mean([len(x['response'].split()) for x in correct_sqa]):.0f} words")

print("\n--- Sample failed StrategyQA ---")
for x in random.sample(failed_sqa, min(5, len(failed_sqa))):
    print(f"Q: {x['question']}")
    print(f"Response: {x['response'][:400]}...")
    print(f"Expected: {x['ground_truth']}, Got: {x['extracted_answer']}")
    print()


# ─── Cross-benchmark: response length vs accuracy ─────────────────────────────

print("=" * 60)
print("CROSS-BENCHMARK: response length vs accuracy")
print("=" * 60)

for name, data in [("gsm8k", gsm), ("mmlu", mmlu), ("strategyqa", sqa)]:
    lengths = [len(x["response"].split()) for x in data]
    median_len = sorted(lengths)[len(lengths) // 2]
    short = [x for x in data if len(x["response"].split()) <= median_len]
    long  = [x for x in data if len(x["response"].split()) >  median_len]
    short_acc = sum(1 for x in short if x["correct"]) / len(short) * 100
    long_acc  = sum(1 for x in long  if x["correct"]) / len(long)  * 100
    print(f"{name}: median={median_len}w  short_acc={short_acc:.1f}%  long_acc={long_acc:.1f}%")
