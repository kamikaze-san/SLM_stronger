#!/usr/bin/env python3
"""Run resumable Phi-4-mini baseline evaluations on GSM8K, MMLU, and StrategyQA."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "microsoft/Phi-4-mini-instruct"
SYSTEM_PROMPT = "Think through this step by step."
MAX_NEW_TOKENS = {
    "gsm8k": 512,
    "mmlu": 256,
    "strategyqa": 256,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=json_default)
        f.write("\n")
    tmp_path.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSONL in {path} at line {line_number}: {exc}") from exc
            rows[str(row["question_id"])] = row
    return rows


def normalize_number(text: Any) -> float | None:
    if text is None:
        return None
    match = re.findall(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", str(text))
    if not match:
        return None
    value = match[-1].replace(",", "")
    try:
        return float(value)
    except ValueError:
        return None


def extract_last_number(response: str) -> float | None:
    return normalize_number(response.replace("$", " "))


def extract_gsm8k_ground_truth(answer: str) -> float | None:
    if "####" in answer:
        return normalize_number(answer.split("####")[-1])
    return normalize_number(answer)


def extract_mmlu_answer(response: str) -> str | None:
    patterns = [
        r"(?:answer\s+is|answer:|final\s+answer\s+is|final\s+answer:)\s*\(?\s*([ABCD])\s*\)?",
        r"(?:option|choice)\s*([ABCD])\b",
        r"\b([ABCD])\s*(?:is|seems|appears)\s+correct\b",
        r"\(([ABCD])\)",
        r"\b([ABCD])\)",
    ]
    found: list[str] = []
    for pattern in patterns:
        found.extend(re.findall(pattern, response, flags=re.IGNORECASE))
    standalone = re.findall(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])", response)
    found.extend(standalone)
    return found[-1].upper() if found else None


def extract_yes_no(response: str) -> str | None:
    found = re.findall(r"\b(yes|no)\b", response, flags=re.IGNORECASE)
    return found[-1].lower() if found else None


def mmlu_answer_to_letter(answer: Any) -> str:
    if isinstance(answer, str):
        answer = answer.strip()
        if answer.upper() in {"A", "B", "C", "D"}:
            return answer.upper()
        if answer.isdigit():
            return "ABCD"[int(answer)]
    return "ABCD"[int(answer)]


def normalize_bool_answer(answer: Any) -> str:
    if isinstance(answer, bool):
        return "yes" if answer else "no"
    text = str(answer).strip().lower()
    if text in {"true", "1", "yes"}:
        return "yes"
    if text in {"false", "0", "no"}:
        return "no"
    raise ValueError(f"Cannot normalize StrategyQA answer: {answer!r}")


def build_prompt(tokenizer: Any, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"System: {SYSTEM_PROMPT}\n\nUser: {user_prompt}\n\nAssistant:"


def generate_batch(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    max_new_tokens: int,
) -> list[str]:
    device = next(model.parameters()).device
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    responses: list[str] = []
    input_lengths = inputs["input_ids"].shape[1]
    for output in outputs:
        generated = output[input_lengths:]
        responses.append(tokenizer.decode(generated, skip_special_tokens=True).strip())
    return responses


def make_gsm8k_examples(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset = load_dataset("gsm8k", "main", split="test")
    examples = []
    for idx, row in enumerate(dataset):
        prompt = (
            "Think through this step by step.\n\n"
            f"Question: {row['question']}\n\n"
            "Show your reasoning, then give your final answer as a number."
        )
        examples.append(
            {
                "question_id": str(idx),
                "question": row["question"],
                "prompt": prompt,
                "ground_truth": extract_gsm8k_ground_truth(row["answer"]),
            }
        )
    return examples, {"dataset": "gsm8k", "config": "main", "split": "test", "n_rows": len(dataset)}


def make_mmlu_examples(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset = load_dataset("cais/mmlu", "all", split="test")
    examples = []
    for idx, row in enumerate(dataset):
        choices = list(row["choices"])
        prompt = (
            "Think through this step by step.\n\n"
            f"Question: {row['question']}\n\n"
            "Options:\n"
            f"A) {choices[0]}\n"
            f"B) {choices[1]}\n"
            f"C) {choices[2]}\n"
            f"D) {choices[3]}\n\n"
            "Choose the correct option. Give your reasoning first,\n"
            "then state your final answer as a single letter: A, B, C, or D."
        )
        examples.append(
            {
                "question_id": str(idx),
                "subject": row["subject"],
                "question": row["question"],
                "prompt": prompt,
                "ground_truth": mmlu_answer_to_letter(row["answer"]),
            }
        )
    return examples, {"dataset": "cais/mmlu", "config": "all", "split": "test", "n_rows": len(dataset)}


def choose_strategyqa_split(dataset_dict: DatasetDict, requested_split: str) -> tuple[str, str]:
    split_sizes = {split: len(dataset) for split, dataset in dataset_dict.items()}
    if requested_split != "auto":
        if requested_split not in dataset_dict:
            raise ValueError(f"Requested StrategyQA split {requested_split!r} not found. Available: {split_sizes}")
        return requested_split, f"explicit split requested: {requested_split}"
    near_paper_test = [split for split, size in split_sizes.items() if 450 <= size <= 530]
    if near_paper_test:
        return near_paper_test[0], f"auto-selected explicit paper-sized split: {near_paper_test[0]}"
    if "test" in dataset_dict:
        return "test", "auto-selected test split; no separate ~490-row split exists"
    if len(dataset_dict) == 1:
        split = next(iter(dataset_dict))
        return split, "auto-selected only available split"
    raise ValueError(f"Cannot auto-select StrategyQA split. Available: {split_sizes}")


def make_strategyqa_examples(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_obj = load_dataset("wics/strategy-qa")
    if isinstance(dataset_obj, Dataset):
        dataset_dict = DatasetDict({"test": dataset_obj})
    else:
        dataset_dict = dataset_obj
    split_sizes = {split: len(dataset) for split, dataset in dataset_dict.items()}
    split, split_reason = choose_strategyqa_split(dataset_dict, args.strategyqa_split)
    dataset = dataset_dict[split]
    answer_key = "answer"
    if len(dataset) and answer_key not in dataset.column_names:
        for candidate in ("label", "target"):
            if candidate in dataset.column_names:
                answer_key = candidate
                break
    examples = []
    for idx, row in enumerate(dataset):
        prompt = (
            "Think through this step by step.\n\n"
            f"Question: {row['question']}\n\n"
            "Think about what facts you need to answer this, then answer\n"
            "with only Yes or No."
        )
        examples.append(
            {
                "question_id": str(idx),
                "question": row["question"],
                "prompt": prompt,
                "ground_truth": normalize_bool_answer(row[answer_key]),
            }
        )
    metadata = {
        "dataset": "wics/strategy-qa",
        "available_splits": split_sizes,
        "split": split,
        "split_selection_reason": split_reason,
        "n_rows": len(dataset),
        "contamination_note": (
            "Only the selected split is loaded into evaluation examples. "
            "Use the same selected split for post-training evaluation."
        ),
    }
    return examples, metadata


def evaluate_examples(
    benchmark: str,
    examples: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    output_dir: Path,
    batch_size: int,
    max_new_tokens: int,
    resume: bool,
    save_every: int,
    extractor: Callable[[str], Any],
    comparator: Callable[[Any, Any], bool] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    final_path = output_dir / f"{benchmark}_baseline_results.json"
    checkpoint_path = output_dir / "checkpoints" / f"{benchmark}_baseline_results.jsonl"
    completed = load_jsonl_by_id(checkpoint_path) if resume else {}
    rows: list[dict[str, Any]] = []

    pending = [ex for ex in examples if str(ex["question_id"]) not in completed]
    progress = tqdm(total=len(examples), initial=len(completed), desc=benchmark)

    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        prompts = [build_prompt(tokenizer, ex["prompt"]) for ex in batch]
        responses = generate_batch(model, tokenizer, prompts, max_new_tokens=max_new_tokens)
        for ex, response in zip(batch, responses):
            extracted = extractor(response)
            extraction_failed = extracted is None
            if extraction_failed:
                correct = False
            elif comparator is not None:
                correct = comparator(extracted, ex["ground_truth"])
            else:
                correct = extracted == ex["ground_truth"]
            row = {
                "question_id": ex["question_id"],
                "question": ex["question"],
                "response": response,
                "extracted_answer": extracted,
                "ground_truth": ex["ground_truth"],
                "correct": correct,
                "extraction_failed": extraction_failed,
            }
            if "subject" in ex:
                row["subject"] = ex["subject"]
            append_jsonl(checkpoint_path, row)
            completed[str(ex["question_id"])] = row
            progress.update(1)

        if len(completed) % save_every < batch_size:
            aggregate_and_save(benchmark, completed, final_path, metadata=metadata)

    progress.close()
    return aggregate_and_save(benchmark, completed, final_path, metadata=metadata)


def aggregate_and_save(
    benchmark: str,
    completed: dict[str, dict[str, Any]],
    final_path: Path,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    per_example = [completed[key] for key in sorted(completed, key=lambda x: int(x))]
    n = len(per_example)
    n_correct = sum(bool(row["correct"]) for row in per_example)
    n_failed = sum(bool(row.get("extraction_failed", False)) for row in per_example)
    result: dict[str, Any] = {
        "benchmark": benchmark,
        "overall_accuracy": n_correct / n if n else math.nan,
        "extraction_failure_rate": n_failed / n if n else math.nan,
        "n_examples": n,
        "n_correct": n_correct,
        "n_extraction_failed": n_failed,
        "metadata": metadata or {},
        "per_example": per_example,
    }
    if benchmark == "mmlu":
        by_subject: dict[str, list[dict[str, Any]]] = {}
        for row in per_example:
            by_subject.setdefault(row["subject"], []).append(row)
        result["per_subdomain_accuracy"] = {
            subject: sum(bool(row["correct"]) for row in rows) / len(rows)
            for subject, rows in sorted(by_subject.items())
        }
    atomic_write_json(final_path, result)
    return result


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    kwargs: dict[str, Any] = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **kwargs)
    model.eval()
    return model, tokenizer


def numeric_equal(left: Any, right: Any) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-6)
    except (TypeError, ValueError):
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["gsm8k", "mmlu", "strategyqa"],
        choices=["gsm8k", "mmlu", "strategyqa"],
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn-implementation", default=None, help="Optional override: eager, sdpa, flash_attention_2")
    parser.add_argument("--strategyqa-split", default="auto", help="StrategyQA split name, or auto.")
    parser.add_argument("--gsm8k-max-new-tokens", type=int, default=MAX_NEW_TOKENS["gsm8k"])
    parser.add_argument("--mmlu-max-new-tokens", type=int, default=MAX_NEW_TOKENS["mmlu"])
    parser.add_argument("--strategyqa-max-new-tokens", type=int, default=MAX_NEW_TOKENS["strategyqa"])
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test limit per benchmark.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(args)
    makers = {
        "gsm8k": make_gsm8k_examples,
        "mmlu": make_mmlu_examples,
        "strategyqa": make_strategyqa_examples,
    }
    extractors = {
        "gsm8k": extract_last_number,
        "mmlu": extract_mmlu_answer,
        "strategyqa": extract_yes_no,
    }
    token_limits = {
        "gsm8k": args.gsm8k_max_new_tokens,
        "mmlu": args.mmlu_max_new_tokens,
        "strategyqa": args.strategyqa_max_new_tokens,
    }
    comparators = {
        "gsm8k": numeric_equal,
        "mmlu": None,
        "strategyqa": None,
    }

    for benchmark in args.benchmarks:
        examples, metadata = makers[benchmark](args)
        if args.limit is not None:
            examples = examples[: args.limit]
            metadata = {**metadata, "limit": args.limit, "limited_n_rows": len(examples)}
        evaluate_examples(
            benchmark=benchmark,
            examples=examples,
            model=model,
            tokenizer=tokenizer,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            max_new_tokens=token_limits[benchmark],
            resume=args.resume,
            save_every=args.save_every,
            extractor=extractors[benchmark],
            comparator=comparators[benchmark],
            metadata=metadata,
        )


if __name__ == "__main__":
    main()
