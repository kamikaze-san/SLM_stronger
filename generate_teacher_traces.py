#!/usr/bin/env python3
"""Generate teacher traces for SFT cold start using Qwen3-14B.

For each ZPD-filtered question:
  1. Generate N completions at temperature=0.7
  2. Self-consistency filter: keep if >=threshold completions agree on final answer
  3. Keep shortest trace from majority
  4. Output in <think>...</think><answer>...</answer> format

Resumable: skips questions already in output JSONL.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = "Think through this step by step."


def set_seed(seed: int) -> None:
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def truncate_at_stop_token(text: str) -> str:
    for marker in ("<|end|>", "<|endoftext|>", "<|im_end|>", "<|user|>", "<|assistant|>", "</s>", "[/INST]"):
        if marker in text:
            text = text[: text.index(marker)]
    return text.strip()


def strip_think_block(text: str) -> str:
    m = re.search(r"</think>(.*)", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else text


def extract_answer_line(text: str) -> str | None:
    m = re.search(r"(?:^|\n)\s*Answer:\s*(.+)", text, re.IGNORECASE)
    return m.group(1).strip() if m else None


def extract_final_answer(response: str, benchmark: str) -> str | None:
    text = strip_think_block(response)
    line = extract_answer_line(text)

    if benchmark == "gsm8k":
        src = line or text
        nums = re.findall(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", src.replace("$", " "))
        return nums[-1].replace(",", "") if nums else None

    elif benchmark == "mmlu":
        src = line or text
        m = re.search(r"\b([ABCD])\b", src, re.IGNORECASE)
        return m.group(1).upper() if m else None

    elif benchmark == "strategyqa":
        src = line or text
        m = re.search(r"\b(yes|no)\b", src, re.IGNORECASE)
        return m.group(1).lower() if m else None

    return None


def build_prompt(tokenizer: Any, question: str, no_think: bool) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    if getattr(tokenizer, "chat_template", None):
        kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        if no_think:
            try:
                return tokenizer.apply_chat_template(messages, **kwargs, enable_thinking=False)
            except TypeError:
                pass
        return tokenizer.apply_chat_template(messages, **kwargs)
    return f"System: {SYSTEM_PROMPT}\n\nUser: {question}\n\nAssistant:"


def generate_completions(
    model: Any,
    tokenizer: Any,
    prompt: str,
    n: int,
    max_new_tokens: int,
    temperature: float,
) -> list[str]:
    device = next(model.parameters()).device
    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    eos_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id else []
    for token in ("<|end|>", "<|endoftext|>", "<|im_end|>"):
        tid = tokenizer.convert_tokens_to_ids(token)
        if tid is not None and tid != tokenizer.unk_token_id and tid not in eos_ids:
            eos_ids.append(tid)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            num_return_sequences=n,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=eos_ids,
        )

    responses = []
    for output in outputs:
        generated = output[input_len:]
        raw = tokenizer.decode(generated, skip_special_tokens=False)
        responses.append(truncate_at_stop_token(raw))
    return responses


def self_consistency_filter(
    responses: list[str],
    benchmark: str,
    threshold: int,
) -> tuple[str | None, str | None]:
    """Return (majority_answer, shortest_trace) or (None, None) if no majority."""
    answers = [extract_final_answer(r, benchmark) for r in responses]
    valid = [(a, r) for a, r in zip(answers, responses) if a is not None]
    if not valid:
        return None, None

    counts: dict[str, list[str]] = {}
    for ans, resp in valid:
        counts.setdefault(ans, []).append(resp)

    majority_ans = max(counts, key=lambda a: len(counts[a]))
    majority_traces = counts[majority_ans]

    if len(majority_traces) < threshold:
        return None, None

    shortest = min(majority_traces, key=len)
    return majority_ans, shortest


def format_output(trace: str, answer: str) -> str:
    """Wrap in <think>...</think><answer>...</answer> format."""
    think_match = re.search(r"<think>(.*?)</think>", trace, re.DOTALL | re.IGNORECASE)
    if think_match:
        think_content = think_match.group(1).strip()
    else:
        think_content = trace.strip()
    return f"<think>{think_content}</think><answer>{answer}</answer>"


def load_model_and_tokenizer(model_name: str, no_think: bool) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    return model, tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", default="Qwen/Qwen3-14B",
                        help="Teacher model to use for trace generation.")
    parser.add_argument("--input", type=Path, default=Path("data/zpd_filtered/all_zpd.jsonl"),
                        help="ZPD-filtered questions JSONL.")
    parser.add_argument("--output", type=Path, default=Path("data/teacher_traces/traces.jsonl"),
                        help="Output JSONL with teacher traces.")
    parser.add_argument("--n-completions", type=int, default=6,
                        help="Completions per question for self-consistency.")
    parser.add_argument("--threshold", type=int, default=4,
                        help="Min completions agreeing for self-consistency pass.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--no-think", action="store_true", default=False,
                        help="Disable thinking mode (for Qwen3 teacher).")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    questions = load_jsonl(args.input)
    if not questions:
        raise FileNotFoundError(f"No questions found at {args.input}")

    done_ids = {row["question_id"] + "_" + row["benchmark"]
                for row in load_jsonl(args.output)}
    pending = [q for q in questions
               if q["question_id"] + "_" + q["benchmark"] not in done_ids]

    print(f"Total questions: {len(questions):,}")
    print(f"Already done:    {len(done_ids):,}")
    print(f"Pending:         {len(pending):,}")

    if not pending:
        print("All done.")
        return

    print(f"\nLoading teacher: {args.teacher}")
    model, tokenizer = load_model_and_tokenizer(args.teacher, args.no_think)
    print("Teacher loaded and frozen.")

    stats = {"passed": 0, "failed_consistency": 0, "no_answer": 0}
    progress = tqdm(pending, desc="teacher traces")

    for q in progress:
        prompt = build_prompt(tokenizer, q["question"], args.no_think)
        responses = generate_completions(
            model, tokenizer, prompt,
            n=args.n_completions,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        majority_ans, shortest_trace = self_consistency_filter(
            responses, q["benchmark"], args.threshold
        )

        if majority_ans is None:
            stats["failed_consistency"] += 1
            continue

        formatted = format_output(shortest_trace, majority_ans)
        row = {
            "question_id": q["question_id"],
            "benchmark": q["benchmark"],
            "subject": q.get("subject", ""),
            "question": q["question"],
            "ground_truth": q["ground_truth"],
            "teacher_answer": majority_ans,
            "output": formatted,
            "n_completions": args.n_completions,
            "n_agreed": sum(
                1 for r in responses
                if extract_final_answer(r, q["benchmark"]) == majority_ans
            ),
        }
        append_jsonl(args.output, row)
        stats["passed"] += 1

        progress.set_postfix(
            passed=stats["passed"],
            failed=stats["failed_consistency"],
        )

    total = len(pending)
    print(f"\n=== TEACHER TRACE GENERATION COMPLETE ===")
    print(f"Passed self-consistency: {stats['passed']:,} / {total:,} ({100*stats['passed']/total:.1f}%)")
    print(f"Failed self-consistency: {stats['failed_consistency']:,}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
