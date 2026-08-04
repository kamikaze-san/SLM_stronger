# SLM_stronger

End-to-end pipeline to make a small language model stronger via:

1. **Baseline evaluation** on GSM8K / MMLU / StrategyQA  
2. **ZPD filtering** (train on what the student gets wrong)  
3. **Teacher trace generation** with self-consistency  
4. **SFT cold start** (LoRA) on teacher traces  
5. **On-Policy Distillation (OPD)** with reverse KL on failed rollouts  
6. **Post-hoc analysis + reporting**

The default setup in this repo targets:

- **Student:** `Qwen/Qwen3-1.7B`
- **Teacher:** `Qwen/Qwen3-14B`

---

## Repository layout

- `eval_baseline.py` — resumable baseline eval for GSM8K, MMLU, StrategyQA  
- `zpd_filter.py` — keeps failed examples (plus MMLU knowledge-subdomain exclusion)  
- `generate_teacher_traces.py` — generates teacher traces with self-consistency filtering  
- `assemble_sft_dataset.py` — builds train/val JSONL for SFT  
- `sft_coldstart.py` — LoRA SFT cold start on teacher traces  
- `opd_train.py` — OPD training loop (vLLM rollout + reverse KL vs teacher)  
- `analyze_mmlu.py` — MMLU subdomain/category analysis + plot  
- `compile_results.py` — compiles baseline outputs into text summary  
- `diagnose.py` — extra diagnostics on failure patterns  
- `inspect_strategyqa.py` — prints StrategyQA split layout  
- `mmlu_category_overrides.json` — optional category override map for MMLU analysis  
- `requirements.txt` — dependencies

---

## Method overview

### 1) Baseline
Evaluate student on benchmarks and store per-example outputs.

### 2) ZPD curation
From train-split eval outputs, keep only questions the student fails (proxy for “learnable next”).

### 3) Teacher trace synthesis
For each ZPD question, sample multiple teacher completions, apply majority self-consistency, keep shortest majority trace, standardize to:

```xml
<think>...</think><answer>...</answer>
```

### 4) SFT cold start
Train student LoRA adapter on teacher traces, masking prompt tokens from loss.

### 5) OPD
Generate student rollouts with vLLM; if a rollout is wrong, apply reverse KL(student||teacher) over generated tokens only.

---

## Environment setup

> Python 3.10+ recommended.  
> CUDA-capable GPU required for training.

Install torch first with matching CUDA build (example from repo comments):

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

Then install remaining deps:

```bash
pip install -r requirements.txt
```

---

## Hardware notes

This repo is designed for high-memory GPU workflows.  
`opd_train.py` includes a rough 48GB memory layout estimate for teacher + student + optimizer + vLLM.

If you hit OOM:
- reduce `--gen-batch-size`
- reduce `--max-new-tokens`
- reduce `--vllm-gpu-mem`
- increase sync interval (`--sync-steps`)

---

## Quickstart (full pipeline)

## 0) (Optional) inspect StrategyQA splits

```bash
python inspect_strategyqa.py
```

## 1) Baseline eval on test splits (report card)

```bash
python eval_baseline.py \
  --model-name Qwen/Qwen3-1.7B \
  --benchmarks gsm8k mmlu strategyqa \
  --output-dir results/qwen3-1.7b-test \
  --batch-size 1 \
  --resume
```

Analyze MMLU and compile summary:

```bash
python analyze_mmlu.py \
  --input results/qwen3-1.7b-test/mmlu_baseline_results.json \
  --output-dir results/qwen3-1.7b-test

python compile_results.py \
  --results-dir results/qwen3-1.7b-test
```

## 2) Baseline eval on training splits (for ZPD mining)

```bash
python eval_baseline.py \
  --model-name Qwen/Qwen3-1.7B \
  --benchmarks gsm8k mmlu strategyqa \
  --gsm8k-split train \
  --mmlu-split auxiliary_train \
  --strategyqa-split train \
  --output-dir results/qwen3-1.7b-train \
  --batch-size 1 \
  --resume
```

## 3) ZPD filter (keep failed examples)

```bash
python zpd_filter.py \
  --results-dir results/qwen3-1.7b-train \
  --output-dir data/zpd_filtered
```

Expected outputs:
- `data/zpd_filtered/gsm8k_zpd.jsonl`
- `data/zpd_filtered/mmlu_zpd.jsonl`
- `data/zpd_filtered/strategyqa_zpd.jsonl`
- `data/zpd_filtered/all_zpd.jsonl`

## 4) Generate teacher traces (self-consistency)

```bash
python generate_teacher_traces.py \
  --teacher Qwen/Qwen3-14B \
  --input data/zpd_filtered/all_zpd.jsonl \
  --output data/teacher_traces/traces.jsonl \
  --n-completions 6 \
  --threshold 4 \
  --temperature 0.7 \
  --batch-size 2
```

## 5) Assemble SFT dataset

```bash
python assemble_sft_dataset.py \
  --traces data/teacher_traces/traces.jsonl \
  --output-dir data/sft_dataset \
  --val-fraction 0.05
```

Outputs:
- `data/sft_dataset/train.jsonl`
- `data/sft_dataset/val.jsonl`
- `data/sft_dataset/all.jsonl`

## 6) SFT cold start (LoRA)

```bash
python sft_coldstart.py \
  --model-name Qwen/Qwen3-1.7B \
  --train-data data/sft_dataset/train.jsonl \
  --val-data data/sft_dataset/val.jsonl \
  --output-dir checkpoints/sft_coldstart \
  --max-steps 500 \
  --batch-size 8 \
  --lr 1e-5
```

Final adapter/tokenizer:
- `checkpoints/sft_coldstart/final/`

## 7) OPD training

```bash
python opd_train.py \
  --student Qwen/Qwen3-1.7B \
  --teacher Qwen/Qwen3-14B \
  --sft-adapter checkpoints/sft_coldstart/final \
  --train-data data/zpd_filtered/all_zpd.jsonl \
  --output-dir checkpoints/opd \
  --max-steps 2000 \
  --gen-batch-size 8 \
  --sync-steps 100 \
  --lr 5e-6
```

Final adapter/tokenizer:
- `checkpoints/opd/final/`

## 8) Evaluate the trained adapter

```bash
python eval_baseline.py \
  --model-name Qwen/Qwen3-1.7B \
  --lora-adapter checkpoints/opd/final \
  --benchmarks gsm8k mmlu strategyqa \
  --output-dir results/qwen3-1.7b-opd-test \
  --batch-size 1 \
  --resume
```

Then re-run:

```bash
python analyze_mmlu.py \
  --input results/qwen3-1.7b-opd-test/mmlu_baseline_results.json \
  --output-dir results/qwen3-1.7b-opd-test

python compile_results.py \
  --results-dir results/qwen3-1.7b-opd-test
```

---

## Input/output schemas

### ZPD JSONL row (`data/zpd_filtered/*.jsonl`)

```json
{
  "question_id": "123",
  "benchmark": "gsm8k|mmlu|strategyqa",
  "question": "...",
  "ground_truth": "...",
  "subject": "..." 
}
```

(`subject` present for MMLU)

### Teacher trace row (`data/teacher_traces/traces.jsonl`)

```json
{
  "question_id": "123",
  "benchmark": "mmlu",
  "subject": "formal_logic",
  "question": "...",
  "ground_truth": "B",
  "teacher_answer": "B",
  "output": "<think>...</think><answer>B</answer>",
  "n_completions": 6,
  "n_agreed": 5
}
```

### SFT dataset row (`data/sft_dataset/train.jsonl`)

```json
{
  "instruction": "Think through this step by step.",
  "input": "...question...",
  "output": "<think>...</think><answer>...</answer>",
  "benchmark": "gsm8k|mmlu|strategyqa",
  "subject": "...",
  "ground_truth": "..."
}
```

---

## Reproducibility / resume behavior

- Most scripts expose `--seed`.
- `eval_baseline.py` writes checkpoint JSONL and can `--resume`.
- `generate_teacher_traces.py` is resumable: skips question ids already in output.
- `opd_train.py` resumes from latest `checkpoints/opd/step_*` if present.

---

## Troubleshooting

- **CUDA OOM during teacher trace generation**
  - lower `--batch-size`
  - lower `--max-new-tokens`
- **CUDA OOM during OPD**
  - lower `--gen-batch-size`
  - lower `--vllm-gpu-mem`
  - lower `--max-new-tokens`
- **Extraction failures are high**
  - inspect prompt format and answer extraction assumptions
  - run diagnostics with `diagnose.py`
- **Missing summary generation**
  - `compile_results.py` requires `mmlu_subdomain_analysis.csv`; run `analyze_mmlu.py` first.

---

## Known caveats (current code)

- `zpd_filter.py` uses `--combined` with `store_true` and `default=True` (effectively always on).
- `opd_train.py` sets `--no-think` as `store_true` with `default=True` (effectively always true).
- `sft_coldstart.py` assumes non-empty train/val files.

---

## License

Apache-2.0 (see `LICENSE`).
