# SLM_stronger

End-to-end training and analysis pipeline for improving a small language model on
GSM8K, MMLU, and StrategyQA with:

1. Baseline evaluation
2. ZPD filtering (train on failed examples)
3. Teacher trace generation + SFT cold start
4. Multiple post-SFT optimization variants (OPD / GRPO / Rewind-GRPO / RFT)
5. Transition and failure analysis

Default model pair used in this repo:
- **Student:** `Qwen/Qwen3-1.7B`
- **Teacher:** `Qwen/Qwen3-14B` (and `Qwen/Qwen3-8B` for teacher-localization probing)

---

## Repository layout

### Core pipeline
- `eval_baseline.py` — resumable eval on GSM8K/MMLU/StrategyQA (+ optional LoRA merge in memory)
- `zpd_filter.py` — keeps failed training-split questions, writes benchmark JSONL + combined JSONL
- `generate_teacher_traces.py` — teacher self-consistency trace generation
- `assemble_sft_dataset.py` — builds train/val/all SFT JSONL from traces
- `sft_coldstart.py` — LoRA SFT initialization on teacher traces

### Optimization trainers (post-SFT or base)
- `opd_train.py` — reverse-KL on incorrect student rollouts (teacher-guided)
- `grpo_train.py` — group-relative policy optimization on student rollouts
- `rewind_train.py` — GRPO variant that rescues dead GSM8K groups via partial-solution rewind
- `rft_train.py` — rejection-sampling FT on correct student rollouts only (teacher-free)

### Analysis and diagnostics
- `analyze_mmlu.py` — subdomain/category breakdown + CSV/plot export
- `compile_results.py` — text summary from benchmark result files
- `analyze_transitions.py` — learned/forgotten per-question transitions between two runs
- `probe_rewind.py` — feasibility probe for rewind behavior (no training)
- `teacher_localize_probe.py` — compares teacher-localized rewind cuts vs fixed cut
- `diagnose.py` — quick baseline diagnostics
- `inspect_strategyqa.py` — prints StrategyQA split availability

### Legacy/ad-hoc scripts
- `check_gsm8k.py`, `check_progressions.py`, `check_regressions.py`, `extract_mmlu.py`
  (one-off local analysis scripts with hard-coded paths)

---

## Environment setup

Python 3.10+ and CUDA GPU recommended.

Install PyTorch CUDA build first, then requirements:

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

---

## Data flow

1. `eval_baseline.py` on training splits
2. `zpd_filter.py` creates `data/zpd_filtered/*.jsonl`
3. `generate_teacher_traces.py` creates `data/teacher_traces/traces.jsonl`
4. `assemble_sft_dataset.py` creates `data/sft_dataset/{train,val,all}.jsonl`
5. `sft_coldstart.py` creates `checkpoints/sft_coldstart/final`
6. Run one or more trainers (`opd_train.py`, `grpo_train.py`, `rewind_train.py`, `rft_train.py`)
7. Re-evaluate with `eval_baseline.py --lora-adapter ...`
8. Analyze with `analyze_mmlu.py`, `compile_results.py`, `analyze_transitions.py`

---

## Quickstart

### 1) Baseline eval (test report card)

```bash
python eval_baseline.py \
  --model-name Qwen/Qwen3-1.7B \
  --benchmarks gsm8k mmlu strategyqa \
  --output-dir results/qwen3-1.7b-test \
  --batch-size 1 \
  --resume
```

### 2) Baseline eval on training splits (for ZPD)

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

### 3) ZPD filtering

```bash
python zpd_filter.py \
  --results-dir results/qwen3-1.7b-train \
  --output-dir data/zpd_filtered
```

### 4) Teacher traces

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

### 5) Assemble SFT dataset

```bash
python assemble_sft_dataset.py \
  --traces data/teacher_traces/traces.jsonl \
  --output-dir data/sft_dataset \
  --val-fraction 0.05
```

### 6) SFT cold start

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

### 7) Choose a trainer

#### OPD (teacher-guided)

```bash
python opd_train.py \
  --student Qwen/Qwen3-1.7B \
  --teacher Qwen/Qwen3-14B \
  --sft-adapter checkpoints/sft_coldstart/final \
  --train-data data/zpd_filtered/all_zpd.jsonl \
  --output-dir checkpoints/opd
```

#### GRPO (teacher-free)

```bash
python grpo_train.py \
  --student Qwen/Qwen3-1.7B \
  --sft-adapter checkpoints/sft_coldstart/final \
  --train-data data/zpd_filtered/all_zpd.jsonl \
  --output-dir checkpoints/grpo
```

#### Rewind-GRPO (dead-group rescue)

```bash
python rewind_train.py \
  --student Qwen/Qwen3-1.7B \
  --sft-adapter checkpoints/sft_coldstart/final \
  --train-data data/zpd_filtered/all_zpd.jsonl \
  --output-dir checkpoints/rewind
```

#### RFT (correct-rollout imitation)

```bash
python rft_train.py \
  --student Qwen/Qwen3-1.7B \
  --sft-adapter checkpoints/sft_coldstart/final \
  --train-data data/zpd_filtered/all_zpd.jsonl \
  --output-dir checkpoints/rft
```

### 8) Evaluate an adapter

```bash
python eval_baseline.py \
  --model-name Qwen/Qwen3-1.7B \
  --lora-adapter checkpoints/opd/final \
  --benchmarks gsm8k mmlu strategyqa \
  --output-dir results/qwen3-1.7b-opd-test \
  --resume
```

### 9) Run analysis

```bash
python analyze_mmlu.py --input results/qwen3-1.7b-opd-test/mmlu_baseline_results.json --output-dir results/qwen3-1.7b-opd-test
python compile_results.py --results-dir results/qwen3-1.7b-opd-test
python analyze_transitions.py --base results/qwen3-1.7b-test --new results/qwen3-1.7b-opd-test --label OPD
```

---

## Rewind probing utilities

```bash
python probe_rewind.py --model checkpoints/grpo1000_kl_merged --n-questions 100 --probes 6 --n-continuations 8
python teacher_localize_probe.py --n-questions 100
```

---

## Notes and caveats

- `eval_baseline.py` default `--model-name` is `microsoft/Phi-4-mini-instruct`; pass Qwen explicitly if you want the repo default setup.
- `zpd_filter.py --combined` is `store_true` with default `True` (effectively always enabled).
- `grpo_train.py`, `rewind_train.py`, `rft_train.py`, and `opd_train.py` use `--no-think` as `store_true` with default `True`.
- `grpo_train.py` excludes MMLU by default because filtered items do not retain answer choices.

---

## License

Apache-2.0 (see `LICENSE`).
