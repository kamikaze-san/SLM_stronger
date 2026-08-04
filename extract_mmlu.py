import json
from datasets import load_dataset

with open(r'C:\Users\NewGr\Downloads\slm\slm\SLM_stronger\results\qwen3-1.7b-opd\mmlu_baseline_results.json', encoding='utf-8') as f:
    opd_data = json.load(f)
with open(r'C:\Users\NewGr\Downloads\slm\slm\SLM_stronger\results\qwen3-1.7b-actual\mmlu_baseline_results.json', encoding='utf-8') as f:
    base_data = json.load(f)

opd = opd_data['per_example']
base = base_data['per_example']
base_by_id = {item['question_id']: item for item in base}

# Load MMLU dataset to get choices
print("Loading MMLU dataset...")
mmlu = load_dataset("cais/mmlu", "all", split="test")
# Build lookup by question text
mmlu_by_q = {}
for row in mmlu:
    mmlu_by_q[row['question']] = row

print(f"MMLU dataset loaded: {len(mmlu_by_q)} questions")

# OPD wrong (with extracted answer), baseline right
cases = []
for item in opd:
    qid = item['question_id']
    if qid in base_by_id:
        b = base_by_id[qid]
        if not item['correct'] and not item['extraction_failed'] and b['correct']:
            cases.append((item, b))

print(f'Found {len(cases)} cases (OPD wrong, baseline right)')
cases.sort(key=lambda x: len(x[0]['question']))

labels = ['A', 'B', 'C', 'D']
count = 0
for item, base_item in cases:
    if count >= 5:
        break
    q_text = item['question']
    if q_text not in mmlu_by_q:
        continue
    row = mmlu_by_q[q_text]
    choices = row['choices']
    correct_letter = labels[row['answer']]

    print()
    print(f'--- [{item["subject"]}] ---')
    print(f'Question: {q_text}')
    for j, c in enumerate(choices):
        print(f'  {labels[j]}. {c}')
    print(f'Correct: {correct_letter}')
    print(f'Baseline: {base_item["extracted_answer"]}  |  OPD: {item["extracted_answer"]}  |  OPD tokens: {item["tokens_generated"]}')
    count += 1
