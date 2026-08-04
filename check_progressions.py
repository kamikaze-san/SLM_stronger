import json

def load_results(path):
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    return {e['question_id']: e for e in d['per_example']}

base_dir = r'C:\Users\NewGr\Downloads\slm\slm\SLM_stronger\results\qwen3-1.7b-actual'
sft_dir  = r'C:\Users\NewGr\Downloads\slm\slm\SLM_stronger\results\qwen3-1.7b-sft'
opd_dir  = r'C:\Users\NewGr\Downloads\slm\slm\SLM_stronger\results\qwen3-1.7b-opd'

def show_progressions(bench, a_results, b_results, a_label, b_label):
    progressions = []
    for qid in a_results:
        if qid not in b_results:
            continue
        if not a_results[qid]['correct'] and b_results[qid]['correct']:
            progressions.append(a_results[qid])

    print(f'=== {bench}: {len(progressions)} progressions ({a_label} wrong, {b_label} right) ===')
    for e in progressions[:5]:
        qid = e['question_id']
        print(f'  Q: {e["question"][:100]}')
        print(f'  GT: {e["ground_truth"]}  {a_label}: {e["extracted_answer"]}  {b_label}: {b_results[qid]["extracted_answer"]}')
        print()

for bench in ['gsm8k', 'mmlu', 'strategyqa']:
    base = load_results(f'{base_dir}\\{bench}_baseline_results.json')
    sft  = load_results(f'{sft_dir}\\{bench}_baseline_results.json')
    opd  = load_results(f'{opd_dir}\\{bench}_baseline_results.json')

    show_progressions(bench, base, sft, 'base', 'SFT')
    show_progressions(bench, base, opd, 'base', 'OPD')
    show_progressions(bench, sft,  opd, 'SFT',  'OPD')
    print()
