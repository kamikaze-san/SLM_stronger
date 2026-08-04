import json

base = {
    'gsm8k': r'C:\Users\NewGr\Downloads\slm\slm\SLM_stronger\results\qwen3-1.7b-actual\gsm8k_baseline_results.json',
    'mmlu': r'C:\Users\NewGr\Downloads\slm\slm\SLM_stronger\results\qwen3-1.7b-actual\mmlu_baseline_results.json',
    'strategyqa': r'C:\Users\NewGr\Downloads\slm\slm\SLM_stronger\results\qwen3-1.7b-actual\strategyqa_baseline_results.json',
}

# Train set sizes (total questions, not just ZPD)
# ZPD = what student got wrong. Train total = ZPD + what student got right
zpd = {'gsm8k': 1045, 'mmlu': 324, 'strategyqa': 640}
teacher_correct = {'gsm8k': 739, 'mmlu': 92, 'strategyqa': 287}

print("Potential delta if all teacher-correct ZPD traces were perfectly learned:")
print()
for bench in ['gsm8k', 'mmlu', 'strategyqa']:
    with open(base[bench], encoding='utf-8') as f:
        d = json.load(f)
    n_test = d['n_examples']
    n_correct = d['n_correct']
    acc = d['overall_accuracy']

    # ZPD questions = student wrong in train. Assume same wrong rate in test.
    # Fraction of train questions student got wrong = zpd / train_total
    # We don't have train_total directly, but test accuracy ~= train accuracy
    # So fraction wrong in test = 1 - acc
    # Teacher-correct fraction of those wrong = teacher_correct / zpd
    # Potential new correct in test = n_test * (1-acc) * (teacher_correct/zpd)
    frac_wrong = 1.0 - acc
    frac_teacher_can_fix = teacher_correct[bench] / zpd[bench]
    potential_new_correct = n_test * frac_wrong * frac_teacher_can_fix
    delta_acc = potential_new_correct / n_test

    print(f"{bench}:")
    print(f"  Test n={n_test}, baseline acc={acc:.1%}")
    print(f"  ZPD (student wrong in train): {zpd[bench]}")
    print(f"  Teacher correct of those: {teacher_correct[bench]} ({teacher_correct[bench]/zpd[bench]:.1%} of ZPD)")
    print(f"  Potential new correct in test: {potential_new_correct:.0f}")
    print(f"  Potential delta: +{delta_acc:.1%}  -> ceiling acc: {acc+delta_acc:.1%}")
    print()
