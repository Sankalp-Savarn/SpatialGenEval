import os
import pdb
import json
import argparse
from tqdm import tqdm

def load_jsonl_lines(jsonl_file):
    lines = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                lines.append(obj)
            except Exception as e:
                print(f"[Warning] Parse line error in {jsonl_file}: {e}")
    return lines

def check_qa_model_preds(answers, all_preds, min_count=3):
    num_questions = len(answers)
    results = []
    selected_options = []
    
    for q_idx in range(num_questions):
        option_count = {}
        valid_preds_for_q = [preds[q_idx] for preds in all_preds if len(preds) > q_idx]
        if not valid_preds_for_q:
            results.append(False)
            selected_options.append(["(No valid predictions)"])
            continue

        for option in valid_preds_for_q:
            option_count[option] = option_count.get(option, 0) + 1
        
        high_freq_options = [opt for opt, count in option_count.items() if count >= min_count]
        is_correct = len(high_freq_options) > 0 and answers[q_idx] in high_freq_options
        results.append(is_correct)
        
        if high_freq_options:
            selected_options.append(high_freq_options)
        else:
            max_count = max(option_count.values())
            most_frequent = [opt for opt, count in option_count.items() if count == max_count]
            selected_options.append(most_frequent)
            
    return results, selected_options

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file', type=str, default='')
    parser.add_argument('--min_count', type=int, default=4)
    args = parser.parse_args()
    
    # read jsonl file
    json_file = args.json_file
    lines = load_jsonl_lines(json_file)

    all_preds = []
    for line in lines:
        answers = line['answers']
        model_preds_list = line['model_preds_cot']
        model_preds_option_list = [[cot[0] for cot in model_preds] for model_preds in model_preds_list]
        yes_or_no_results, selected_options = check_qa_model_preds(answers=answers, all_preds=model_preds_option_list, min_count=args.min_count)
        all_preds.append(yes_or_no_results)

    column_sums = [round(sum(col)/len(all_preds), 3) for col in zip(*all_preds)]
    print(f"{json_file}")
    print(f"====== avg_acc: {sum(column_sums)/10:.3f}, basic_acc: {sum(column_sums[:2])/2:.3f}, spatial_acc: {sum(column_sums[2:])/8:.3f} ======")
    print(list(map(lambda x: f"{x:.3f}", column_sums)))