import os
import json
import pandas as pd
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str)
parser.add_argument('--pred_path', type=str, default=None)
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

if args.pred_path is not None:
    df = pd.read_csv(args.pred_path)
    args.output_dir = os.path.dirname(args.pred_path)
else:
    df = pd.read_csv(os.path.join(args.output_dir, 'results.csv'))

# Load ground truth data

# Calculate task-specific accuracies
if 'pred_answer' in df.columns:
    task_to_counts = {}
    for _, row in df.iterrows():
        # Find matching ground truth conversation
        video_id = row['video_id']
        question = row['question']
        task = row['task']
        if task not in task_to_counts:
            task_to_counts[task] = {'correct': 0, 'total': 0}
        task_to_counts[task]['total'] += 1

        # Compare prediction with ground truth answer
        
        # Check if pred_answer matches the correct choice letter
        try:
            correct = row['pred_choice'][0] == row['correct_choice'].strip()
        except:
            breakpoint()
            exit()
        if correct:
            task_to_counts[task]['correct'] += 1
        
    # Print overall accuracy
    total_correct = sum(counts['correct'] for counts in task_to_counts.values())
    total_questions = sum(counts['total'] for counts in task_to_counts.values())
    overall_acc = total_correct / total_questions if total_questions > 0 else 0
    task_acc =[]
    # Print task-specific accuracies
    for task, counts in sorted(task_to_counts.items()):
        acc = counts['correct'] / counts['total'] if counts['total'] > 0 else 0
        print(f'{task}: {counts["correct"]}/{counts["total"]}={acc:.4f}')
        task_acc.append(acc)
    
    avg_acc = sum(task_acc) / len(task_acc)
    print(f"avg acc in task: {sum(task_acc)} / {len(task_acc)} = {avg_acc:.4f}")
    print(f"Overall: {total_correct}/{total_questions}={overall_acc:.4f}")
