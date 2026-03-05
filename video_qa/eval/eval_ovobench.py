import os
import json
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns


import ast
def safe_parse_retrieval_data(value):
    """安全解析retrieval数据"""
    if pd.isna(value) or value == '' or value == '{}':
        return {}
    
    if isinstance(value, dict):
        return value
    
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return {}
    
    return {}

def calc_average_metric(results, save_dir, metric, vmin=None, vmax=None):
    if isinstance(results, list):
        average_metric = sum([item[metric] for item in results]) / len(results)
        print(f'#Samples: {len(results)}')
        print(f'Average {metric}: {average_metric:.2f}')

    elif isinstance(results, dict):
        average_recall = {}
        for key, value in results.items():
            recalls = [item[metric] for item in value]
            if len(value) > 0:
                average_recall[key] = (sum(recalls) / len(recalls))
            else:
                average_recall[key] = None

        df = pd.DataFrame.from_dict(average_recall, orient='index')
        df.index = pd.MultiIndex.from_tuples(df.index, names=['retrieve_size', 'chunk_size'])
        df = df.reset_index()
        df.columns = ['retrieve_size', 'chunk_size', 'value']
        heatmap_data = df.pivot(index='chunk_size', columns='retrieve_size', values='value')
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="RdPu", cbar_kws={'label': 'Value'}, 
                        xticklabels=True, yticklabels=True, vmin=vmin, vmax=vmax)
        ax.invert_yaxis()
        plt.title(f'Heatmap of Average {metric.capitalize()}')
        plt.xlabel('Retrieve Size')
        plt.ylabel('Chunk Size')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{metric}.png'))
        plt.close()

        print(f'#Samples: {len(results[list(results.keys())[0]])}')
        print(average_recall)
        os.system(f"imgcat {os.path.join(save_dir, f'{metric}.png')}")
    else:
        raise ValueError(f"Invalid record type: {type(results)}")

    print(f'save_dir: {save_dir}')


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

# Validate and fix column alignment issues
# Check if choices column contains non-list values (indicates misalignment)
if 'choices' in df.columns and 'task' in df.columns:
    # Check if any choices value looks like a task name (e.g., 'REC', 'EPM', etc.)
    task_names = ['EPM', 'HLD', 'SSR', 'ASI', 'STU', 'OJR', 'ATR', 'ACR', 'FPD', 'OCR', 'CRR', 'REC']
    # Check if choices contains task names (misalignment indicator)
    choices_str = df['choices'].astype(str).str.strip()
    misaligned_mask = choices_str.isin(task_names)
    
    if misaligned_mask.any():
        print(f"Warning: Found {misaligned_mask.sum()} rows with column misalignment. Attempting to fix...")
        
        # For misaligned rows, restore correct column values:
        # choices column contains task value -> move to task column
        if 'task' in df.columns:
            df.loc[misaligned_mask, 'task'] = df.loc[misaligned_mask, 'choices']
        
        # correct_choice column contains retrieve_size value -> move to retrieve_size column
        if 'retrieve_size' in df.columns:
            df.loc[misaligned_mask, 'retrieve_size'] = pd.to_numeric(
                df.loc[misaligned_mask, 'correct_choice'], errors='coerce'
            )
        
        # pred_choice column contains chunk_size value -> move to chunk_size column
        if 'chunk_size' in df.columns:
            df.loc[misaligned_mask, 'chunk_size'] = pd.to_numeric(
                df.loc[misaligned_mask, 'pred_choice'], errors='coerce'
            )
        
        # Clear the misaligned columns (they don't have valid values for these rows)
        df.loc[misaligned_mask, 'choices'] = None
        df.loc[misaligned_mask, 'correct_choice'] = None
        df.loc[misaligned_mask, 'pred_choice'] = None
        df.loc[misaligned_mask, 'qa_acc'] = None
        
        print(f"Fixed {misaligned_mask.sum()} misaligned rows.")

# Load ground truth data


if 'retrieve_size' in df.columns:
    results = {}
    for _, row in df.iterrows():
        key = (row['retrieve_size'], row['chunk_size'])
        value = {col: row[col] for col in df.columns if col not in ['retrieve_size', 'chunk_size']}
        if key not in results:
            results[key] = []
        results[key].append(value)
else:
    results = df.to_dict(orient='records')

if 'recall' in df.columns:
    metrics = ['recall', 'precision', 'f1', 'qa_acc', 'acc_at_gqa']
else:
    metrics = ['qa_acc']

# for metric in metrics:
#     calc_average_metric(results, args.output_dir, metric)

# Calculate task-specific accuracies
if 'pred_answer' in df.columns:
    task_to_counts = {}
    task_to_response_times = {}
    
    total_response_time_s = 0
    total_processing_fps = 0
    total_question = 0 
    
    for _, row in df.iterrows():
        task = row['task']
        if task not in task_to_counts:
            task_to_counts[task] = {'correct': 0, 'total': 0}
            task_to_response_times[task] = []
        task_to_counts[task]['total'] += 1

        try:
            is_future = task in ['CRR', 'REC', 'SSR']
            if is_future:
                correct = row['pred_answer'].replace(')', '').replace('(', '').strip() == row['answer'].strip()
            else:
                correct = row['pred_answer'][0] == row['correct_choice'].strip()
            if correct:
                task_to_counts[task]['correct'] += 1
        except:
            breakpoint()
            exit()
        
        retrieval_info = safe_parse_retrieval_data(row['retrieval_info'])
        response_time = retrieval_info['response_time_s']
        processing_fps = retrieval_info['processing_fps']

        task_to_response_times[task].append(response_time)
        
        total_response_time_s += response_time
        total_processing_fps += processing_fps
        total_question += 1
    
    rt_accs, bt_accs, fr_accs = [], [], []
    
    for task, counts in task_to_counts.items():
        acc = counts["correct"] / counts["total"] if counts["total"] > 0 else 0
        print(f'{task}: {counts["correct"]}/{counts["total"]}={acc}')
        if task in ['OCR', 'ACR', 'ATR', 'STU', 'FPD', 'OJR']:
            rt_accs.append(acc)
        elif task in ['EPM', 'ASI', 'HLD']:
            bt_accs.append(acc)
        else:
            fr_accs.append(acc)

    if rt_accs:
        print(f'Real-Time Visual Perception avg.: {sum(rt_accs)}/{len(rt_accs)}={sum(rt_accs)/len(rt_accs)}')
    if bt_accs:
        print(f'Backward Tracing avg.: {sum(bt_accs)}/{len(bt_accs)}={sum(bt_accs)/len(bt_accs)}')
    if fr_accs:
        print(f'Forward Tracing avg.: {sum(fr_accs)}/{len(fr_accs)}={sum(fr_accs)/len(fr_accs)}')
    
    rt_resp_times, bt_resp_times, fr_resp_times = [], [], []
    for task, resp_times in task_to_response_times.items():
        if task in ['OCR', 'ACR', 'ATR', 'STU', 'FPD', 'OJR']:
            rt_resp_times.extend(resp_times)
        elif task in ['EPM', 'ASI', 'HLD']:
            bt_resp_times.extend(resp_times)
        else:
            fr_resp_times.extend(resp_times)
    
    if rt_resp_times:
        print(f'Real-Time Visual Perception avg. Response Time: {sum(rt_resp_times)/len(rt_resp_times)}s')
    if bt_resp_times:
        print(f'Backward Tracing avg. Response Time: {sum(bt_resp_times)/len(bt_resp_times)}s')
    if fr_resp_times:
        print(f'Forward Tracing avg. Response Time: {sum(fr_resp_times)/len(fr_resp_times)}s')
    
    
    print(f'Average QA Accuracy: {sum(rt_accs+bt_accs+fr_accs)/len(rt_accs+bt_accs+fr_accs)}')
    print(f'Average Response Time: {total_response_time_s/total_question}s')
    print(f'Average Processing FPS: {total_processing_fps/total_question}')