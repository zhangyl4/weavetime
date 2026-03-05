import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def calc_average_metric(results, save_dir, metric, vmin=None, vmax=None):
    if isinstance(results, list):
        average_metric = sum([item[metric] for item in results]) / len(results) if len(results) > 0 else 0.0
        print(f'#Samples: {len(results)}')
        print(f'Average {metric}: {average_metric:.2f}')

    elif isinstance(results, dict):
        average_metric_map = {}
        for key, value in results.items():
            values = [item[metric] for item in value]
            if len(value) > 0:
                average_metric_map[key] = (sum(values) / len(values))
            else:
                average_metric_map[key] = None

        df = pd.DataFrame.from_dict(average_metric_map, orient='index')
        df.index = pd.MultiIndex.from_tuples(df.index, names=['retrieve_size', 'chunk_size'])
        df = df.reset_index()
        df.columns = ['retrieve_size', 'chunk_size', 'value']
        heatmap_data = df.pivot(index='chunk_size', columns='retrieve_size', values='value')
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".1f",
            cmap="RdPu",
            cbar_kws={'label': 'Value'},
            xticklabels=True,
            yticklabels=True,
            vmin=vmin,
            vmax=vmax,
        )
        ax.invert_yaxis()
        plt.title(f'Heatmap of Average {metric.capitalize()}')
        plt.xlabel('Retrieve Size')
        plt.ylabel('Chunk Size')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{metric}.png'))
        plt.close()

        # Print summary
        first_key = next(iter(results.keys())) if len(results) > 0 else None
        print(f'#Samples: {len(results[first_key]) if first_key is not None else 0}')
        print(average_metric_map)
        # Try to preview image if available (optional)
        os.system(f"imgcat {os.path.join(save_dir, f'{metric}.png')}")
    else:
        raise ValueError(f"Invalid record type: {type(results)}")

    print(f'save_dir: {save_dir}')


def normalize_binary_label(text: str) -> str:
    if text is None:
        return ''
    s = str(text).strip().lower()
    # quick accept exact tokens
    if s in {'yes', 'y', 'true', 't', '1'}:
        return 'yes'
    if s in {'no', 'n', 'false', 'f', '0'}:
        return 'no'
    # heuristic: leading word
    if s.startswith('yes'):
        return 'yes'
    if s.startswith('no'):
        return 'no'
    # strip punctuation and retry
    sp = s.strip(" .,!?:;()[]{}\"'`“”’•-_")
    if sp in {'yes', 'y', 'true', 't', '1'}:
        return 'yes'
    if sp in {'no', 'n', 'false', 'f', '0'}:
        return 'no'
    if sp.startswith('yes'):
        return 'yes'
    if sp.startswith('no'):
        return 'no'
    return s


parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str)
parser.add_argument('--results_path', type=str, default=None)
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

# Load CSV
if args.results_path is not None:
    df = pd.read_csv(args.results_path)
    args.save_dir = os.path.dirname(args.results_path)
else:
    df = pd.read_csv(os.path.join(args.save_dir, 'results.csv'))

# Determine columns
answer_col = 'answer' if 'answer' in df.columns else ('correct_choice' if 'correct_choice' in df.columns else None)
pred_col = 'pred_answer' if 'pred_answer' in df.columns else ('pred_choice' if 'pred_choice' in df.columns else None)
if answer_col is None or pred_col is None:
    raise ValueError(f'Missing required columns. Found: {list(df.columns)}')

# Normalize and compute qa_acc
norm_answers = df[answer_col].apply(normalize_binary_label)
norm_preds = df[pred_col].apply(normalize_binary_label)
df['pred_choice'] = norm_preds
df['qa_acc'] = (norm_preds == norm_answers).astype(float) * 100.0

# Group for heatmap if retrieve_size present
if 'retrieve_size' in df.columns and 'chunk_size' in df.columns:
    results = {}
    for _, row in df.iterrows():
        key = (row['retrieve_size'], row['chunk_size'])
        value = {col: row[col] for col in df.columns if col not in ['retrieve_size', 'chunk_size']}
        if key not in results:
            results[key] = []
        results[key].append(value)
else:
    results = df.to_dict(orient='records')

# Only qa_acc for binary
calc_average_metric(results, args.save_dir, 'qa_acc')

# Optional error rate report
if 'pred_choice' in df.columns and answer_col in df.columns:
    n_errors = int((df['pred_choice'] != norm_answers).sum())
    total = len(df)
    print(f'%Errors: {(n_errors / total * 100.0) if total > 0 else 0.0:.2f}')

