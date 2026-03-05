import json


if __name__ == '__main__':
    import csv
    import ast
    import argparse
    import os
    import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='/2022233235/.cache/huggingface/hub/datasets--mjuicem--StreamingBench/snapshots/48872fa707124474ce4c5172ddc58efb8bc88058/StreamingBench/Real_Time_Visual_Understanding.csv', help='Input CSV file path')
    parser.add_argument('--jsonl', type=str, default='streamingbench-rtvu-formatted.jsonl', help='Output JSONL file path')
    args = parser.parse_args()

    with open(args.csv, 'r', encoding='utf-8') as fin, open(args.jsonl, 'w', encoding='utf-8') as fout:
        reader = csv.DictReader(fin)
        for row in tqdm.tqdm(reader, desc='Processing CSV'):
            # 解析 options 字段为列表
            if 'options' in row:
                try:
                    options_list = ast.literal_eval(row['options'])
                except Exception:
                    options_list = row['options']  # 保持原样
            else:
                options_list = []
            
            # 构建标准格式的JSON对象
            formatted_row = {
                'id': row.get('question_id', ''),
                'task': row.get('task_type', ''),
                'question': row.get('question', ''),
                'options': options_list,
                'answer': row.get('answer', ''),
                'video_end': sum(int(x) * 60 ** i for i, x in enumerate(reversed(row.get('time_stamp', '00:00:00').split(':')))),  # 解析时间戳为秒
                'video': f"sample_{row.get('question_id', '').split('_')[-2]}/video.mp4",  # 从question_id提取视频文件名
                'frames_required': row.get('frames_required', ''),
                'temporal_clue_type': row.get('temporal_clue_type', '')
            }
            
            fout.write(json.dumps(formatted_row, ensure_ascii=False) + '\n')