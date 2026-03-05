import json
import shutil

def convert_path(path):
    return path.replace('2022233235', '2024233235')

input_file = '/2024233235/videollm-online/EyeWO2/data/cof_qwen2vl.jsonl'
input_file = '/2024233235/videollm-online/EyeWO2/data/llava_video_178k_with_seeks_sample_valid.jsonl'
input_file = "/2024233235/videollm-online/EyeWO2/data/etbench_qwen2vl.jsonl"
# input_file = "/2024233235/videollm-online/EyeWO2/data/cof_qwen2vl.jsonl"
backup_file = input_file + '.bak'

# 先copy一份input_file
shutil.copy(input_file, backup_file)

output_lines = []

with open(input_file, 'r') as f:
    lines = f.readlines()
    for line in lines[:-1]:  # 跳过最后一行
        data = json.loads(line)
        
        # Process each conversation
        for message in data:
            if message['role'] == 'user':
                for content in message['content']:
                    if content.get('type') == 'video' and 'video' in content:
                        content['video'] = convert_path(content['video'])
        
        output_lines.append(json.dumps(data))

# Write back to input_file
with open(input_file, 'w') as f:
    for line in output_lines[:-1]:
        f.write(line + '\n')
    f.write(output_lines[-1])
