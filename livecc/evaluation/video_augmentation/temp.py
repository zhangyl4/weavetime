import json

# 读取JSONL文件
video_paths = []
with open('/2022233235/videollm-online/livecc/ovo-bench-formatted.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line.strip())
        if 'video' in data:
            video_paths.append(data['video'])

print(f"Total video paths found: {len(video_paths)}")

import json
import random
import os

# 随机采样500个视频路径
random.seed(42)  # 设置随机种子以确保可重复性
sampled_video_paths = random.sample(video_paths, min(500, len(video_paths)))

# 创建保存目录
save_dir = '/2022233235/videollm-online/livecc/evaluation/video_augmentation'
os.makedirs(save_dir, exist_ok=True)

# 保存为JSON文件
save_path = os.path.join(save_dir, 'sampled_video_paths.json')
with open(save_path, 'w') as f:
    json.dump(sampled_video_paths, f, indent=2)

print(f"Successfully sampled {len(sampled_video_paths)} video paths")
print(f"Saved to: {save_path}")

# 显示前几个路径作为示例
print("\nFirst 5 video paths:")
for i, path in enumerate(sampled_video_paths[:5]):
    print(f"{i+1}. {path}")

