import os
import json

# 源问题文件夹和视频文件夹
QUESTIONS_DIR = "/home/docker_shared/asus/zhangyl/model/huggingface/hub/EventHallusion/questions"
VIDEOS_DIR = "/home/docker_shared/asus/zhangyl/model/huggingface/hub/EventHallusion/videos/"
# 输出文件
OUTPUT_PATH = "all_questions.json"

CHOICES = ["Yes", "No"]

def convert_questions():    
    all_samples = []
    for fname in os.listdir(QUESTIONS_DIR):
        if not fname.endswith('_questions.json'):
            continue
        split = fname.split('_')[0] if fname.split('_')[0] != 'mix' else 'interleave'
        task_video_dir = os.path.join(VIDEOS_DIR, split)
        fpath = os.path.join(QUESTIONS_DIR, fname)
        with open(fpath, 'r') as f:
            data = json.load(f)
        # 兼容单个样本或样本列表
        if isinstance(data, dict):
            data = [data]
        for sample in data:
            sample_id = sample.get('id').split('_')[-1]
            questions = sample.get('questions', [])
            # 生成video_path
            video_path = os.path.join(task_video_dir, f"{split}_{sample_id}.mp4")
            # conversations标准化
            conversations = []
            for q in questions:
                conversations.append({
                    "question": q.get("question", ""),
                    # "choices": CHOICES,
                    "answer": q.get("answer", None).replace(".", "")
                })
            new_sample = {
                "video_id": sample_id,
                "video_path": video_path,
                "conversations": conversations
            }
            all_samples.append(new_sample)
    return all_samples


def main():
    output_file = os.path.join(os.path.dirname(__file__), OUTPUT_PATH)
    samples = convert_questions()
    with open(output_file, 'w') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    print(f"转换完成，输出文件: {output_file}，共 {len(samples)} 个样本")


if __name__ == "__main__":
    main() 