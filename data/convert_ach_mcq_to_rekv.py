import os
import json

# VidHalluc ach_mcq.json 路径
ACH_MCQ_PATH = "/home/docker_shared/asus/zhangyl/model/huggingface/hub/datasets--chaoyuli--VidHalluc/blobs/89b37bd117379ba1f3f3cfca150fc614d6b713d1"
# 输出目录
OUT_DIR = os.path.join(os.path.dirname(__file__), "vidhalluc")
# 输出文件
OUT_PATH = os.path.join(OUT_DIR, "ach_mcq_rekv.json")

# 假设视频路径规则（可根据实际情况调整）
VIDEO_BASE = "/home/docker_shared/asus/zhangyl/model/huggingface/hub/datasets--chaoyuli--VidHalluc/snapshots/76f942760e77e3759f4974db579928c032dcd05d/data/ACH"  # TODO: 替换为实际视频路径

os.makedirs(OUT_DIR, exist_ok=True)

def main():
    with open(ACH_MCQ_PATH, 'r') as f:
        ach_mcq = json.load(f)

    out_data = []
    for group in ach_mcq.values():
        for vid, qa in group.items():
            # 选项按 A/B/C/D 顺序
            choices = [qa["Choices"].get(opt, "") for opt in ["A", "B", "C", "D"]]
            out_data.append({
                "video_id": vid,
                "video_path": os.path.join(VIDEO_BASE, f"{vid}.mp4"),
                "conversations": [
                    {
                        "question": qa["Question"],
                        "choices": choices,
                        "answer": qa["Choices"][qa["Correct Answer"]]
                    }
                ]
            })

    with open(OUT_PATH, 'w') as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    print(f"Converted {len(out_data)} items to {OUT_PATH}")

if __name__ == "__main__":
    main() 