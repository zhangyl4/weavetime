import os
import json
import argparse
from collections import defaultdict
from typing import Any, Dict, List


corrupted_video_path = ["/root/.cache/huggingface/hub/datasets--PolyU-ChenLab--ETBench/snapshots/2d7bce92b69624c3c26ac054e5d7947463568283/videos/summe/cooking.mp4"]



def infer_videos_root_from_annotations(ann_path: str) -> str:
    # annotations/... -> videos/
    root = ann_path
    # go up until 'annotations'
    parts = os.path.normpath(ann_path).split(os.sep)
    if 'annotations' in parts:
        idx = parts.index('annotations')
        snapshot_root = os.sep.join(parts[:idx])
        return os.path.join(snapshot_root, 'videos')
    # fallback: parent dir of annotations file, then sibling videos
    return os.path.join(os.path.dirname(os.path.dirname(ann_path)), 'videos')


def load_etbench_annotations(ann_path: str) -> List[Dict[str, Any]]:
    with open(ann_path, 'r') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError('Expected a list of ETBench samples')
    return data


def to_rekv_json(samples: List[Dict[str, Any]], videos_root: str) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}

    def ensure_item(video_rel: str, duration: Any) -> Dict[str, Any]:
        if video_rel not in grouped:
            video_abs = os.path.join(videos_root, video_rel) if video_rel else ""
            
            video_id = os.path.splitext(os.path.basename(video_rel))[0] if video_rel else ""
            grouped[video_rel] = {
                'video_id': video_id or video_rel,
                'video_path': video_abs,
                'duration': duration if duration is not None else 0,
                'conversations': []
            }
        else:
            # update duration if not set
            if grouped[video_rel].get('duration') in (None, 0) and duration is not None:
                grouped[video_rel]['duration'] = duration
        return grouped[video_rel]

    for s in samples:
        video_rel = s.get('video', '')
        duration = s.get('duration')
        
        if os.path.join(videos_root, video_rel) in corrupted_video_path:
            continue
        
        item = ensure_item(video_rel, duration)

        question = s.get('q', '')
        choices = s.get('o') if isinstance(s.get('o'), list) else None
        answer = None
        if choices is not None and 'p' in s:
            try:
                answer_idx = int(s.get('p'))
                if 0 <= answer_idx < len(choices):
                    answer = choices[answer_idx]
            except Exception:
                answer = None

        conv: Dict[str, Any] = {
            'question': question,
            'answer': answer or ""  # Always include answer field, empty string if no answer
        }
        if choices is not None:
            conv['choices'] = choices
        # Optional temporal windows if ETBench provides targets
        if isinstance(s.get('tgt'), list) and len(s['tgt']) > 0:
            conv['temporal_windows'] = s['tgt']

        item['conversations'].append(conv)

    return list(grouped.values())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='/root/.cache/huggingface/hub/datasets--PolyU-ChenLab--ETBench/snapshots/2d7bce92b69624c3c26ac054e5d7947463568283/annotations/etbench_txt_v1.0.json', help='Path to ETBench annotations JSON (e.g., etbench_txt_v1.0.json)')
    parser.add_argument('--output', default='/root/videollm-online/ReKV/data/etbench/etbench_rekv.json')
    parser.add_argument('--videos_root', default="/root/.cache/huggingface/hub/datasets--PolyU-ChenLab--ETBench/snapshots/2d7bce92b69624c3c26ac054e5d7947463568283/videos/", help='Override videos root; default inferred from snapshot path')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    videos_root = args.videos_root or infer_videos_root_from_annotations(args.input)
    samples = load_etbench_annotations(args.input)
    rekv = to_rekv_json(samples, videos_root)

    with open(args.output, 'w') as f:
        json.dump(rekv, f, indent=4)
    print(f'[I] Converted {len(rekv)} videos into ReKV format: {args.output}')

if __name__ == "__main__":
    main()