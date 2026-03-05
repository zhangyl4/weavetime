#!/usr/bin/env python3
"""
Convert StreamingBench RTVU formatted JSONL to ReKV format (like ovo_bench_rekv.json)
"""

import json
from collections import defaultdict
from pathlib import Path


def convert_streamingbench_to_rekv(input_file, output_file):
    """
    Convert StreamingBench format to ReKV format
    
    StreamingBench format (JSONL):
    {
        "id": "Real-Time Visual Understanding_sample_1_1",
        "task": "Object Perception",
        "question": "What logos are visible...",
        "options": ["A. FIFA and La Liga.", "B. UEFA and La Liga.", ...],
        "answer": "D",
        "video_end": 4,
        "video": "sample_1/video.mp4",
        "frames_required": "single",
        "temporal_clue_type": "Prior"
    }
    
    ReKV format (JSON array):
    {
        "video_id": "sample_1",
        "video_path": "data/StreamingBench/sample_1/video.mp4",
        "conversations": [
            {
                "question": "What logos are visible...",
                "choices": ["FIFA and La Liga.", "UEFA and La Liga.", ...],
                "answer": "Aeromexico and La Liga.",
                "start_time": 0,
                "end_time": 4,
                "task": "Object Perception"
            },
            ...
        ]
    }
    """
    
    # Read JSONL and group by video
    video_groups = defaultdict(list)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                video_path = item['video']
                video_groups[video_path].append(item)
    
    # Convert to ReKV format
    rekv_data = []
    
    for video_path, items in sorted(video_groups.items()):
        # Extract video_id from video path (e.g., "sample_1/video.mp4" -> "sample_1")
        video_id = video_path.split('/')[0] if '/' in video_path else Path(video_path).stem
        
        max_end_time = None
        if video_path == 'sample_332/video.mp4':
            max_end_time = 200
        
        # Sort items by video_end to maintain temporal order
        items_sorted = sorted(items, key=lambda x: x['video_end'])
        
        conversations = []
        for item in items_sorted:
            # Extract choices from options (remove "A. ", "B. ", etc.)
            choices = []
            for option in item['options']:
                # Remove the letter prefix (e.g., "A. " -> "")
                choice_text = option.split('. ', 1)[1] if '. ' in option else option
                choices.append(choice_text)
            
            # Convert answer letter to actual answer text
            answer_index = ord(item['answer']) - ord('A')
            answer_text = choices[answer_index] if 0 <= answer_index < len(choices) else ""
            
            # Create conversation entry
            if max_end_time is not None:
                end_time = min(item['video_end'], max_end_time)
            else:
                end_time = item['video_end']
            
            conversation = {
                "question": item['question'],
                "choices": choices,
                "answer": answer_text,
                "start_time": 0,  # StreamingBench doesn't have start_time, use 0
                "end_time": end_time,
                "task": item['task']
            }
            
            # Add optional fields if needed
            if 'frames_required' in item:
                conversation['frames_required'] = item['frames_required']
            if 'temporal_clue_type' in item:
                conversation['temporal_clue_type'] = item['temporal_clue_type']
            
            conversations.append(conversation)
        
        # Create video entry
        video_base_dir = "/root/.cache/huggingface/hub/datasets--mjuicem--StreamingBench/snapshots/48872fa707124474ce4c5172ddc58efb8bc88058"
        video_entry = {
            "video_id": video_id,
            "video_path": f"{video_base_dir}/{video_path}",
            "conversations": conversations
        }
        
        rekv_data.append(video_entry)
    
    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(rekv_data, f, indent=2, ensure_ascii=False)
    
    print(f"Conversion complete!")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Total videos: {len(rekv_data)}")
    print(f"Total conversations: {sum(len(v['conversations']) for v in rekv_data)}")


if __name__ == "__main__":
    input_file = "/root/videollm-online/ReKV/data/StreamingBench/streamingbench-rtvu-formatted.jsonl"
    output_file = "/root/videollm-online/ReKV/data/StreamingBench/streamingbench_rekv.json"
    
    convert_streamingbench_to_rekv(input_file, output_file)

