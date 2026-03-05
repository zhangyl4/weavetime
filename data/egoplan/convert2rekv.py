import os
import json
import pandas as pd

def convert_egoplan_to_rekv():
    # Read the parquet file
    df = pd.read_parquet("/root/.cache/huggingface/hub/datasets--lmms-lab--EgoPlan/snapshots/19a36d8fdf16de5169b9161442eb32f93f639ed1/egoplan/validation-00000-of-00001.parquet")
    
    output = []
    
    for _, row in df.iterrows():
        breakpoint()
        video_id = row['video_id']
        video_path = os.path.join("data/egoplan/videos", f"{video_id}.mp4")
        
        # Skip if video doesn't exist
        if not os.path.exists(video_path):
            continue
            
        conversations = []
        
        # Convert each QA pair
        for q, a in zip(row['questions'], row['answers']):
            qa_item = {
                "question": q,
                "choices": [a], # Since EgoPlan is not multiple choice, put answer as only choice
                "answer": a,
                "temporal_windows": [[0, -1]] # Use full video since timestamps not provided
            }
            conversations.append(qa_item)
            
        video_item = {
            "video_id": video_id,
            "video_path": video_path,
            "duration": -1, # Duration will be read from video file
            "conversations": conversations
        }
        
        output.append(video_item)

    # Save as JSON
    os.makedirs("data/egoplan", exist_ok=True)
    with open("data/egoplan/test_mc.json", "w") as f:
        json.dump(output, f, indent=4)

if __name__ == "__main__":
    convert_egoplan_to_rekv()
