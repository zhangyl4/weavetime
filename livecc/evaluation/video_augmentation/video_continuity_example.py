#!/usr/bin/env python3
"""
Example script for video continuity detection using LiveCC/Qwen2VL models.
This script demonstrates how to use the VideoContinuityDataset to test if a model
can detect when a video has been artificially cropped in the middle.
"""

import os
import json
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from distributed_evaluate_split import video_continuity_predict, evaluate_video_continuity_results

def main():
    # Configuration
    model_path = "Qwen/Qwen2-VL-7B-Instruct"  # or "chenjoya/LiveCC-7B-Instruct"
    use_liger_kernel = 'LiveCC' in model_path
    
    # Load model and processor
    print(f"Loading model from {model_path}...")
    if "Qwen2.5" in model_path:
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype="auto", 
            attn_implementation='flash_attention_2'
        )
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype="auto", 
            attn_implementation='flash_attention_2'
        )
    
    processor = AutoProcessor.from_pretrained(model_path, padding_side='left')
    
    # Example video paths - replace with your actual video paths
    # You can use any video files you have available
    video_paths = [
        # Add your video paths here, for example:
        # "/path/to/video1.mp4",
        # "/path/to/video2.mp4",
        # "/path/to/video3.mp4",
    ]
    
    # If no video paths provided, create a simple test
    if not video_paths:
        print("No video paths provided. Please add video paths to the video_paths list.")
        print("Example usage:")
        print("video_paths = ['/path/to/video1.mp4', '/path/to/video2.mp4']")
        return
    
    print(f"Found {len(video_paths)} video files")
    
    # Run video continuity prediction
    print("Running video continuity detection...")
    predictions, dataset, process_index = video_continuity_predict(
        model=model, 
        processor=processor, 
        video_paths=video_paths,
        crop_ratio=0.3,  # Crop 30% from the middle for discontinuous videos
        question_prefix="", 
        question_postfix="\nPlease answer Yes or No.", 
        answer_prefix="The answer is:\n", 
        yes_no_previous_str="\n",
        remote_loader=None,
        use_liger_kernel=use_liger_kernel,
        per_device_eval_batch_size=1,  # Reduce batch size if you have memory issues
        dataloader_num_workers=2,
        sample=10  # Use only 10 samples for testing
    )
    
    # Evaluate results (only on main process)
    if process_index == 0:
        print("\n" + "="*50)
        print("VIDEO CONTINUITY DETECTION RESULTS")
        print("="*50)
        
        metrics = evaluate_video_continuity_results(predictions, dataset)
        
        # Save results
        save_dir = f'evaluation/video_continuity/results'
        os.makedirs(save_dir, exist_ok=True)
        
        save_json_path = os.path.join(save_dir, f'{os.path.basename(model_path)}_continuity_results.json')
        with open(save_json_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nResults saved to: {save_json_path}")
        
        # Print detailed results
        print("\nDetailed Results:")
        print("-" * 30)
        for i, (prediction, datum) in enumerate(zip(predictions, dataset.datums)):
            predicted_answer = "Yes" if prediction == 1 else "No"
            actual_answer = "Yes" if datum['is_continuous'] else "No"
            correct = "✓" if prediction == (1 if datum['is_continuous'] else 0) else "✗"
            
            print(f"Sample {i+1}: {datum['id']}")
            print(f"  Video: {os.path.basename(datum['video_path'])}")
            print(f"  Crop ratio: {datum['crop_ratio']:.1%}")
            print(f"  Actual: {actual_answer} (continuous: {datum['is_continuous']})")
            print(f"  Predicted: {predicted_answer} {correct}")
            print()

if __name__ == "__main__":
    main() 