#!/usr/bin/env python3
"""
Test script for video continuity detection functionality.
This script tests the core functions without requiring actual video files or models.
"""

import torch
import numpy as np
from distributed_evaluate_split import create_video_with_middle_crop, VideoContinuityDataset

def test_create_video_with_middle_crop():
    """Test the video cropping function with synthetic data."""
    print("Testing create_video_with_middle_crop()...")
    
    # Create a synthetic video tensor (10 frames, 3 channels, 64x64)
    T, C, H, W = 10, 3, 64, 64
    video = torch.randn(T, C, H, W)
    
    # Test with 30% crop ratio
    crop_ratio = 0.3
    cropped_video = create_video_with_middle_crop(video, crop_ratio)
    
    # Calculate expected dimensions
    crop_start = int(T * (1 - crop_ratio) / 2)  # 3
    crop_end = int(T * (1 + crop_ratio) / 2)    # 7
    expected_frames = crop_start + (T - crop_end)  # 3 + 3 = 6
    
    print(f"Original video shape: {video.shape}")
    print(f"Cropped video shape: {cropped_video.shape}")
    print(f"Expected frames: {expected_frames}")
    print(f"Actual frames: {cropped_video.shape[0]}")
    
    assert cropped_video.shape[0] == expected_frames, f"Expected {expected_frames} frames, got {cropped_video.shape[0]}"
    assert cropped_video.shape[1:] == video.shape[1:], "Channel, height, width should remain the same"
    
    print("✓ create_video_with_middle_crop() test passed!")
    return True

def test_video_continuity_dataset():
    """Test the VideoContinuityDataset class."""
    print("\nTesting VideoContinuityDataset...")
    
    # Create mock video paths
    mock_video_paths = [
        "/mock/path/video1.mp4",
        "/mock/path/video2.mp4",
        "/mock/path/video3.mp4"
    ]
    
    # Create dataset
    dataset = VideoContinuityDataset(
        video_paths=mock_video_paths,
        crop_ratio=0.3,
        sample=2  # Use only 2 videos for testing
    )
    
    print(f"Dataset length: {len(dataset)}")
    print(f"Expected length: {2 * 2} (2 videos × 2 versions each)")
    
    assert len(dataset) == 4, f"Expected 4 samples, got {len(dataset)}"
    
    # Check data structure
    for i in range(min(4, len(dataset))):
        datum = dataset.datums[i]
        print(f"Sample {i}: {datum}")
        
        assert 'id' in datum, "Missing 'id' field"
        assert 'video_path' in datum, "Missing 'video_path' field"
        assert 'is_continuous' in datum, "Missing 'is_continuous' field"
        assert 'crop_ratio' in datum, "Missing 'crop_ratio' field"
        
        # Check that continuous samples have crop_ratio = 0
        if datum['is_continuous']:
            assert datum['crop_ratio'] == 0.0, f"Continuous sample should have crop_ratio=0, got {datum['crop_ratio']}"
        else:
            assert datum['crop_ratio'] == 0.3, f"Discontinuous sample should have crop_ratio=0.3, got {datum['crop_ratio']}"
    
    print("✓ VideoContinuityDataset test passed!")
    return True

def test_data_collator():
    """Test the data collator function."""
    print("\nTesting data collator...")
    
    # Create mock dataset
    mock_video_paths = ["/mock/path/video1.mp4"]
    dataset = VideoContinuityDataset(video_paths=mock_video_paths, sample=1)
    
    # Mock processor for testing
    class MockProcessor:
        def apply_chat_template(self, conversations, tokenize=False, add_generation_prompt=True):
            return ["Mock text " + str(i) for i in range(len(conversations))]
        
        def __call__(self, text, images, videos, padding, return_tensors):
            # Mock the processor call
            return {
                'input_ids': torch.randint(0, 1000, (len(text), 10)),
                'attention_mask': torch.ones(len(text), 10),
                'video_patches': torch.randn(len(videos), 10, 768) if videos else None
            }
    
    # Create mock batch
    mock_batch = []
    for i in range(2):  # 2 samples
        conversation = [{"role": "user", "content": [{"type": "text", "text": "test"}]}]
        video_input = torch.randn(5, 3, 224, 224)  # Mock video tensor
        label = i % 2  # 0 or 1
        mock_batch.append((conversation, video_input, label))
    
    # Test data collator
    processor = MockProcessor()
    inputs, labels = dataset.data_collator(mock_batch, processor)
    
    print(f"Inputs keys: {inputs.keys()}")
    print(f"Labels shape: {labels.shape}")
    print(f"Labels: {labels}")
    
    assert 'input_ids' in inputs, "Missing input_ids in processor output"
    assert 'attention_mask' in inputs, "Missing attention_mask in processor output"
    assert labels.shape[0] == 2, f"Expected 2 labels, got {labels.shape[0]}"
    
    print("✓ Data collator test passed!")
    return True

def main():
    """Run all tests."""
    print("=" * 50)
    print("VIDEO CONTINUITY DETECTION TESTS")
    print("=" * 50)
    
    try:
        test_create_video_with_middle_crop()
        test_video_continuity_dataset()
        test_data_collator()
        
        print("\n" + "=" * 50)
        print("ALL TESTS PASSED! ✓")
        print("=" * 50)
        print("\nThe video continuity detection functionality is working correctly.")
        print("You can now use it with actual video files and models.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main() 