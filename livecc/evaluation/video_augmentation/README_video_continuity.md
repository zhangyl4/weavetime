# Video Continuity Detection

This module provides functionality to test video-language models' ability to detect video continuity. It creates artificial discontinuities by cropping the middle portion of videos and asks the model to determine if the video is continuous or not.

## Overview

The video continuity detection task works as follows:

1. **Input**: A list of video files
2. **Processing**: For each video, create two versions:
   - **Continuous**: Original video (no modification)
   - **Discontinuous**: Video with middle portion cropped out
3. **Question**: Ask the model "Is this video continuous without any cuts or jumps in the middle?"
4. **Output**: Model predicts "Yes" (continuous) or "No" (discontinuous)

## Key Components

### 1. `create_video_with_middle_crop()`
Creates artificial discontinuities by removing the middle portion of a video.

```python
def create_video_with_middle_crop(video: torch.Tensor, crop_ratio: float = 0.3):
    """
    Create a video with middle part cropped out to simulate discontinuity.
    
    Args:
        video (torch.Tensor): Input video tensor with shape (T, C, H, W)
        crop_ratio (float): Ratio of frames to crop from the middle (0.0-1.0)
    
    Returns:
        torch.Tensor: Video with middle part cropped out
    """
```

### 2. `VideoContinuityDataset`
A PyTorch Dataset class that handles video loading, cropping, and question generation.

```python
class VideoContinuityDataset(Dataset):
    def __init__(self, video_paths, remote_loader=None, crop_ratio=0.3, 
                 question_prefix="", question_postfix="\nPlease answer Yes or No.", 
                 answer_prefix="Answer:", sample=None):
```

### 3. `video_continuity_predict()`
Main function to run video continuity detection using a trained model.

```python
def video_continuity_predict(model, processor, video_paths, crop_ratio=0.3, ...):
```

### 4. `evaluate_video_continuity_results()`
Evaluates the model's performance on the continuity detection task.

## Usage

### Basic Usage

```python
from distributed_evaluate_split import video_continuity_predict, evaluate_video_continuity_results
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# Load model and processor
model_path = "Qwen/Qwen2-VL-7B-Instruct"  # or "chenjoya/LiveCC-7B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto")
processor = AutoProcessor.from_pretrained(model_path, padding_side='left')

# Define video paths
video_paths = ['/path/to/video1.mp4', '/path/to/video2.mp4']

# Run prediction
predictions, dataset, process_index = video_continuity_predict(
    model=model,
    processor=processor,
    video_paths=video_paths,
    crop_ratio=0.3,  # Crop 30% from middle
    sample=10  # Use 10 samples for testing
)

# Evaluate results
if process_index == 0:
    metrics = evaluate_video_continuity_results(predictions, dataset)
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
```

### Using the Example Script

1. Edit `video_continuity_example.py` and add your video paths:
```python
video_paths = [
    "/path/to/your/video1.mp4",
    "/path/to/your/video2.mp4",
    # Add more video paths...
]
```

2. Run the script:
```bash
python video_continuity_example.py
```

## Parameters

### `video_continuity_predict()` Parameters

- `model`: The video-language model to use
- `processor`: The model's processor
- `video_paths`: List of video file paths
- `crop_ratio`: Ratio of frames to crop from middle (default: 0.3)
- `question_prefix/postfix`: Customize the question text
- `answer_prefix`: Prefix for model's answer
- `yes_no_previous_str`: String before Yes/No options
- `use_liger_kernel`: Whether to use liger kernel (for LiveCC models)
- `per_device_eval_batch_size`: Batch size per device
- `dataloader_num_workers`: Number of dataloader workers
- `sample`: Number of samples to use (for testing)

## Output

The evaluation provides the following metrics:

- **Overall Accuracy**: Overall classification accuracy
- **Continuous Videos Accuracy**: Accuracy on continuous videos
- **Discontinuous Videos Accuracy**: Accuracy on discontinuous videos
- **Sample counts**: Total number of samples in each category

## Example Output

```
==================================================
VIDEO CONTINUITY DETECTION RESULTS
==================================================
Overall Accuracy: 0.8500
Continuous Videos Accuracy: 0.8000 (4/5)
Discontinuous Videos Accuracy: 0.9000 (9/10)

Detailed Results:
------------------------------
Sample 1: 0_continuous
  Video: video1.mp4
  Crop ratio: 0.0%
  Actual: Yes (continuous: True)
  Predicted: Yes ✓

Sample 2: 0_discontinuous
  Video: video1.mp4
  Crop ratio: 30.0%
  Actual: No (continuous: False)
  Predicted: No ✓
```

## Supported Models

- **Qwen2-VL**: `Qwen/Qwen2-VL-7B-Instruct`
- **LiveCC**: `chenjoya/LiveCC-7B-Instruct`
- **Qwen2.5-VL**: Any Qwen2.5-VL model

## Notes

- The crop ratio determines how much of the middle portion is removed (0.3 = 30%)
- Each video generates two samples: one continuous and one discontinuous
- The model is asked to classify each sample as "Yes" (continuous) or "No" (discontinuous)
- Results are saved in JSON format for further analysis
- The task tests the model's temporal understanding and ability to detect artificial discontinuities 