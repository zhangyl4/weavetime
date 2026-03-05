import json
import random
from decord import VideoReader
from typing import Tuple
import multiprocessing as mp
from tqdm import tqdm

def is_video_valid_and_info(video_path: str) -> Tuple[bool, float, float]:
    """Validate video readability with decord by sampling 10 random frames.
    Returns (is_valid, fps, duration_seconds)."""
    try:
        vr = VideoReader(video_path, num_threads=1)
        if len(vr) <= 0:
            return False, 0.0, 0.0
            
        # Sample 10 random frames
        frame_indices = random.sample(range(len(vr)), min(10, len(vr)))
        for idx in frame_indices:
            # Try to read each sampled frame
            vr[idx].asnumpy()
            
        # Get video info
        vr.get_frame_timestamp(0)
        video_pts = vr._frame_pts[:, 1]
        duration = float(video_pts[-1]) if len(video_pts) > 0 else (len(vr) / float(vr.get_avg_fps()))
        fps = float(vr.get_avg_fps())
        
        return True, fps, duration
        
    except Exception as e:
        print(f"Invalid video {video_path}: {e}")
        return False, 0.0, 0.0

def check_video_worker(video_path: str) -> Tuple[str, bool, float, float]:
    """Worker function to check a single video"""
    is_valid, fps, duration = is_video_valid_and_info(video_path)
    return video_path, is_valid, fps, duration

def main():
    # Read JSON file
    json_path = "/root/videollm-online/ReKV/data/etbench/etbench_rekv.json"
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Get all unique video paths
    video_paths = list({item['video_path'] for item in data})
    
    # Create process pool
    pool = mp.Pool(processes=mp.cpu_count() // 2)
    # pool= mp.Pool(processes=1)
    
    # Process videos in parallel with progress bar
    results = []
    for result in tqdm(pool.imap_unordered(check_video_worker, video_paths), 
                      total=len(video_paths)):
        results.append(result)
    
    pool.close()
    pool.join()
    
    # Print summary
    valid_count = sum(1 for _, is_valid, _, _ in results if is_valid)
    print(f"\nResults:")
    print(f"Total videos: {len(video_paths)}")
    print(f"Valid videos: {valid_count}")
    print(f"Invalid videos: {len(video_paths) - valid_count}")
    
    # Print invalid video paths
    print("\nInvalid videos:")
    for path, is_valid, _, _ in results:
        if not is_valid:
            print(path)

if __name__ == "__main__":
    main()