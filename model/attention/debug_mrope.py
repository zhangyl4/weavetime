import torch

def test_optimized_mrope():
    B, L = 2, 500
    spatial_merge_size = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup tokens
    vision_start_token_id = 151644
    vision_end_token_id = 151645
    video_token_id = 151646
    image_token_id = 151647
    
    input_ids = torch.zeros((B, L), dtype=torch.long, device=device)
    
    # Case 1: Time prompt video pattern (vision_start, video_tokens..., vision_end)
    # Frame block: 10 tokens (1 start, 8 video, 1 end)
    # Grid: 1x4x4 (pre-merge) -> 1x2x2 (post-merge) -> 4 tokens
    video_len = 8
    # [0...9] is frame 1
    input_ids[0, 0] = vision_start_token_id
    input_ids[0, 1:1+video_len] = video_token_id
    input_ids[0, 1+video_len] = vision_end_token_id
    
    # Case 2: Image
    input_ids[0, 20] = vision_start_token_id
    input_ids[0, 21:21+video_len] = image_token_id
    input_ids[0, 21+video_len] = vision_end_token_id

    video_grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long, device=device)
    image_grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long, device=device)
    
    # Run optimized logic (simulation)
    print("Simulating optimized logic...")
    
    # 1. Find all vision_start
    vision_start_mask = (input_ids == vision_start_token_id)
    # 2. Check next token for type
    # ... (implementation logic to be tested)

if __name__ == "__main__":
    test_optimized_mrope()
