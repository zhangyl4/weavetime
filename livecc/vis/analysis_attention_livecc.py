
import argparse

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as Colormap
from matplotlib.colors import LogNorm


import json, os, torch, functools, tqdm, random, sys
import numpy as np
import decord
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, logging, Qwen2VLForConditionalGeneration, AutoProcessor

from livecc_utils import _read_video_decord_plus, _spatial_resize_video
from qwen_vl_utils.vision_process import process_vision_info, smart_nframes, FPS

def visualize_attention(multihead_attention,output_path="atten_map_1.png",title="Layer 5"):
    # Assuming the input is a numpy array of shape (1, num_heads, n_tokens, n_tokens)
    # First, we average the attention scores over the multiple heads
    averaged_attention = torch.mean(multihead_attention, axis=1)[0].float()# Shape: (n_tokens, n_tokens)
    # pooling the attention scores  with stride 20
    averaged_attention = torch.nn.functional.avg_pool2d(averaged_attention.unsqueeze(0).unsqueeze(0), 20, stride=20).squeeze(0).squeeze(0)
    
    cmap = plt.cm.get_cmap("viridis")
    plt.figure(figsize=(5, 5),dpi=400)

    # Log normalization
    log_norm = LogNorm(vmin=0.0007, vmax=averaged_attention.max())

    # set the x and y ticks to 20x of the original


    ax = sns.heatmap(averaged_attention,
                cmap=cmap,  # custom color map
                norm=log_norm,  # 
                # cbar_kws={'label': 'Attention score'},
                )
    
    # remove the x and y ticks
    
    # replace the x and y ticks with string

    x_ticks = [str(i*20) for i in range(0,averaged_attention.shape[0])]
    y_ticks = [str(i*20) for i in range(0,averaged_attention.shape[0])]
    ax.set_xticks([i for i in range(0,averaged_attention.shape[0])])
    ax.set_yticks([i for i in range(0,averaged_attention.shape[0])])
    ax.set_xticklabels(x_ticks)
    ax.set_yticklabels(y_ticks)

    # change the x tinks font size
    plt.xticks(fontsize=3)
    plt.yticks(fontsize=3)
    
    # make y label vertical
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)     
    
    plt.title(title)
    # tight layout
    plt.savefig(output_path, bbox_inches='tight')
    # plt.show()

    top_five_attentions = []
    for row in averaged_attention:
        # Use torch.topk to get the top 5 values and their indices
        top_values, top_indices = torch.topk(row, 10)
        # Convert to lists and append to the overall list
        top_five_line = list(zip(top_indices.tolist(), top_values.tolist()))
        top_five_attentions.append(top_five_line)
        
    return top_five_attentions,averaged_attention    





if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    # 0. config
    parser.add_argument('--model-path', type=str, required=False, default="chenjoya/LiveCC-7B-Instruct")
    parser.add_argument('--video-path', type=str, required=False, default='/2022233235/videollm-online/livecc/demo/sources/howto_fix_laptop_mute_1080p.mp4')
    parser.add_argument('--prompt', type=str, required=False, default='discribe the video')
    parser.add_argument('--output-path', type=str, required=False, default='./output')
    parser.add_argument('--device', type=str, required=False, default='cuda:0')
    pargs = parser.parse_args()
    

    # 1. load model and processor
    model = Qwen2VLForConditionalGeneration.from_pretrained(pargs.model_path, torch_dtype="auto", attn_implementation='eager', device_map=pargs.device)
    processor = AutoProcessor.from_pretrained(pargs.model_path, padding_side='left')
    

    # 2. parser input with time stamp
    # Load video and parse input with timestamp
    video, sample_fps, clip_pts = _read_video_decord_plus({'video': pargs.video_path, 'remote_loader': None}, return_pts=True, strict_fps=True)
    video = _spatial_resize_video(video)
    video_start_timestamp, video_end_timestamp = 0, clip_pts[-1]
    video_inputs = []
    
    # Format conversation with timestamp and video clip
    conversation = [{
        'role': 'user', 
        'content': []
    }]
    streaming_fps_frames = int(len(video))
    for i in range(0, len(video), streaming_fps_frames):
        start_timestamp, end_timestamp = i / FPS, (i + streaming_fps_frames) / FPS
        conversation[-1]['content'].append({'type': 'text', 'text': f'Time={start_timestamp:.1f}-{end_timestamp:.1f}s'})
        conversation[-1]['content'].append({'type': 'video', 'video': video[i:i+streaming_fps_frames]})
        video_inputs.append(video[i:i+streaming_fps_frames])
    conversation[-1]['content'].append({'type': 'text', 'text': pargs.prompt})
    

    # Prepare inputs for model
    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=text,
        images=None,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    # 3. inference with output attention
    # Move inputs to device
    inputs = {k: v.to(pargs.device) for k, v in inputs.items()}
    
    # Set up model to output attention
    model.eval()
    with torch.no_grad():
        # Forward pass with output_attentions=True to get attention maps
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            output_attentions=True,
            return_dict_in_generate=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
    # Extract generated text and attention maps
    generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs['input_ids'], outputs.sequences)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    outputs_attention = outputs.attentions  # This contains attention maps from all layers
    
    # Get total number of layers
    total_layers = len(outputs_attention[0]) if outputs_attention else 0
    
    print(f"Generated text: {output_text}")
    print(f"Number of attention layers: {total_layers}")

    # 4. save output
    output_path = pargs.output_path + "/" + pargs.model_path.split("/")[-1] + f"_{streaming_fps_frames}"

    try:
        os.mkdir(output_path)
    except:
        pass
    
    try:
        os.mkdir(output_path+"/attn_maps")
    except:
        pass



    with open(output_path+"/output.json","w") as f:
        # json dumps
        json.dump({"prompt":pargs.prompt,"video":pargs.video_path,"output": output_text},f,indent=4)
    
    
    
    # 5. merge attention maps from all generation steps
            
    # Merge attention maps from all generation steps
    # outputs_attention is a tuple where each element contains attention maps for one generation step
    # Each element has shape [1, 28, seq_len, seq_len] where seq_len increases with each step
    
    # Initialize merged attention maps with the first step
    merged_attention = list(outputs_attention[0])  # Convert to list for modification
    
    # For each subsequent generation step, concatenate along the sequence dimension
    for step_idx in range(1, len(outputs_attention)):
        current_step_attention = outputs_attention[step_idx]
        
        for layer_idx in range(len(merged_attention)):
            # Get current layer attention from this step
            current_layer_attn = current_step_attention[layer_idx].cpu()  # Shape: [1, 28, 1, seq_len]
            
            # Concatenate along sequence dimension (dim=2 for query, dim=3 for key)
            # This extends the attention matrix to include the new token
            
            # Create a zero tensor with the target shape [1, 28, seq_len-1, seq_len-1]
            target_shape = list(merged_attention[layer_idx].shape)
            target_shape[2] += 1  # Make seq_len consistent
            target_shape[3] += 1  # Make seq_len consistent
            zero_padded_attn = torch.zeros(target_shape, dtype=current_layer_attn.dtype)
            
            # Copy the previous attention maps to the upper-left block
            zero_padded_attn[:, :, :target_shape[2]-1, :target_shape[3]-1] = merged_attention[layer_idx].cpu()
            
            # Place the new attention weights [1, 28, 1, seq_len] in the last row (second dimension)
            zero_padded_attn[:, :, -1:, :] = current_layer_attn[:, :, :, :]
            
            # Update the merged attention
            merged_attention[layer_idx] = zero_padded_attn
            
    
    # Convert back to tuple
    merged_attention = tuple(merged_attention)
    
    print(f"Merged attention shape: {merged_attention[0].shape}")
    # draw attention maps
    for j in range(0,len(merged_attention)):
        top5_attention,average_attentions = visualize_attention(merged_attention[j].cpu(),output_path=output_path+"/attn_maps/atten_map_"+str(j)+".png",title="Layer "+str(j+1))
            
# VIDEO_MAX_PIXELS=1204224 FPS_MAX_FRAMES=50 python /2022233235/videollm-online/livecc/vis/analysis_attention_livecc.py