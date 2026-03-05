
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


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    # 0. config
    parser.add_argument('--model-path', type=str, required=False, default="chenjoya/LiveCC-7B-Instruct")
    parser.add_argument('--video-path', type=str, required=False, default='/2022233235/videollm-online/livecc/demo/sources/howto_fix_laptop_mute_1080p.mp4')
    parser.add_argument('--prompt', type=str, required=False, default='describe the two time periods of this video respectively.')
    parser.add_argument('--output-path', type=str, required=False, default='./output')
    parser.add_argument('--device', type=str, required=False, default='cuda:0')
    pargs = parser.parse_args()
    streaming_fps_frames = int(FPS)

    # 1. load model and processor
    model = Qwen2VLForConditionalGeneration.from_pretrained(pargs.model_path, torch_dtype="auto", attn_implementation='flash_attention_2', device_map=pargs.device)
    processor = AutoProcessor.from_pretrained(pargs.model_path, padding_side='left')
    

    # 2. parser input with time stamp
    # Load video and parse input with timestamp
    video, sample_fps, clip_pts = _read_video_decord_plus({'video': pargs.video_path, 'remote_loader': None}, return_pts=True, strict_fps=True)
    video = _spatial_resize_video(video)
    video_start_timestamp, video_end_timestamp = clip_pts[0], clip_pts[-1]
    video_inputs = []
    
    # Format conversation with timestamp and video clip
    conversation = [{
        'role': 'user', 
        'content': []
    }]
    for i in range(0, len(video), streaming_fps_frames):
        start_timestamp, end_timestamp = i / FPS, (i + streaming_fps_frames) / FPS
        conversation[-1]['content'].append({'type': 'text', 'text': f'Time={start_timestamp:.1f}-{end_timestamp:.1f}s'})
        conversation[-1]['content'].append({'type': 'video', 'video': video[i:i+streaming_fps_frames]})
        video_inputs.append(video[i:i+streaming_fps_frames])
        
    assert video.shape[0] == len(clip_pts)
    
    # HACK: split video
    # # conversation[-1]['content'].append({'type': 'text', 'text': f'Time={video_start_timestamp:.1f}-{clip_pts[len(clip_pts) // 2 - 1]:.1f}s'})
    # conversation[-1]['content'].append({'type': 'video', 'video': video[:len(clip_pts) // 2]})   
    # video_inputs.append(video[:len(clip_pts) // 2])
    
    # # conversation[-1]['content'].append({'type': 'text', 'text': f'Time={50 + clip_pts[len(clip_pts) // 2]:.1f}-{50 + clip_pts[-1]:.1f}s'})
    # conversation[-1]['content'].append({'type': 'video', 'video': video[len(clip_pts) // 2: ]})  
    # video_inputs.append(video[len(clip_pts) // 2:])
        
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
            output_hidden_states=True,
            output_attentions=True,
            use_cache=True,
        )
    # Extract generated text and attention maps
    generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs['input_ids'], outputs)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    for ms in conversation:
        if ms['role'] == 'user':
            for content in ms['content']:
                if content['type'] == 'text':
                    print(content)
    print(f"Generated text: {output_text}")

    # 4. save output
    output_path = pargs.output_path + "/" + pargs.model_path.split("/")[-1]

    try:
        os.mkdir(output_path)
    except:
        pass


    with open(output_path+"/output_split.json","w") as f:
        # json dumps
        json.dump({"prompt":pargs.prompt,"video":pargs.video_path,"output": output_text},f,indent=4)
    
            
#  python /2022233235/videollm-online/livecc/vis/demo_livecc.py