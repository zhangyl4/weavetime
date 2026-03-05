import json, os, torch, functools, tqdm, random, sys, typing, argparse, multiprocessing, shutil
import numpy as np
import decord
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, logging, Qwen2VLForConditionalGeneration, AutoProcessor, SinkCache
from functools import partial

from livecc_utils import prepare_multiturn_multimodal_inputs_for_generation, _read_video_decord_plus, _spatial_resize_video
from qwen_vl_utils.vision_process import process_vision_info, smart_nframes, FPS

from utils.multiprocessor import local_mp


logger = logging.get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Distributed OVOBench evaluation with MRopeSinkCache"
    )
    parser.add_argument(
        "--model_name_or_path", default="Qwen/Qwen2-VL-7B-Instruct", type=str,
        help="HuggingFace model path, e.g., chenjoya/LiveCC-7B-Instruct"
    )
    parser.add_argument(
        "--num_workers", type=int, default=8,
        help="Number of parallel processes/gpus to use"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="evaluation/ovobench/results",
        help="Directory to write generated JSON outputs"
    )
    parser.add_argument(
        "--benchmark_path", type=str,
        default="ovo-bench-formatted.jsonl",
        help="Path to OVOBench dataset"
    )
    parser.add_argument(
        "--sample_size", type=int, default=None,
        help="Number of samples to evaluate (None for all)"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=32,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Run in single-card debug mode"
    )
    return parser.parse_args()

import sys
sys.path.insert(0, '/2022233235/videollm-online/')
from demo.inference_qwen2vl_memory_v2 import MRopeDynamicCache

def _read_may1fps_video_decord(ele: dict):
    """read video using decord.VideoReader. can handle more cases compared to _read_video_decord.

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
        sample_fps
        clip_pts if return_pts=True
    """
    video_path = ele["video"]
    if os.path.exists(video_path):
        vr = decord.VideoReader(video_path, num_threads=2)
    elif ele['remote_loader'] is not None:
        vr = decord.VideoReader(ele['remote_loader'](video_path), num_threads=2)
    else:
        raise ValueError(f'video_path {video_path} not found')
    video_start = ele.get('video_start', None)
    video_end = ele.get('video_end', None)
    video_fps = vr.get_avg_fps()
    clip_idxs, clip_pts = None, None
    if video_start is not None or video_end is not None:
        vr.get_frame_timestamp(0)
        video_pts = vr._frame_pts[:,1]
        video_start = video_pts[0] if not video_start else video_start
        video_end = video_pts[-1] if not video_end else video_end
        video_start = min(max(video_pts[0], video_start), video_pts[-1])
        video_end = min(max(video_pts[0], video_end), video_pts[-1])
        video_end = max(video_start + 1, video_end)
        clip_idxs = ((video_start <= video_pts) & (video_pts <= video_end)).nonzero()[0]
        total_frames = len(clip_idxs)
    else:
        total_frames = len(vr)
    total_frames_for_smart_nframes = total_frames
    video_fps_for_smart_nframes = video_fps
    if total_frames < 2:
        total_frames_for_smart_nframes = 2
    if video_fps < FPS:
        total_frames_for_smart_nframes = int(total_frames * FPS / video_fps)
        video_fps_for_smart_nframes = FPS
    nframes = smart_nframes(ele, total_frames=total_frames_for_smart_nframes, video_fps=video_fps_for_smart_nframes) 
    nframes_idxs = np.linspace(0, total_frames - 1, nframes).round().astype(int)
    clip_idxs = nframes_idxs if clip_idxs is None else clip_idxs[nframes_idxs]
    clip = torch.from_numpy(vr.get_batch(clip_idxs).asnumpy()).permute(0, 3, 1, 2)  # Convert to TCHW format
    sample_fps = len(clip_idxs) / max(total_frames, 1e-6) * video_fps
    return clip, sample_fps

def save_function_print(function: callable, save_path: str, *args, **kwargs):
    original_stdout = sys.stdout
    try:
        with open(save_path, 'w') as f:
            sys.stdout = f  
            function(*args, **kwargs)          
    finally:
        sys.stdout = original_stdout 


def preprocess_logits_for_metrics(logits, labels, strict_option_ids): 
    return torch.stack([logit[(logit[:, 0] != -100).nonzero().squeeze()[-1], strict_option_ids] for logit in logits]).argmax(dim=-1)

def process_past_key_values(past_key_values, model, processor, answer_prefix, max_new_tokens=32, custom_processing_func=None):
    """
    Process past_key_values and generate answer with answer_prefix
    
    Args:
        past_key_values: MRopeSinkCache instance with cached key-value pairs
        model: The model to use for generation
        processor: The processor for tokenization
        answer_prefix: The prefix to add before generation
        max_new_tokens: Maximum number of new tokens to generate
        custom_processing_func: Optional custom function to process past_key_values before generation
    
    Returns:
        generated_text: The generated text after the answer_prefix
    """
    # Apply custom processing if provided
    if custom_processing_func is not None:
        past_key_values = custom_processing_func(past_key_values)
    
    # Tokenize the answer prefix
    prefix_inputs = processor(
        text=answer_prefix,
        return_tensors="pt",
        add_special_tokens=False
    )
    prefix_inputs.to(model.device)
    
    # Generate with the cached past_key_values
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=prefix_inputs.input_ids,
            attention_mask=prefix_inputs.attention_mask,
            past_key_values=past_key_values,
            return_dict_in_generate=True,
            max_new_tokens=max_new_tokens,
            pad_token_id=model.config.eos_token_id,
            do_sample=False,
        )
    
    # Decode the generated text (excluding the prefix)
    generated_text = processor.decode(outputs.sequences[0, prefix_inputs.input_ids.size(1):], skip_special_tokens=True)
    return generated_text

def forward_with_mrope_cache(model, processor, conversation, video_input, mrope_section, window_length=2048, num_sink_tokens=1024):
    """
    Forward pass to get past_key_values using MRopeSinkCache
    
    Args:
        model: The model to use for forward pass
        processor: The processor for tokenization
        conversation: The conversation to process
        video_input: The video input
        mrope_section: MRoPE section configuration
        window_length: Cache window length
        num_sink_tokens: Number of sink tokens
    
    Returns:
        past_key_values: MRopeSinkCache instance with cached key-value pairs
    """
    # Initialize MRopeSinkCache
    past_key_values = MRopeSinkCache(
        mrope_section=mrope_section, 
        window_length=window_length, 
        num_sink_tokens=num_sink_tokens
    )
    
    # Process the conversation (without answer_prefix)
    texts = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    
    inputs = processor(
        text=texts,
        images=None,
        videos=[video_input],
        return_tensors="pt",
        return_attention_mask=False
    )
    inputs.to(model.device)
    
    # Forward pass to get past_key_values (without generation)
    with torch.inference_mode():
        model_outputs = model(
            **inputs,
            past_key_values=past_key_values,
            return_dict=True,
            use_cache=True
        )
        # Update past_key_values with the forward pass results
        past_key_values = model_outputs.past_key_values
    
    return past_key_values


def ovobench_worker(
    device_id: int,
    model_name_or_path: str,
    save_dir: str,
    num_workers: int,
    benchmark_path: str,
    sample_size: int = None,
    max_new_tokens: int = 32,
):
    """
    Worker function for distributed OVOBench evaluation
    
    Args:
        device_id: GPU device ID
        model_name_or_path: Model path
        save_dir: Directory to save results
        num_workers: Total number of workers
        benchmark_path: Path to benchmark dataset
        sample_size: Number of samples to evaluate
        window_length: Cache window length
        num_sink_tokens: Number of sink tokens
        max_new_tokens: Maximum new tokens to generate
    """
    
    # Vision selection criteria for cache refresh
    vision_selection_criteria = {
        'keep_ratio': 0.8,                # Not used in conversation-level selection
        'importance_threshold': 0.5,       # Not used in conversation-level selection  
        'temporal_weight': 0.3,           # Not used in conversation-level selection
        'vision_weight': 0.7,             # Not used in conversation-level selection
        'keep_recent_turns': 5           # Keep first turn + recent 10 turns (including vision content)
        # Example configurations:
        # 'keep_recent_turns': 5    # More aggressive compression
        # 'keep_recent_turns': 20   # Less compression, keep more history
    }
    
    # 1. Load model on specific device
    if "Qwen2.5" in model_name_or_path:
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path, 
            torch_dtype="auto", 
            attn_implementation='flash_attention_2', 
            device_map=f'cuda:{device_id}'
        )
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name_or_path, 
            torch_dtype="auto", 
            attn_implementation='flash_attention_2', 
            device_map=f'cuda:{device_id}'
        )
    model.prepare_inputs_for_generation = functools.partial(prepare_multiturn_multimodal_inputs_for_generation, model)
    processor = AutoProcessor.from_pretrained(model_name_or_path, padding_side='left')
    
    # 2. Load dataset and split for this device
    lines = open(benchmark_path).readlines()
    if sample_size is not None:
        random.seed(42)
        lines = random.sample(lines, sample_size)
    
    datums = [json.loads(line) for line in lines]
    if isinstance(datums[0], str):
        datums = [json.loads(datum) for datum in datums]
    datums = [datum for datum in datums if datum['task'] not in ['REC', 'SSR', 'CRR']]
    
    # Split datums for this device
    idxs = list(range(len(datums)))
    idxs_on_device = idxs[device_id::num_workers]
    datums_on_device = [datums[i] for i in idxs_on_device]
    
    # 3. Get mrope_section from model config
    mrope_section = model.config.rope_scaling.get("mrope_section", [])
    
    # 4. Setup paths
    src_video_dir = os.path.dirname("/2022233235/.cache/huggingface/hub/datasets--JoeLeelyf--OVO-Bench/snapshots/fec29e3385747b5642d995370143ba92d2819bd2/src_videos/")
    os.makedirs(save_dir, exist_ok=True)
    
    # 5. Options for MCQ
    options = ['No', 'Yes', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E']
    
    # 6. Process each datum on this device
    for idx in tqdm.tqdm(idxs_on_device, desc=f"Device {device_id}", total=len(idxs_on_device)):
        save_path = os.path.join(save_dir, f"{idx}.json")
        if os.path.exists(save_path):
            continue
        
        datum = datums[idx]
        
        # try:
        # Prepare conversation
        conversation = [{"role": "user", "content": []}]
        
        # Read video
        if datum['task'] in ['REC', 'SSR', 'CRR']:
            query = datum['question']
            video, _ = _read_may1fps_video_decord({
                'video': os.path.join(src_video_dir, datum['video']), 
                'video_end': datum['video_end'],
                'remote_loader': None
            })
        else:
            query = datum['question'] + '\n' + '\n'.join(datum['options']) + '\nPlease select the correct answer.'
            video, _ = _read_may1fps_video_decord({
                'video': os.path.join(src_video_dir, datum['video']), 
                'video_end': datum['video_end'], 
                'remote_loader': None
            })
        
        video = _spatial_resize_video(video)
        
        conversation[0]['content'].append({"type": "video", "video": video})
        conversation[0]['content'].append({"type": "text", "text": query})
        video_inputs = [video]
        
        # HACK: add timestamp to video
        # video_inputs = []
        # streaming_fps_frames = int(FPS)
        # vison_content = []
        # for i in range(0, len(video), streaming_fps_frames):
        #     start_timestamp, end_timestamp = i / FPS, (i + streaming_fps_frames) / FPS
            
        #     vison_content.append([{'type': 'text', 'text': f'Time={start_timestamp:.1f}-{end_timestamp:.1f}s'},
        #                           {'type': 'video', 'video': video[i:i+streaming_fps_frames]}])
        #     video_inputs.append(video[i:i+streaming_fps_frames])
        
        # for i in range(0, len(vison_content)):
        #     conversation[-1]['content'].extend(vison_content[i])
        
        # Step 1: Forward pass to get past_key_values
        past_key_values = MRopeDynamicCache(mrope_section=mrope_section)
        
        # Process the conversation (without answer_prefix)
        texts = processor.apply_chat_template([conversation], tokenize=False, add_generation_prompt=True)
        answer_prefix = 'The answer is:\n'
        texts = [text + answer_prefix for text in texts]
        
        inputs = processor(
            text=texts,
            images=None,
            videos=video_inputs,
            return_tensors="pt",
            return_attention_mask=False
        )
        inputs.to(model.device)
        
        # # Forward pass to get past_key_values (without generation)
        # with torch.inference_mode():
        #     model_outputs = model(
        #         **inputs,
        #         past_key_values=past_key_values,
        #         return_dict=True,
        #         use_cache=True
        #     )
        #     # Update past_key_values with the forward pass results
        #     past_key_values = model_outputs.past_key_values
        #     past_ids = inputs['input_ids']
        #     print(past_key_values.get_seq_length())
        #     print(past_ids.shape)
        # # Step 2: Refresh cache
        # if vision_selection_criteria is not None:
        #     # Create full inputs dict for global storage update
        #     global_inputs = {
        #         'input_ids': past_ids,
        #         'image_grid_thw': inputs.get('image_grid_thw', None),
        #         'video_grid_thw': inputs.get('video_grid_thw', None),
        #         'attention_mask': inputs.get('attention_mask', None)
        #     }
        #     past_key_values.update_global(global_inputs)
            
        #     refreshed_input_ids = past_key_values.refresh_cache(global_inputs, model, vision_selection_criteria)
        #     past_ids = refreshed_input_ids['input_ids']
        #     print(past_key_values.get_seq_length())
        #     print(past_ids.shape)
        # Step 2: Process past_key_values and generate answer
        
        # Tokenize the answer prefix
        # answer_prefix = 'The answer is:\n'
        # texts = [text + answer_prefix for text in texts]
        # inputs = processor(
        #     text=texts,
        #     images=None,
        #     videos=video_inputs,
        #     return_tensors="pt",
        #     return_attention_mask=False
        # )
        
        # Generate with the cached past_key_values
        with torch.inference_mode():
            outputs = model.generate(
                **inputs, past_key_values=past_key_values, 
                return_dict_in_generate=True, 
                max_new_tokens=max_new_tokens,
                pad_token_id=model.config.eos_token_id,
                use_cache=True,
            )
        
        # Step 3: Decode the generated text (excluding the prefix)
        generated_text = processor.decode(outputs.sequences[0, inputs.input_ids.size(1):], skip_special_tokens=True)
        
        # Find the option that matches the generated text
        prediction_idx = None
        if generated_text.strip():
            first_char = generated_text.strip()[0].upper()
            for j, option in enumerate(options):
                if option.startswith(first_char):
                    prediction_idx = j
                    break

        if prediction_idx is None:
            for j, option in enumerate(options):
                if option in generated_text:
                    prediction_idx = j
                    break
        
        if prediction_idx is None:
            prediction_idx = 0
        
        # Prepare result
        result = {
            'id': datum['id'],
            "task": datum['task'],
            "question": datum['question'],
            "options": datum['options'],
            "answer": datum['answer'],
            "response": options[prediction_idx],
            "generated_text": generated_text,
            "is_correct": options[prediction_idx] == datum['answer']
        }
        
        # Save individual result
        with open(save_path, 'w') as wf:
            json.dump(result, wf)
        
        # Clear cache to save memory
        del past_key_values
        torch.cuda.empty_cache()

def evaluate_ovobench_results(results: list):
    task_to_counts = {}
    for result in results:
        task = result['task']
        if task not in task_to_counts:
            task_to_counts[task] = {'correct': 0, 'total': 0}
        task_to_counts[task]['total'] += 1
        if result['response'][:len(result['answer'])] == result['answer']:
            task_to_counts[task]['correct'] += 1
    rt_accs, bt_accs, fr_accs = [], [], []
    for task, counts in task_to_counts.items():
        print(f'{task}: {counts["correct"]}/{counts["total"]}={counts["correct"]/counts["total"]}')
        if task in ['OCR', 'ACR', 'ATR', 'STU', 'FPD', 'OJR']:
            rt_accs.append(counts['correct']/counts['total'])
        elif task in ['EPM', 'ASI', 'HLD']:
            bt_accs.append(counts['correct']/counts['total'])
        else:
            fr_accs.append(counts['correct']/counts['total'])
    if rt_accs:
        print(f'Real-Time Visual Perception avg.: {sum(rt_accs)}/{len(rt_accs)}={sum(rt_accs)/len(rt_accs)}')
    if bt_accs:
        print(f'Backward Tracing avg.: {sum(bt_accs)}/{len(bt_accs)}={sum(bt_accs)/len(bt_accs)}')
    if fr_accs:
        print(f'Forward Tracing avg.: {sum(fr_accs)}/{len(fr_accs)}={sum(fr_accs)/len(fr_accs)}')

if __name__ == '__main__':
    args = parse_args()
    
    # Setup save directory
    save_dir = os.path.join(args.output_dir, os.path.basename(args.model_name_or_path))
    save_path = save_dir + '2.jsonl'
    results = []

    # Debug: single-card process for quick testing
    if getattr(args, "debug", False):
        print("Running in single-card debug mode...")
        # 只用0号worker，直接调用worker函数
        ovobench_worker(
            0,
            model_name_or_path=args.model_name_or_path,
            save_dir=save_dir,
            num_workers=1,
            benchmark_path=args.benchmark_path,
            sample_size=args.sample_size,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        # Set multiprocessing start method
        multiprocessing.set_start_method('spawn', force=True)
        # Create worker function with partial arguments
        worker_fn = partial(
            ovobench_worker,
            model_name_or_path=args.model_name_or_path,
            save_dir=save_dir,
            num_workers=args.num_workers,
            benchmark_path=args.benchmark_path,
            sample_size=args.sample_size,
            max_new_tokens=args.max_new_tokens,
        )
        # Run distributed evaluation
        print(f"Starting distributed evaluation with {args.num_workers} workers...")
        local_mp(
            list(range(args.num_workers)),
            worker_fn,
            desc="ovobench_evaluation",
            num_workers=args.num_workers
        )
    
    # Combine results from individual JSON files to JSONL
    print("Combining results...")
    results = []
    with open(save_path, 'w') as wf:
        for file in os.listdir(save_dir):
            if file.endswith('.json'):
                try:
                    datum = json.load(open(os.path.join(save_dir, file)))
                    wf.write(json.dumps(datum) + '\n')
                    results.append(datum)
                except Exception as e:
                    print(f"Error reading {file}: {e}")
    
    # Save evaluation summary
    save_txt_path = save_path.replace('.jsonl', '.txt')
    save_function_print(
        evaluate_ovobench_results,
        save_txt_path,
        results
    )
    
    # Clean up individual JSON files
    print("Cleaning up temporary files...")
    shutil.rmtree(save_dir)
    
    print(f"Results saved to {save_path}")
    print(f"Evaluation summary saved to {save_txt_path}")

# Usage examples:
# CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python evaluation/ovobench/distributed_evaluate_ovobench_with_mrope.py --num_workers 1 --sample_size 100
# CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=$(pwd) python evaluation/ovobench/distributed_evaluate_ovobench_with_mrope.py --num_workers 4 --sample_size 100
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONPATH=$(pwd) python evaluation/ovobench/distributed_evaluate_ovobench_with_mrope.py --num_workers 8 --model_name_or_path chenjoya/LiveCC-7B-Instruct 