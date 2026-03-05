import os
import json
import tqdm
import shutil
import argparse
import multiprocessing
from functools import partial
from utils.multiprocessor import local_mp

import functools, torch
from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl
apply_liger_kernel_to_qwen2_vl()
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, LogitsProcessor, logging
from livecc_utils import prepare_multiturn_multimodal_inputs_for_generation, get_smart_resized_clip, get_smart_resized_video_reader, _read_video_decord_plus, _spatial_resize_video
from qwen_vl_utils import process_vision_info

def parse_args():
    parser = argparse.ArgumentParser(
        description="Distributed LiveCC generation over the LiveSports-3K CC split"
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
        default="evaluation/vcrbench/qwen2vl",
        help="Directory to write generated JSON outputs"
    )
    parser.add_argument(
        "--video_source_dir", type=str,
        default="/2022233235/.cache/huggingface/hub/datasets--VLM-Reasoning--VCR-Bench/snapshots/1a7350d01fb6d8f58e6cce8d00389194faffc3d5/v1/videos",
        help="Directory to write generated JSON outputs"
    )
    return parser.parse_args()

def livecc_worker(
    device_id: int,
    model_name_or_path: str,
    save_dir: str,
    num_workers: int,
    video_source_dir: str
):
    
    # 1. load model
    model_path = "chenjoya/LiveCC-7B-Instruct"
    model_path = "Qwen/Qwen2-VL-7B-Instruct"
    if "Qwen2.5" in model_name_or_path:
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name_or_path, torch_dtype="auto", attn_implementation='flash_attention_2', device_map=f'cuda:{device_id}')
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_name_or_path, torch_dtype="auto", attn_implementation='flash_attention_2',  device_map=f'cuda:{device_id}')
    processor = AutoProcessor.from_pretrained(model_name_or_path, padding_side='left')
    
    
    # 2. load dataset json
    ds = json.load(open('/2022233235/videollm-online/EyeWO2/data/vcrbench_reasoning.json'))
    idxs = list(range(len(ds)))
    idxs_on_device = idxs[device_id::num_workers]
    '''{
        "id": 169,
        "video_path": "video/MLVU/ego_26.mp4",
        "duration": 375,
        "dimension": "Temporal Spatial Reasoning",
        "question": "In the video, where is the location where the first-person perspective shooter stands for the longest time?",
        "multiple-choice": true,
        "choices": {
            "A": "On the stool",
            "B": "In front of the table",
            "C": "Outside the room",
            "D": "Beside the bicycle"
        },
        "answer": "D",
        "reasoning": [
            "1. 0:00-0:04, the cameraman stands in the aisle next to the workbench and opens the tool cabinet.",
            "2. 0:05-0:32, the person filming walks to the bicycle and starts repairing the rear wheel with a drill; 0:33-0:42, the person filming walks back to the workbench to select a drill bit; 0:43-0:53, the person filming walks to the bicycle again to repair the rear wheel.",
            "3. 0:53-1:03, The cameraman walks back to the workbench.",
            "4. 1:03-04:59, the person filming continuously repairs beside the bicycle. Although there was some movement in between, most of the time was still spent beside the bicycle.",
            "5. From 05:02 to 05:40, the cameraperson walked back to the workbench and used various tools and lubricants.",
            "6. 05:41-06:14, the person filming is standing next to the bicycle again, repairing the bicycle.",
            "7. Therefore, apart from a small amount of time spent away from the bicycle to select tools, the photographer is mostly beside the bicycle repairing it."
        ],
        "parser_reasoning": [
            {
    ...
            }
        ]
    }'''
    
   
    # Prepare temporary save folder for this model
    os.makedirs(save_dir, exist_ok=True)

    for idx in tqdm.tqdm(idxs_on_device, desc=f"Device {device_id}", total=len(idxs_on_device)):
        save_path = os.path.join(save_dir, f"{idx}.json")
        if os.path.exists(save_path):
            continue

        record = ds[idx]
        
        # 3. construct prompt 
        video_path = os.path.join(video_source_dir, record['video_path'])
        parser_reasoning = record['parser_reasoning']
        half_key = parser_reasoning[len(parser_reasoning) // 2]
        clip, _ = _read_video_decord_plus({'video': video_path, 'video_start': 0, 'video_end': half_key['end_time'], 'remote_loader': None})
        clip = _spatial_resize_video(clip)
        
        
        if record['multiple-choice']:
            question_prefix = ''
            question_postfix = '\nPlease select the correct answer.'
            query = question_prefix + record['question'] + '\n' + '\n'.join(record['choices']) + question_postfix
        else:
            question_prefix = ''
            question_postfix = ''
            query = question_prefix + record['question'] + question_postfix
        
        conversation = [{"role": "user", "content": []}]
        conversation[0]['content'].append({"type": "video", "video": clip})
        video_inputs = [clip]
        conversation[0]['content'].append({"type": "text", "text": query})
        
        texts = processor.apply_chat_template([conversation], tokenize=False, add_generation_prompt=True)
        answer_prefix = 'The answer is:\n'
        texts = [text + answer_prefix for text in texts]
        
        # 4. get response
        inputs = processor(
            text=texts,
            images=None,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=16, return_dict_in_generate=True,)
            response = processor.decode(outputs.sequences[0, inputs.input_ids.size(1):], skip_special_tokens=True)
        
        record['response'] = response
            
        # 5. 评估正确性
        if 'multiple-choice' in record and record['multiple-choice']:
            # 选择题：比较response第一个非空字符和answer
            response_first = next((c for c in response if c.strip()), None)
            record['is_correct'] = (response_first is not None and response_first.upper() == str(record['answer']).strip().upper())
        else:
            # 非选择题：用DeepSeek评估
            if not hasattr(livecc_worker, '_deepseek_evaluator'):
                # 只初始化一次
                livecc_worker._deepseek_evaluator = DeepSeekCorrectnessEvaluator(api_key="sk-43a08cfb3ae64b6288ee67db8009c8ca")
            evaluator = livecc_worker._deepseek_evaluator
            record['is_correct'] = evaluator.evaluate(record['question'], record['answer'], response)

        with open(save_path, 'w') as wf:
            json.dump(record, wf)

if __name__ == "__main__":
    args = parse_args()
    multiprocessing.set_start_method('spawn', force=True)
    save_dir = os.path.join(args.output_dir, os.path.basename(args.model_name_or_path))
    worker_fn = partial(
        livecc_worker,
        model_name_or_path=args.model_name_or_path,
        save_dir=save_dir,
        video_source_dir=args.video_source_dir,
        num_workers=args.num_workers,
    )
    local_mp(
        list(range(args.num_workers)),
        worker_fn,
        desc="livecc_generation",
        num_workers=args.num_workers
    )
    # jsons -> jsonl
    save_path = save_dir + '.jsonl'
    with open(save_path, 'w') as wf:
        for file in os.listdir(save_dir):
            datum = json.load(open(os.path.join(save_dir, file))) 
            wf.write(json.dumps(datum) + '\n')
    # remove save_dir
    shutil.rmtree(save_dir)

# CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONPATH=$(pwd) python evaluation/vcrbench/distributed_generate_livecc.py  --num_workers 4