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

import openai

EVALUATOR_PROMPT = [
    {"role": "system", "content": (
        "You are an evaluator for a video question answering system. Your task is to judge whether the predicted answer is correct compared to the ground truth answer.\n"
        "Respond with 'True' if the predicted answer is correct, and 'False' if it is incorrect.\n"
        "Here are some examples to guide you:")},
    {"role": "user", "content": "Question: what is the category of the object I hold?\nGround Truth Answer: The object you hold is a computer mouse.\nPredicted Answer: It is a computer mouse."},
    {"role": "assistant", "content": "True"},
    {"role": "user", "content": "Question: Can you remind me when the state of the trunk changes? \nGround Truth Answer: The trunk is opened by the observer.\nPredicted Answer: The trunk is closed."},
    {"role": "assistant", "content": "False"},
]

class DeepSeekCorrectnessEvaluator:
    def __init__(self, api_key):
        self.client = openai.OpenAI(
            api_key="sk-43a08cfb3ae64b6288ee67db8009c8ca",
            base_url="https://api.deepseek.com",
        )
        self.conversation = EVALUATOR_PROMPT

    def evaluate(self, question, gold_answer, pred_answer):
        messages = self.conversation + [
            {"role": "user", "content": f"Question: {question}\nGround Truth Answer: {gold_answer}\nPredicted Answer: {pred_answer}"}
        ]
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
            )
            content = response.choices[0].message.content.strip().lower()
            if content.startswith('true'):
                return True
            elif content.startswith('false'):
                return False
            else:
                # fallback: treat as incorrect
                return False
        except Exception as e:
            print(f"Error in DeepSeek evaluation: {e}")
            return False

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

    
    # 2. load dataset json
    ds = []
    with open('/2022233235/videollm-online/livecc/evaluation/vcrbench/qwen2vl/Qwen2-VL-7B-Instruct.jsonl', 'r') as f:
        for line in f:
            ds.append(json.loads(line.strip()))
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
        response: ""
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
        response = record['response']

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
    save_path = save_dir + 'eval_results.jsonl'
    with open(save_path, 'w') as wf:
        for file in os.listdir(save_dir):
            datum = json.load(open(os.path.join(save_dir, file))) 
            wf.write(json.dumps(datum) + '\n')
    # remove save_dir
    shutil.rmtree(save_dir)

# PYTHONPATH=$(pwd) python evaluation/vcrbench/distributed_evalation.py