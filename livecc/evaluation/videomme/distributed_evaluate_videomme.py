import json
import os
import argparse
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

from evaluation.videomme.eval_your_results import eval_your_results
from evaluation.utils import save_function_print

import os, torch
import numpy as np
import decord # NOTE: import decord should be after torch, otherwise seg fault
from transformers import logging
from torchvision import transforms

os.environ['FORCE_QWENVL_VIDEO_READER'] = 'decord+'
os.environ['VIDEO_MAX_PIXELS'] = str(int(os.environ.get('VIDEO_MAX_PIXELS', 24576 * 28 * 28))) # increase this for streaming. 24576 * 28 * 28 = 19267584
import qwen_vl_utils.vision_process
from qwen_vl_utils import process_vision_info
qwen_vl_utils.vision_process.VIDEO_MIN_PIXELS = int(os.environ.get('VIDEO_MIN_PIXELS', 100 * 28 * 28)) # follow qwen2vl paper
qwen_vl_utils.vision_process.FPS_MAX_FRAMES = int(os.environ.get('FPS_MAX_FRAMES', 640)) # decrease this for efficiency 
from qwen_vl_utils.vision_process import (
    FORCE_QWENVL_VIDEO_READER, VIDEO_TOTAL_PIXELS, FPS_MAX_FRAMES, VIDEO_MIN_PIXELS, VIDEO_MAX_PIXELS, FRAME_FACTOR, IMAGE_FACTOR, FPS,
    smart_nframes, smart_resize
)

import torch, json, functools, tqdm, random, os
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, logging
class MCQDataset(Dataset):
    def __init__(self, remote_loader, path, question_prefix, question_postfix, answer_prefix, with_subtitles: bool = False, sample: int = None):
        lines = open(path).readlines()
        if sample is not None:
            random.seed(42)
            lines = random.sample(lines, sample)
        self.datums = [json.loads(line) for line in tqdm.tqdm(lines, desc='load datums')]
        if isinstance(self.datums[0], str):
            self.datums = [json.loads(datum) for datum in tqdm.tqdm(self.datums, desc='load datumsx2')]
        self.question_prefix = question_prefix
        self.question_postfix = question_postfix
        self.answer_prefix = answer_prefix
        self.remote_loader = remote_loader
        self.with_subtitles = with_subtitles
        
    def __len__(self):
        return len(self.datums)
        
    def __getitem__(self, i):
        datum = self.datums[i]
        query = self.question_prefix + datum['question'] + '\n' + '\n'.join(datum['options']) + self.question_postfix
        conversation = [{"role": "user", "content": []}]
        if 'video' in datum:
            conversation[0]['content'].append(
                {"type": "video", "video": datum['video'], 'remote_loader': self.remote_loader},
            )
            if 'video_start' in datum:
                conversation[0]['content'][-1]['video_start'] = datum['video_start']
            if 'video_end' in datum:
                conversation[0]['content'][-1]['video_end'] = datum['video_end']
        if not self.with_subtitles:
            conversation[0]['content'].append({"type": "text", "text": query})
        else:
            query = f"This video's subtitles are listed below:\n{datum['subtitles']}\nAccording to the video and subtitles, "  + query
            conversation[0]['content'].append({"type": "text", "text": query})
        return conversation

    def data_collator(self, conversations, processor):
        texts = processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
        texts = [text + self.answer_prefix for text in texts]
        image_inputs, video_inputs = process_vision_info(conversations)
                
        inputs = processor(
            text=texts,
            images=None,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
                
        return inputs

def preprocess_logits_for_metrics(logits, labels, strict_letter_ids): 
    return torch.stack([logit[(logit[:, 0] != -100).nonzero().squeeze()[-1], strict_letter_ids] for logit in logits]).argmax(dim=-1)

def mcq_predict(
    model, 
    processor, 
    benchmark_path: str, 
    letters: list[str], 
    remote_loader: callable = None,
    question_prefix: str = '', 
    question_postfix: str = '\nPlease select the correct answer.', 
    answer_prefix: str = 'Answer:', 
    abcd_previous_str: str = ': ',
    use_liger_kernel: bool = True,
    per_device_eval_batch_size: int = 2,
    dataloader_num_workers: int = 4,
    with_subtitles: bool = False
):
    strict_letter_ids = [processor.tokenizer(f'{abcd_previous_str}{_}').input_ids[-1] for _ in letters] 
    dataset = MCQDataset(remote_loader, benchmark_path, question_prefix=question_prefix, question_postfix=question_postfix, answer_prefix=answer_prefix, with_subtitles=with_subtitles)
    trainer = Trainer(
        model=model, 
        args=TrainingArguments(
            output_dir='outputs/', do_predict=True, 
            per_device_eval_batch_size=per_device_eval_batch_size, 
            dataloader_num_workers=dataloader_num_workers, 
            report_to='none', use_liger_kernel=use_liger_kernel
        ), 
        data_collator=functools.partial(dataset.data_collator, processor=processor),
        processing_class=processor,
        preprocess_logits_for_metrics=functools.partial(preprocess_logits_for_metrics, strict_letter_ids=strict_letter_ids),
    )
    letter_idxs_predictions = trainer.predict(dataset, ignore_keys=['past_key_values', 'hidden_states', 'attentions', 'rope_deltas']).predictions
    return letter_idxs_predictions, dataset.datums, trainer.args.process_index




def main():
    parser = argparse.ArgumentParser(
        description="Distributed evaluation for VideoMME models"
    )
    parser.add_argument(
        "--model_name_or_path", type=str, required=True,
        help="Path or identifier of the pretrained model"
    )
    parser.add_argument(
        "--benchmark_path", type=str, required=True,
        help="Path to the benchmark JSONL file"
    )
    parser.add_argument(
        "--with_subtitles", action="store_true",
        help="Flag to indicate evaluation on subtitles-enabled benchmark"
    )
    args = parser.parse_args()

    # Load model and processor
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype="auto",
        attn_implementation="flash_attention_2"
    )
    processor_name = (
        args.model_name_or_path
        if args.model_name_or_path != 'Qwen/Qwen2-VL-7B'
        else 'Qwen/Qwen2-VL-7B-Instruct'
    )
    processor = AutoProcessor.from_pretrained(
        processor_name,
        padding_side='left'
    )

    # Run distributed prediction
    letter_idxs_predictions, benchmark_datums, process_index = mcq_predict(
        model=model,
        processor=processor,
        benchmark_path=args.benchmark_path,
        letters=['A', 'B', 'C', 'D'],
        use_liger_kernel='LiveCC' in args.model_name_or_path,
        per_device_eval_batch_size=1,
        with_subtitles=args.with_subtitles
    )

    # Only process rank 0 for result aggregation and saving
    if process_index == 0:
        video_id_to_results = {}
        for datum, letter_idx_prediction in zip(
            benchmark_datums, letter_idxs_predictions
        ):
            vid = datum['video_id']
            if vid not in video_id_to_results:
                video_id_to_results[vid] = {
                    'video_id': vid,
                    'duration': datum['duration'],
                    'domain': datum['domain'],
                    'sub_category': datum['sub_category'],
                    'questions': [],
                }
            video_id_to_results[vid]['questions'].append({
                "question_id": datum['question_id'],
                "task_type": datum['task_type'],
                "question": datum['question'],
                "options": datum['options'],
                "answer": datum['answer'],
                "response": datum['options'][letter_idx_prediction],
            })

        results = list(video_id_to_results.values())

        # Determine output paths
        suffix = 'with_subtitles' if args.with_subtitles else 'no_subtitles'
        out_dir = f'evaluation/videomme/results'
        os.makedirs(out_dir, exist_ok=True)
        save_json_path = os.path.join(
            out_dir,
            f"{os.path.basename(args.model_name_or_path)}_{suffix}.json"
        )

        # Save JSON
        with open(save_json_path, 'w') as f:
            json.dump(results, f)

        # Save evaluation text report
        save_txt_path = save_json_path.replace('.json', '.txt')
        save_function_print(
            eval_your_results,
            save_txt_path,
            save_json_path,
            video_types=['short', 'medium', 'long'],
            return_categories_accuracy=True,
            return_sub_categories_accuracy=True,
            return_task_types_accuracy=True,
        )

if __name__ == '__main__':
    main()

# PYTHONPATH=$(pwd) torchrun --standalone --nproc_per_node=8 evaluation/videomme/distributed_evaluate_videomme.py --model_name_or_path Qwen/Qwen2-VL-7B-Instruct --benchmark_path videomme_with_subtitles.jsonl