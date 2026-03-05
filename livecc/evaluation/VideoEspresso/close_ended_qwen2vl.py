import json, os, torch, functools, tqdm, random, sys
import numpy as np
import decord
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, logging, Qwen2VLForConditionalGeneration, AutoProcessor
from decord import cpu
from livecc_utils import _read_video_decord_plus, _spatial_resize_video
from qwen_vl_utils.vision_process import process_vision_info, smart_nframes, FPS

logger = logging.get_logger(__name__)

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
        vr = decord.VideoReader(video_path, num_threads=1, ctx=cpu(0))
    elif ele['remote_loader'] is not None:
        vr = decord.VideoReader(ele['remote_loader'](video_path), num_threads=1, ctx=cpu(0))
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

# import os, torch
# import numpy as np
# import decord # NOTE: import decord should be after torch, otherwise seg fault
# from transformers import logging
# from torchvision import transforms

# os.environ['FORCE_QWENVL_VIDEO_READER'] = 'decord+'
# os.environ['VIDEO_MAX_PIXELS'] = str(int(os.environ.get('VIDEO_MAX_PIXELS', 24576 * 28 * 28))) # increase this for streaming. 24576 * 28 * 28 = 19267584
# import qwen_vl_utils.vision_process
# qwen_vl_utils.vision_process.VIDEO_MIN_PIXELS = int(os.environ.get('VIDEO_MIN_PIXELS', 128 * 28 * 28)) # follow qwen2vl paper
# qwen_vl_utils.vision_process.FPS_MAX_FRAMES = int(os.environ.get('FPS_MAX_FRAMES', 480)) # decrease this for efficiency 
# from qwen_vl_utils.vision_process import (
#     FORCE_QWENVL_VIDEO_READER, VIDEO_TOTAL_PIXELS, FPS_MAX_FRAMES, VIDEO_MIN_PIXELS, VIDEO_MAX_PIXELS, FRAME_FACTOR, IMAGE_FACTOR, FPS,
#     smart_nframes, smart_resize
# )

def clean_options(option):
    cleaned_option = option.split("):", 1)[-1].strip()
    return cleaned_option

class VideoEspressoMCQDataset(Dataset):
    def __init__(self, remote_loader, path, with_evidence=0):
        with open(path, "r") as f:
            self.datums = json.load(f)
        
        self.src_video_dir = os.path.dirname("/2022233235/.cache/huggingface/hub/datasets--hshjerry0315--VideoEspresso-Test/snapshots/744dae23b48b5756ca48d52a47d63e6ccc102d4a/")
        self.with_evidence = with_evidence
        self.remote_loader = remote_loader
        
    def __len__(self):
        return len(self.datums)

    def __getitem__(self, i):
        # copy from https://github1s.com/hshjerry/VideoEspresso/blob/main/eval/close_ended.py
        datum = self.datums[i]
        video_path = datum["video_path"]

        options = datum["options"]
        options_prompt = ""
        option_list = ["\nA. ","B. ","C. ","D. "]
        for j, opt in enumerate(options):
            options_prompt += option_list[j] + clean_options(opt) + "\n"
        correct_answer = datum["correct_answer"]
        evidence = datum["evidence"]
        task = datum['task']
        question = datum['question']
        
        if self.with_evidence:
            query = f"Please finish the {task} task. Question: {question}. Your inference evidence is {evidence}. You have the following options: {options_prompt}. Select the answer and only give the option letters."
        else: 
            query = f"Please finish the {task} task. Question: {question}. You have the following options: {options_prompt}. Select the answer and only give the option letters."
        
        # rewrtie from LiveCC
        conversation = [{"role": "user", "content": []}]
        video_inputs = None

        video, _ = _read_may1fps_video_decord({'video': os.path.join(self.src_video_dir, datum['video_path']), 'remote_loader': self.remote_loader})
        video = _spatial_resize_video(video)
        conversation[0]['content'].append({"type": "video", "video": video})
        video_inputs = [video]
        conversation[0]['content'].append({"type": "text", "text": query})
        if video_inputs is None:
            for _ in range(10):
                try:
                    _, video_inputs = process_vision_info(conversation)
                    break
                except:
                    print(f"{_}-th process_vision_info failed. retry...")
        return conversation, video_inputs[0]

    def data_collator(self, batch, processor):
        conversations, video_inputs = zip(*batch)
        texts = processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=texts,
            images=None,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs

def preprocess_logits_for_metrics(logits, labels, strict_option_ids): 
    return torch.stack([logit[(logit[:, 0] != -100).nonzero().squeeze()[-1], strict_option_ids] for logit in logits]).argmax(dim=-1)

def mcq_predict(
    model, 
    processor, 
    benchmark_path: str, 
    options: list[str], 
    remote_loader: callable,
    question_prefix: str = '', 
    question_postfix: str = '\nPlease select the correct answer.', 
    answer_prefix: str = 'Answer:', 
    abcd_previous_str: str = ': ',
    use_liger_kernel: bool = True,
    per_device_eval_batch_size: int = 2,
    dataloader_num_workers: int = 4,
    with_evidence: bool = False,
):
    strict_option_ids = [processor.tokenizer(f'{abcd_previous_str}{_}').input_ids[-1] for _ in options] 
    dataset = VideoEspressoMCQDataset(remote_loader, benchmark_path, with_evidence=with_evidence)
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
        preprocess_logits_for_metrics=functools.partial(preprocess_logits_for_metrics, strict_option_ids=strict_option_ids),
    )
    letter_idxs_predictions = trainer.predict(dataset, ignore_keys=['past_key_values', 'hidden_states', 'attentions', 'rope_deltas']).predictions
    return letter_idxs_predictions, dataset.datums, trainer.args.process_index

def evaluate_VideoEspresso_results(results: list):
    task_to_counts = {}
    for result in results:
        task = result['task']
        if task not in task_to_counts:
            task_to_counts[task] = {'correct': 0, 'total': 0}
        task_to_counts[task]['total'] += 1
        if result['model_output'] in result['correct_answer']:
            task_to_counts[task]['correct'] += 1
    for task, counts in task_to_counts.items():
        print(f'{task}: {counts["correct"]}/{counts["total"]}={counts["correct"]/counts["total"]}')

if __name__ == '__main__':
    with_evidence = input("with evidence? (y/n): ")
    with_evidence = with_evidence == 'y'
    
    model_path = "chenjoya/LiveCC-7B-Instruct"
    # model_path = "Qwen/Qwen2-VL-7B-Instruct"
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", attn_implementation='flash_attention_2')
    processor = AutoProcessor.from_pretrained(model_path, padding_side='left')
    options = ['A', 'B', 'C', 'D']
    benchmark_path = '/2022233235/.cache/huggingface/hub/datasets--hshjerry0315--VideoEspresso-Test/snapshots/744dae23b48b5756ca48d52a47d63e6ccc102d4a/bench_hard.json' 
    letter_idxs_predictions, benchmark_datums, process_index = mcq_predict(
        model=model, processor=processor, benchmark_path=benchmark_path, 
        options=options, use_liger_kernel=True,
        answer_prefix = 'The answer is:\n', 
        abcd_previous_str = '\n',
        remote_loader=None,
        # per_device_eval_batch_size=1,
        # dataloader_num_workers=2,
    )
    if process_index == 0:
        results = []
        for datum, letter_idx_prediction in zip(benchmark_datums, letter_idxs_predictions):
            datum["model_output"] = options[letter_idx_prediction]
            results.append(datum)
        save_json_path = f'evaluation/VideoEspresso/results/{os.path.basename(model_path)}_{"with_evidence" if with_evidence else "no_evidence"}.json'
        os.makedirs(os.path.dirname(save_json_path), exist_ok=True)
        json.dump(results, open(save_json_path, 'w'), indent=4)
        save_txt_path = save_json_path.replace('.json', '.txt')

        save_function_print(
            evaluate_VideoEspresso_results,
            save_txt_path,
            results
        )
        json.dump(results, open(save_json_path, 'w'), indent=4)
# torchrun --standalone --nproc_per_node=8 evaluation/VideoEspresso/close_ended_qwen2vl.py