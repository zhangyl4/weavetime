import json, os, torch, functools, tqdm, random, sys
import numpy as np
import decord
from collections import defaultdict
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, logging, Qwen2VLForConditionalGeneration, AutoProcessor

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
        vr = decord.VideoReader(video_path, num_threads=1)
    elif ele['remote_loader'] is not None:
        vr = decord.VideoReader(ele['remote_loader'](video_path), num_threads=1)
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
    
    # 修正：确保索引不越界
    nframes_idxs = np.clip(nframes_idxs, 0, total_frames - 1)
    
    if clip_idxs is not None:
        # 确保clip_idxs有足够的元素
        if len(clip_idxs) > 0:
            clip_idxs = clip_idxs[nframes_idxs]
        else:
            clip_idxs = nframes_idxs
    else:
        clip_idxs = nframes_idxs
    
    # 修正：确保clip_idxs不为空且索引有效
    if len(clip_idxs) == 0:
        clip_idxs = np.array([0])
    
    try:
        clip = torch.from_numpy(vr.get_batch(clip_idxs).asnumpy()).permute(0, 3, 1, 2)  # Convert to TCHW format
    except Exception as e:
        logger.warning(f"Error reading video batch: {e}")
        # 如果获取batch失败，尝试读取单个帧
        if len(clip_idxs) > 0:
            clip = torch.from_numpy(vr.get_batch([clip_idxs[0]]).asnumpy()).permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Failed to read video: {video_path}")
    
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
# qwen_vl_utils.vision_process.VIDEO_MIN_PIXELS = int(os.environ.get('VIDEO_MIN_PIXELS', 100 * 28 * 28)) # follow qwen2vl paper
# qwen_vl_utils.vision_process.FPS_MAX_FRAMES = int(os.environ.get('FPS_MAX_FRAMES', 640)) # decrease this for efficiency 
# from qwen_vl_utils.vision_process import (
#     FORCE_QWENVL_VIDEO_READER, VIDEO_TOTAL_PIXELS, FPS_MAX_FRAMES, VIDEO_MIN_PIXELS, VIDEO_MAX_PIXELS, FRAME_FACTOR, IMAGE_FACTOR, FPS,
#     smart_nframes, smart_resize
# )
# os.environ['FORCE_QWENVL_VIDEO_READER'] = 'decord+'
# os.environ['VIDEO_MAX_PIXELS'] = str(int(16384 * 28 * 28)) # increase this for streaming. 24576 * 28 * 28 = 19267584
# import qwen_vl_utils.vision_process
# qwen_vl_utils.vision_process.VIDEO_MIN_PIXELS = int(os.environ.get('VIDEO_MIN_PIXELS', 128 * 28 * 28)) # follow qwen2vl paper
# qwen_vl_utils.vision_process.FPS_MAX_FRAMES = int(os.environ.get('FPS_MAX_FRAMES', 768)) # decrease this for efficiency 
# from qwen_vl_utils.vision_process import (
#     FORCE_QWENVL_VIDEO_READER, VIDEO_TOTAL_PIXELS, FPS_MAX_FRAMES, VIDEO_MIN_PIXELS, VIDEO_MAX_PIXELS, FRAME_FACTOR, IMAGE_FACTOR, FPS,
#     smart_nframes, smart_resize
# )

# def _spatial_resize_video(video: torch.Tensor, nframes: int = None): # 
#     if not nframes:
#         nframes, _, height, width = video.shape
#     else:
#         height, width = video.shape[2:]
#     max_pixels = max(min(VIDEO_MAX_PIXELS, VIDEO_TOTAL_PIXELS / nframes * FRAME_FACTOR), int(VIDEO_MIN_PIXELS * 1.05))
#     resized_height, resized_width = smart_resize( 
#         height,
#         width,
#         factor=IMAGE_FACTOR,
#         min_pixels=VIDEO_MIN_PIXELS,
#         max_pixels=max_pixels,
#     )
#     video = transforms.functional.resize(
#         video,
#         [resized_height, resized_width],
#         interpolation=transforms.InterpolationMode.BICUBIC,
#         antialias=True,
#     ).float() # need float?
#     return video

PROMPT_TEMPLATE = '''You are an advanced video question-answering AI assistant. You have been provided with some frames from the video and a multiple-choice question related to the video. Your task is to carefully analyze the video and provide the best answer to question, choosing from the four options provided. Respond with only the letter (A, B, C, or D) of the correct option.

Question: {}

Options:
{}
{}
{}
{}'''

class StreamingBenchMCQDataset(Dataset):
    
    omi_task = ['Emotion Recognition', 'Scene Understanding', 'Source Discrimination', 'Multimodal Alignment']
    ct_task = ['Misleading Context Understanding', 'Anomaly Context Understanding', 'Sequential Question Answering', 'Proactive Output']
    
    
    def __init__(self, remote_loader, path, question_prefix, question_postfix, answer_prefix, sample: int = None):
        lines = open(path).readlines()
        if sample is not None:
            random.seed(42)
            lines = random.sample(lines, sample)
        self.datums = [json.loads(line) for line in tqdm.tqdm(lines, desc='load datums')]
        if isinstance(self.datums[0], str):
            self.datums = [json.loads(datum) for datum in tqdm.tqdm(self.datums, desc='load datumsx2')]
        self.datums = [datum for datum in self.datums if datum['task'] not in self.omi_task and datum['task'] not in self.ct_task]
        self.src_video_dir = os.path.dirname("/2022233235/.cache/huggingface/hub/datasets--mjuicem--StreamingBench/snapshots/48872fa707124474ce4c5172ddc58efb8bc88058/real/")
        self.question_prefix = question_prefix
        self.question_postfix = question_postfix
        self.answer_prefix = answer_prefix
        self.remote_loader = remote_loader
        
    def __len__(self):
        return len(self.datums)

    def __getitem__(self, i):
        datum = self.datums[i]
        conversation = [{"role": "user", "content": []}]
        video_inputs = None
        if datum['task'] == 'Proactive Output': # 'REC', 'SSR', 'CRR' have already been chunked
            query = datum['question']
            video, _ = _read_may1fps_video_decord({'video': os.path.join(self.src_video_dir, datum['video']), 'video_end': datum['video_end'],'remote_loader': self.remote_loader})
        else:
            query = PROMPT_TEMPLATE.format(datum['question'], *datum['options'])
            video, _ = _read_may1fps_video_decord({'video': os.path.join(self.src_video_dir, datum['video']), 'video_end': datum['video_end'],  'remote_loader': self.remote_loader})
        video = _spatial_resize_video(video)
        conversation[0]['content'].append({"type": "video", "video": video})
        video_inputs = [video]
        conversation[0]['content'].append({"type": "text", "text": query})
        
        return conversation, video_inputs[0]

    def data_collator(self, batch, processor):
        conversations, video_inputs = zip(*batch)
        texts = processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
        texts = [text + self.answer_prefix for text in texts]
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
):
    strict_option_ids = [processor.tokenizer(f'{abcd_previous_str}{_}').input_ids[-1] for _ in options] 
    dataset = StreamingBenchMCQDataset(remote_loader, benchmark_path, question_prefix=question_prefix, question_postfix=question_postfix, answer_prefix=answer_prefix)
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

        
def evaluate_streambench_results(results: list):
    # Initialize counters
    stats = defaultdict(lambda: defaultdict(int))

    omi_task = ['Emotion Recognition', 'Scene Understanding', 'Source Discrimination', 'Multimodal Alignment']
    ct_task = ['Misleading Context Understanding', 'Anomaly Context Understanding', 'Sequential Question Answering', 'Proactive Output']
    
    # Process each entry in the JSON data
    total = 0
    for result in results:
        task = result['task']
        if task == "Proactive Output":
            if 'response' in result:
                # 修正：使用result而不是question，并且需要检查数据结构
                if 'ground_truth_time_stamp' in result:
                    ground_truth_timestamp = result["ground_truth_time_stamp"]
                    ground_truth_time = sum(int(x) * 60 ** i for i, x in enumerate(reversed(ground_truth_timestamp.split(":"))))
                    
                    # 修正：检查response是否为列表格式
                    if isinstance(result['response'], list) and len(result['response']) > 0:
                        last_time = result['response'][-1].get("time", 0)
                        last_answer = result['response'][-1].get("content", "")
                    else:
                        # 如果response不是列表格式，直接使用
                        last_time = 0
                        last_answer = result['response']

                    total += 1
                    stats[task]["total"] += 1
                    if -2 <= last_time - ground_truth_time <= 2:
                        stats[task]["time_correct"] += 1
                        if result.get("ground_truth_output", "") in last_answer:
                            stats[task]["answer_correct"] += 1
                
        else:
            if 'response' in result:
                total += 1
                stats[task]['total'] += 1
                if result['response'][:len(result['answer'])] == result['answer']:
                    stats[task]['correct'] += 1
    print(f"{total} items have been statisticed")
    
    rt_accs, om_accs, ct_accs = [], [], []
    for task_type, counts in stats.items():
        if task_type == "Proactive Output":
            counts["time_accuracy"] = counts["time_correct"] / counts["total"] if counts["total"] > 0 else 0
            counts["answer_accuracy"] = counts["answer_correct"] / counts["total"] if counts["total"] > 0 else 0
            print(f'Proactive Output: {counts["time_correct"]}/{counts["total"]}={counts["time_correct"]/counts["total"]}')
            print(f'Proactive Output: {counts["answer_correct"]}/{counts["total"]}={counts["answer_correct"]/counts["total"]}')
            ct_accs.append(counts["time_accuracy"])
        else:
            counts["accuracy"] = counts["correct"] / counts["total"] if counts["total"] > 0 else 0
            if task_type in omi_task:
                om_accs.append(counts["accuracy"])
            elif task_type in ct_task:
                ct_accs.append(counts["accuracy"])
            else:
                rt_accs.append(counts["accuracy"])
            print(f'{task_type}: {counts["correct"]}/{counts["total"]}={counts["correct"]/counts["total"]}')
    
    if rt_accs:
        print(f'Real-Time Visual Perception avg.: {sum(rt_accs)}/{len(rt_accs)}={sum(rt_accs)/len(rt_accs)}')
    if om_accs:
        print(f'Omi avg.: {sum(om_accs)}/{len(om_accs)}={sum(om_accs)/len(om_accs)}')
    if ct_accs:
        print(f'Ct avg.: {sum(ct_accs)}/{len(ct_accs)}={sum(ct_accs)/len(ct_accs)}')
    
    


if __name__ == '__main__':
    model_path = "chenjoya/LiveCC-7B-Instruct"
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", attn_implementation='flash_attention_2')
    processor = AutoProcessor.from_pretrained(model_path, padding_side='left')
    options = ['No', 'Yes', 'A', 'B', 'C', 'D']
    benchmark_path = 'streamingbench-rtvu-formatted.jsonl' 
    letter_idxs_predictions, benchmark_datums, process_index = mcq_predict(
        model=model, processor=processor, benchmark_path=benchmark_path, 
        options=options, use_liger_kernel='LiveCC' in model_path,
        answer_prefix = '\n\nThe best option is:\n', 
        abcd_previous_str = '\n',
        remote_loader=None,
        # per_device_eval_batch_size=1,
        # dataloader_num_workers=2,
    )
    if process_index == 0:
        results = []
        for datum, letter_idx_prediction in zip(benchmark_datums, letter_idxs_predictions):
            results.append({
                'id': datum['id'],
                "task": datum['task'],
                "question": datum['question'],
                "options": datum['options'],
                "answer": datum['answer'],
                "response": options[letter_idx_prediction],
            })
        save_json_path = f'evaluation/streamingbench/results/{os.path.basename(model_path)}.json'
        os.makedirs(os.path.dirname(save_json_path), exist_ok=True)
        json.dump(results, open(save_json_path, 'w'))
        save_txt_path = save_json_path.replace('.json', '.txt')
        save_function_print(
            evaluate_streambench_results,
            save_txt_path,
            results
        )

#CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 evaluation/streamingbench/distributed_evaluate_streamingbench.py