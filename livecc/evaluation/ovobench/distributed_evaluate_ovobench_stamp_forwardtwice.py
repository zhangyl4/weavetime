import json, os, torch, functools, tqdm, random, sys
import numpy as np
import decord
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, logging, Qwen2VLForConditionalGeneration, AutoProcessor
from transformers.cache_utils import DynamicCache
import typing

from livecc_utils import _read_video_decord_plus, _spatial_resize_video
from qwen_vl_utils.vision_process import process_vision_info, smart_nframes, FPS

logger = logging.get_logger(__name__)

class SimpleMRopeDynamicCache(DynamicCache):
    def __init__(self, mrope_section: list[int], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mrope_section = mrope_section
        self.original_cos_cache = []
        self.original_sin_cache = []
        
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """Store cos and sin for future rerotation."""
        cos = cache_kwargs.get("cos") if cache_kwargs else None
        sin = cache_kwargs.get("sin") if cache_kwargs else None
        
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
            
        if cos is not None and sin is not None and layer_idx == 0:
            if len(self.original_cos_cache) == 0:
                self.original_cos_cache = cos.clone()
                self.original_sin_cache = sin.clone()
            else:
                self.original_cos_cache = torch.cat([self.original_cos_cache, cos], dim=-2)
                self.original_sin_cache = torch.cat([self.original_sin_cache, sin], dim=-2)
        
        return super().update(key_states, value_states, layer_idx, cache_kwargs)
    
    def refresh_cache(self, inputs: dict, model):
        """Refresh cache by keeping only text-related tokens."""
        if len(self.key_cache) == 0:
            return inputs
            
        # Find text token positions
        selected_indices = self._find_text_tokens(inputs['input_ids'])
        
        # Apply selection to KV cache
        self._apply_kv_selection(selected_indices)
        
        # Recompute rotation for selected keys
        if len(selected_indices) > 0:
            self._rerotate_selected_keys(inputs, model)
            
        return inputs
    
    def _find_text_tokens(self, input_ids: torch.Tensor) -> list:
        """Find positions of text tokens (excluding vision tokens)."""
        if input_ids.dim() > 1:
            input_ids = input_ids.squeeze(0)
            
        text_positions = []
        in_vision = False
        
        for i, token_id in enumerate(input_ids):
            token_id = token_id.item()
            if token_id == 151652:  # <|vision_start|>
                in_vision = True
            elif token_id == 151653:  # <|vision_end|>
                in_vision = False
            elif not in_vision:
                text_positions.append(i)
                
        return text_positions
    
    def _apply_kv_selection(self, selected_indices: list):
        """Keep only selected indices in KV cache."""
        if not selected_indices:
            return
            
        for layer_idx in range(len(self.key_cache)):
            if len(self.key_cache[layer_idx]) > 0:
                self.key_cache[layer_idx] = self.key_cache[layer_idx][:, :, selected_indices, :]
                self.value_cache[layer_idx] = self.value_cache[layer_idx][:, :, selected_indices, :]
        
        self._seen_tokens = len(selected_indices)
        
        if len(self.original_cos_cache) > 0:
            self.original_cos_cache = self.original_cos_cache[:, :, selected_indices, :]
            self.original_sin_cache = self.original_sin_cache[:, :, selected_indices, :]
    
    def _rerotate_selected_keys(self, inputs: dict, model):
        """Rerotate selected keys using stored cos/sin values."""
        if len(self.original_cos_cache) == 0 or len(self.key_cache) == 0:
            return
            
        mrope_section = self.mrope_section * 2
        
        # Get rotary embeddings
        rotary_emb = model.model.rotary_emb
        position_ids = torch.arange(self._seen_tokens, device=self.key_cache[0].device)
        
        # Compute new cos/sin
        shifted_cos, shifted_sin = rotary_emb(self.key_cache[0], position_ids)
        
        original_cos, original_sin = self.original_cos_cache, self.original_sin_cache
        
        def mrope_cat(x):
            return torch.cat([m[i % 3] for i, m in enumerate(x.split(mrope_section, dim=-1))], dim=-1).unsqueeze(1)
            
        original_cos = mrope_cat(original_cos)
        shifted_cos = mrope_cat(shifted_cos)
        original_sin = mrope_cat(original_sin)
        shifted_sin = mrope_cat(shifted_sin)
        
        rerotation_cos = original_cos * shifted_cos + original_sin * shifted_sin
        rerotation_sin = -original_sin * shifted_cos + original_cos * shifted_sin
        
        rerotation_cos = rerotation_cos.to(self.key_cache[0].dtype)
        rerotation_sin = rerotation_sin.to(self.key_cache[0].dtype)
        
        for layer_idx in range(len(self.key_cache)):
            if len(self.key_cache[layer_idx]) > 0:
                current_keys = self.key_cache[layer_idx]
                rerotated_keys = self._apply_key_rotary_pos_emb(
                    current_keys,
                    rerotation_cos,
                    rerotation_sin
                )
                self.key_cache[layer_idx] = rerotated_keys
                
    @staticmethod
    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_key_rotary_pos_emb(
        self, key_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        rotated_key_states = (key_states * cos) + (self._rotate_half(key_states) * sin)
        return rotated_key_states

class Qwen2VLForConditionalGenerationForwardTwice(Qwen2VLForConditionalGeneration):
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        """
        Forward twice:
        1. First, forward all input_ids except the last token, get past_key_values.
        2. Then, forward the full input_ids, passing in past_key_values.
        """
        # Step 1: Forward all except last token
        if input_ids is None:
            raise ValueError("input_ids must be provided")
        if input_ids.dim() == 2:
            # batch, seq
            input_ids_first = input_ids[:, :-1]
            input_ids_full = input_ids
            if attention_mask is not None:
                attention_mask_first = attention_mask[:, :-1]
            else:
                attention_mask_first = None
        else:
            # single sequence
            input_ids_first = input_ids[:-1]
            input_ids_full = input_ids
            if attention_mask is not None:
                attention_mask_first = attention_mask[:-1]
            else:
                attention_mask_first = None

        # Forward pass 1: all except last token
        outputs_first = super().forward(
            input_ids=input_ids_first,
            attention_mask=attention_mask_first,
            use_cache=True,
            **{k: v for k, v in kwargs.items() if k not in ['input_ids', 'attention_mask', 'use_cache']}
        )
        past_key_values = outputs_first.past_key_values if hasattr(outputs_first, "past_key_values") else outputs_first.get("past_key_values", None)

        # Forward pass 2: full input_ids, with past_key_values
        outputs_second = super().forward(
            input_ids=input_ids_full,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **{k: v for k, v in kwargs.items() if k not in ['input_ids', 'attention_mask', 'past_key_values']}
        )
        return outputs_second

def _read_may1fps_video_decord(ele: dict, return_pts: bool = False):
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
    if return_pts:
        vr.get_frame_timestamp(0)
        video_pts = vr._frame_pts[:,1]
        return clip, sample_fps, video_pts[clip_idxs]
    else:
        return clip, sample_fps

def save_function_print(function: callable, save_path: str, *args, **kwargs):
    original_stdout = sys.stdout
    try:
        with open(save_path, 'w') as f:
            sys.stdout = f  
            function(*args, **kwargs)          
    finally:
        sys.stdout = original_stdout 

import os, torch
import numpy as np
import decord # NOTE: import decord should be after torch, otherwise seg fault
from transformers import logging
from torchvision import transforms

os.environ['FORCE_QWENVL_VIDEO_READER'] = 'decord+'
os.environ['VIDEO_MAX_PIXELS'] = str(int(os.environ.get('VIDEO_MAX_PIXELS', 24576 * 28 * 28))) # increase this for streaming. 24576 * 28 * 28 = 19267584
import qwen_vl_utils.vision_process
qwen_vl_utils.vision_process.VIDEO_MIN_PIXELS = int(os.environ.get('VIDEO_MIN_PIXELS', 100 * 28 * 28)) # follow qwen2vl paper
qwen_vl_utils.vision_process.FPS_MAX_FRAMES = int(os.environ.get('FPS_MAX_FRAMES', 768)) # decrease this for efficiency 
from qwen_vl_utils.vision_process import (
    FORCE_QWENVL_VIDEO_READER, VIDEO_TOTAL_PIXELS, FPS_MAX_FRAMES, VIDEO_MIN_PIXELS, VIDEO_MAX_PIXELS, FRAME_FACTOR, IMAGE_FACTOR, FPS,
    smart_nframes, smart_resize
)

# os.environ['FORCE_QWENVL_VIDEO_READER'] = 'decord+'
# os.environ['VIDEO_MAX_PIXELS'] = str(int(16384 * 28 * 28)) # increase this for streaming. 24576 * 28 * 28 = 19267584
# import qwen_vl_utils.vision_process
# qwen_vl_utils.vision_process.VIDEO_MIN_PIXELS = int(os.environ.get('VIDEO_MIN_PIXELS', 128 * 28 * 28)) # follow qwen2vl paper
# qwen_vl_utils.vision_process.FPS_MAX_FRAMES = int(os.environ.get('FPS_MAX_FRAMES', 768)) # decrease this for efficiency 
# from qwen_vl_utils.vision_process import (
#     FORCE_QWENVL_VIDEO_READER, VIDEO_TOTAL_PIXELS, FPS_MAX_FRAMES, VIDEO_MIN_PIXELS, VIDEO_MAX_PIXELS, FRAME_FACTOR, IMAGE_FACTOR, FPS,
#     smart_nframes, smart_resize
# )


def _spatial_resize_video(video: torch.Tensor, nframes: int = None): # 
    if not nframes:
        nframes, _, height, width = video.shape
    else:
        height, width = video.shape[2:]
    max_pixels = max(min(VIDEO_MAX_PIXELS, VIDEO_TOTAL_PIXELS / nframes * FRAME_FACTOR), int(VIDEO_MIN_PIXELS * 1.05))
    resized_height, resized_width = smart_resize( 
        height,
        width,
        factor=IMAGE_FACTOR,
        min_pixels=VIDEO_MIN_PIXELS,
        max_pixels=max_pixels,
    )
    video = transforms.functional.resize(
        video,
        [resized_height, resized_width],
        interpolation=transforms.InterpolationMode.BICUBIC,
        antialias=True,
    ).float() # need float?
    return video


class OvoBenchMCQDataset(Dataset):
    def __init__(self, remote_loader, path, question_prefix, question_postfix, answer_prefix, sample: int = None):
        lines = open(path).readlines()
        if sample is not None:
            random.seed(42)
            lines = random.sample(lines, sample)
        self.datums = [json.loads(line) for line in tqdm.tqdm(lines, desc='load datums')]
        if isinstance(self.datums[0], str):
            self.datums = [json.loads(datum) for datum in tqdm.tqdm(self.datums, desc='load datumsx2')]
        # self.datums = [datum for datum in self.datums if datum['task'] in ['REC', 'SSR', 'CRR']]
        self.src_video_dir = os.path.dirname("/2022233235/.cache/huggingface/hub/datasets--JoeLeelyf--OVO-Bench/snapshots/fec29e3385747b5642d995370143ba92d2819bd2/src_videos/")
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
        if datum['task'] in ['REC', 'SSR', 'CRR']: # 'REC', 'SSR', 'CRR' have already been chunked
            query = datum['question']
            video, _, clip_pts = _read_may1fps_video_decord({'video': os.path.join(self.src_video_dir, datum['video']), 'video_end': datum['video_end'],'remote_loader': self.remote_loader}, return_pts=True)
        # elif datum['task'] not in ['OCR', 'ACR', 'ATR', 'STU', 'FPD', 'OJR']: # backward tracing tasks
        #     query = self.question_prefix + datum['question'] + '\n' + '\n'.join(datum['options']) + self.question_postfix
        #     video, _ = _read_may1fps_video_decord({'video': os.path.join(self.src_video_dir, datum['video']), 'video_end': datum['video_end'], 'remote_loader': self.remote_loader})
        else:
            query = self.question_prefix + datum['question'] + '\n' + '\n'.join(datum['options']) + self.question_postfix
            video, _, clip_pts = _read_may1fps_video_decord({'video': os.path.join(self.src_video_dir, datum['video']),  'video_end': datum['video_end'], 'remote_loader': self.remote_loader}, return_pts=True)
        video = _spatial_resize_video(video)
        
        video_inputs = []
        streaming_fps_frames = int(FPS)
        # NOTE: add timestamp to video
        vison_content = []
        for i in range(0, len(video), streaming_fps_frames):
            start_timestamp, end_timestamp = i / FPS, (i + streaming_fps_frames) / FPS
            
            vison_content.append([{'type': 'text', 'text': f'Time={start_timestamp:.1f}-{end_timestamp:.1f}s'},
                                {'type': 'video', 'video': video[i:i+streaming_fps_frames]}])
            video_inputs.append(video[i:i+streaming_fps_frames])
        
        # # shuffle vison_content and video_inputs
        # combined = list(zip(vison_content, video_inputs))
        # random.shuffle(combined)
        # vison_content, video_inputs = zip(*combined)
        # vison_content = list(vison_content)
        # video_inputs = list(video_inputs)
        
        for i in range(0, len(vison_content)):
            conversation.append({"role": "user", "content": []})
            conversation[-1]['content'].extend(vison_content[i])
            conversation.append({"role": "assistant", "content": [{"type": "text", "text": " ..."}]})
        
        conversation[-2]['content'].append({'type': 'text', 'text': query})
        # pop -1 assitant 
        conversation.pop(-1)
        
        return conversation, video_inputs

    def data_collator(self, batch, processor):
        conversations, video_inputs = zip(*batch)
        texts = processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
        texts = [text + self.answer_prefix for text in texts]
        inputs = processor(
            text=texts[0],
            images=None,
            videos=video_inputs[0],
            padding=True,
            return_tensors="pt",
        )
        
        # Create a SimpleMRopeDynamicCache instance for cache management
        cache = SimpleMRopeDynamicCache(mrope_section=[32, 32, 32])  # Assuming 32 for each section
        inputs['cache_class'] = cache
        
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
    dataset = OvoBenchMCQDataset(remote_loader, benchmark_path, question_prefix=question_prefix, question_postfix=question_postfix, answer_prefix=answer_prefix)
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
    model_path = "chenjoya/LiveCC-7B-Instruct"
    model_path = "Qwen/Qwen2-VL-7B-Instruct"
    model_path = "/2022233235/videollm-online/livecc/outputs/qwen2vl_sft_24k768x100_llava178kSampleStream_lora_lr_clear1e-5/checkpoint-497"
    if "Qwen2.5" in model_path:
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", attn_implementation='flash_attention_2')
    else:
        model = Qwen2VLForConditionalGenerationForwardTwice.from_pretrained(model_path, torch_dtype="auto", attn_implementation='flash_attention_2')
    
    # Override the default cache class with our SimpleMRopeDynamicCache
    model.config._attn_implementation_internal = 'flash_attention_2'
    
    processor = AutoProcessor.from_pretrained(model_path, padding_side='left')
    options = ['No', 'Yes', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E']
    benchmark_path = 'ovo-bench-formatted.jsonl' 
    letter_idxs_predictions, benchmark_datums, process_index = mcq_predict(
        model=model, processor=processor, benchmark_path=benchmark_path, 
        options=options, use_liger_kernel='LiveCC' in model_path,
        answer_prefix = 'The answer is:\n', 
        abcd_previous_str = '\n',
        remote_loader=None,
        per_device_eval_batch_size=1,
        dataloader_num_workers=2,
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
        save_json_path = f'evaluation/ovobench/results/qwen2vl_{os.path.basename(model_path)}_stream_768_forwardtwice.json'
        os.makedirs(os.path.dirname(save_json_path), exist_ok=True)
        json.dump(results, open(save_json_path, 'w'))
        save_txt_path = save_json_path.replace('.json', '.txt')
        save_function_print(
            evaluate_ovobench_results,
            save_txt_path,
            results
        )

# torchrun --standalone --nproc_per_node=8 evaluation/ovobench/distributed_evaluate_ovobench_stamp.py