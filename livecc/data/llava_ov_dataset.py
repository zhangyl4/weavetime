import ast
import copy
import hashlib
import json
import logging
import math
import os
import re
import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from transformers import AutoTokenizer
from decord import VideoReader
from decord import cpu

from livecc_utils import _read_video_decord_plus, _spatial_resize_video

# Import LLaVA conversation utilities

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100

@dataclass
class DataArguments:
    annotation_paths: list[str] = field(default_factory=list)
    lazy_preprocess: bool = False
    is_multimodal: bool = True
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = field(default=None)
    video_fps: Optional[int] = field(default=1)
    frames_upbound: Optional[int] = field(default=64)
    force_sample: bool = False
    processor: Optional[Any] = None
    # Time-prompt settings
    enable_time_prompt: bool = True
    frames_per_input: int = 1
    # Sampling settings
    enable_streaming_sampling: bool = False
    streaming_beta: float = 3.0
    sampling_seed: Optional[int] = 42
    # Segment reconstruction settings (only when enable_time_prompt=True)
    enable_segment_reconstruction: bool = False
    segment_reconstruction_paths: list[str] = field(default_factory=list)
    segment_reconstruction_seed: Optional[int] = None
    segment_reconstruction_prob: float = 1  # Probability (0.0-1.0) to apply reconstruction for samples in paths
    # Group segments into this many clips; shuffle clips, keep intra-clip order
    segment_reconstruction_clip_number: Optional[int] = None
    segment_reconstruction_answer_joiner: str = " | "
    segment_reconstruction_answer_template: str = "Correct timestamps (in displayed order): {timestamps_joined}"
    # Degree control: True -> randomly pick degree in [0,1); False -> force 1.0 (shuffle all)
    segment_reconstruction_degree: float = 1.0

# --------- some utils copy from https://github1s.com/EvolvingLMMs-Lab/EgoLife/blob/main/EgoGPT/egogpt/train/train_audio.py#L551-L627---------


def process_video_with_decord(video_file, data_args):
    vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    avg_fps = round(vr.get_avg_fps() / data_args.video_fps)
    frame_idx = [i for i in range(0, total_frame_num, avg_fps)]
    frame_time = [i / avg_fps for i in frame_idx]

    if data_args.frames_upbound > 0:
        if len(frame_idx) > data_args.frames_upbound or data_args.force_sample:
            uniform_sampled_frames = np.linspace(
                0, total_frame_num - 1, data_args.frames_upbound, dtype=int
            )
            frame_idx = uniform_sampled_frames.tolist()
            frame_time = [i / vr.get_avg_fps() for i in frame_idx]
    frames = vr.get_batch(frame_idx).asnumpy()
    # resized_frames = np.array([cv2.resize(frame, (384, 384)) for frame in frames])
    # video = resized_frames
    video = frames
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])

    num_frames_to_sample = num_frames = len(frame_idx)
    # https://github.com/dmlc/decord/issues/208
    vr.seek(0)
    return video, video_time, frame_time, num_frames_to_sample


def process_video_with_decord_byframe(
    video_file, start_frame, end_frame, data_args, current_observation_frame=None
):
    try:
        vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        selected_frame = min(total_frame_num - 1, end_frame)
        avg_fps = round(vr.get_avg_fps() / data_args.video_fps)
        frame_idx = [i for i in range(start_frame, selected_frame, avg_fps)]

        if data_args.frames_upbound > 0:
            if len(frame_idx) > data_args.frames_upbound:
                uniform_sampled_frames = np.linspace(
                    start_frame, selected_frame, data_args.frames_upbound, dtype=int
                )
                frame_idx = uniform_sampled_frames.tolist()
        if current_observation_frame:
            frame_idx.append(current_observation_frame)
        video = vr.get_batch(frame_idx).asnumpy()
        # https://github.com/dmlc/decord/issues/208
        vr.seek(0)
    except:
        raise SyntaxError("Video processing error")
    return video

def split_text(text, keywords):
    pattern = "(" + "|".join(map(re.escape, keywords)) + ")"
    parts = re.split(pattern, text)
    parts = [part for part in parts if part]
    return parts


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    """Add video special tokens to the beginning of the conversation."""
    for source in sources:
        for sentence in source:
            if "<video>" in sentence["value"]:
                sentence["value"] = sentence["value"].replace("<video>", "").strip()
                sentence["value"] = "<video>\n" + sentence["value"]
                sentence["value"] = sentence["value"].strip()
    return sources

def preprocess_llava_conversation(
    sources,
    has_speech: bool = False,
    has_video: bool = False,
    has_image: bool = False,
    
) -> Dict:
    """
    Process conversations using LLaVA conversation template style
    """
    """
    conversation = [
        {
            'role': 'user', 'content': [
                {'type': 'video', 'video': video},
                {'type': 'text', 'text': text},
            ]
        },
        {'role': 'assistant', 'content': [{'type': 'text', 'text':  phrase }]}
        ...
    ]
    
    stream_conversation = [
            {
                'role': 'user', 'content': [
                    {'type': 'text', 'text': f'Time={start_timestamp:.1f}-{end_timestamp:.1f}s'}, 
                    {'type': 'video', 'video': frames},
                    {'type': 'text', 'text': f'Time={start_timestamp:.1f}-{end_timestamp:.1f}s'}, 
                    {'type': 'video', 'video': frames},
                    {'type': 'text', 'text': f'Time={start_timestamp:.1f}-{end_timestamp:.1f}s'}, 
                    {'type': 'video', 'video': frames},
                    ...
                    {'type': 'text', 'text': f'Time={start_timestamp:.1f}-{end_timestamp:.1f}s'}, 
                    {'type': 'video', 'video': frames},
                    {'type': 'text', 'text': question},
                ],  
            },
            {'role': 'assistant', 'content': [{'type': 'text', 'text':  ' ...'}]} # ' ...' denotes the streaming is not ended
        ]
    
    """
    
    conversations = []
    roles = {'human':'user', 'gpt':'assistant'}
    
    for i, source in enumerate(sources):
        # Create conversation template
        conversation = [{
            "role": "system",
            "content": [
                {"type": "text", "text": "You are a helpful assistant."},
            ],
        }]
        # Process each turn in the conversation
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            content = sentence["value"]
            conversation.append({'role': role, 'content': []})
            
            split_sentence = split_text(content, ["<video>\n", "<image>\n"])
            sorted_split_sentence = sorted(split_sentence, key=lambda x: "<image>\n" if x == "<image>\n" else "<video>\n" if x == "<video>\n" else "text", reverse=False)
            for part in sorted_split_sentence:
                if part in ["<video>\n", "<image>\n"]:
                    if has_video:
                        conversation[-1]['content'].append({'type': 'video'})
                    if has_image:
                        conversation[-1]['content'].append({'type': 'image'})
                else:
                    conversation[-1]['content'].append({'type': 'text', 'text': part})
        conversations.append(conversation)
    
    return conversations
# --------- some utils ---------


# --------- dataset ---------

class LLaVAOVDataset(Dataset):
    """Dataset for LLaVA OneVersion supervised fine-tuning."""

    def __init__(
        self,
        annotation_paths: list,
        tokenizer: PreTrainedTokenizer,
        data_args: DataArguments,
    ):
        super(LLaVAOVDataset, self).__init__()
        
        logging.warning("Loading data...")
        self.samples: List[Dict[str, Any]] = []
        for annotation_path in annotation_paths:
            if annotation_path.endswith('.jsonl'):
                seek_path = annotation_path.replace('.jsonl', '.seek.json')
                if os.path.exists(seek_path):
                    seeks = json.load(open(seek_path))
                    for s in seeks:
                        self.samples.append({'format': 'qwen', 'path': annotation_path, 'seek': int(s), 'source': annotation_path})
                        # if len(self.samples) >= 100:
                        #     print(f"Loaded {len(self.samples)} samples")
                        #     break
                else:
                    with open(annotation_path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                data = json.loads(line)
                            except Exception:
                                continue
                            self.samples.append({'format': 'qwen', 'data': data, 'source': annotation_path})
                            # if len(self.samples) >= 100:
                            #     print(f"Loaded {len(self.samples)} samples")
                            #     break
            else:
                data = json.load(open(annotation_path, 'r'))
                if isinstance(data, list):
                    for item in data:
                        self.samples.append({'format': 'llava', 'data': item, 'source': annotation_path})
                elif isinstance(data, dict):
                    self.samples.append({'format': 'llava', 'data': data, 'source': annotation_path})
                else:
                    raise ValueError(f"Unsupported JSON structure in {annotation_path}")
        
        logging.warning("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.data_args = data_args
        # Basic stride info for sampler compatibility
        self.video_info = [1 for _ in range(len(self.samples))]
        
        # HACK: add special tokens for LLaVA
        self.processor = data_args.processor
        self.im_start_id, self.assistant_id, self.newline_id, self.im_end_id = self.processor.tokenizer(
                '<|im_start|>assistant\n<|im_end|>').input_ids
        
        # Ensure the chat_template preserves content order for LLaVA-OneVision
        if self.data_args.enable_time_prompt:
            self._setup_chat_template()
    def __len__(self):
        return len(self.samples)

    def process_video(self, video_file, start_frame=None, end_frame=None, necessity_indices=None):
        """Process video with optional necessity + streaming sampling.
        Returns: frames (N,H,W,C), selected_indices (list[int]), timestamps (list[float]), fps (float)
        """
        if not os.path.exists(video_file):
            print(f"File {video_file} does not exist!")
            raise FileNotFoundError(f"Video file {video_file} not found")
        vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        actual_fps = float(vr.get_avg_fps()) if vr.get_avg_fps() is not None else float(self.data_args.video_fps or 1)
        # Define range
        seg_start = int(start_frame) if start_frame is not None else 0
        seg_end = min(int(end_frame) if end_frame is not None else total_frame_num, total_frame_num)
        if seg_end <= seg_start:
            seg_end = min(seg_start + 1, total_frame_num)
        # Base stride according to requested video_fps
        step = 1
        if self.data_args.video_fps and self.data_args.video_fps > 0 and actual_fps > 0:
            step = max(1, int(round(actual_fps / float(self.data_args.video_fps))))
        base_indices = list(range(seg_start, seg_end, step))
        # frames_upbound cap
        target_cap = int(self.data_args.frames_upbound or 0)
        if target_cap > 0 and (len(base_indices) > target_cap or self.data_args.force_sample):
            base_sampled_indices = np.linspace(0, len(base_indices) - 1, target_cap, dtype=int).tolist()
            base_indices = [base_indices[i] for i in base_sampled_indices]
        # Combine
        if necessity_indices is not None:
            combined_indices = sorted(set(base_indices) | set(necessity_indices))
        else:
            combined_indices = base_indices
        # Streaming sampling weights (favor later frames)
        if self.data_args.enable_streaming_sampling and target_cap > 0 and len(combined_indices) > target_cap:
            # Always keep necessity indices; sample the rest with weights
            keep_set = set(necessity_indices)
            remaining = [idx for idx in combined_indices if idx not in keep_set]
            remaining_needed = max(0, target_cap - len(keep_set))
            if remaining_needed < len(remaining):
                rng = np.random.default_rng(self.data_args.sampling_seed)
                # weights via exp(beta * normalized_position)
                denom = (seg_end - seg_start - 1) if (seg_end - seg_start - 1) > 0 else 1
                weights = np.array([np.exp(self.data_args.streaming_beta * ((idx - seg_start) / denom)) for idx in remaining], dtype=np.float64)
                weights = weights / weights.sum()
                chosen = rng.choice(remaining, size=remaining_needed, replace=False, p=weights)
                selected_indices = sorted(set(necessity_indices) | set(map(int, chosen.tolist())))
            else:
                selected_indices = sorted(set(necessity_indices) | set(remaining))
        else:
            selected_indices = combined_indices
        # Guard cap again
        if target_cap > 0 and len(selected_indices) > target_cap:
            # Fallback: uniform downsample but ensure necessity retained
            keep_set = set(necessity_indices)
            remaining = [idx for idx in selected_indices if idx not in keep_set]
            remaining_needed = max(0, target_cap - len(keep_set))
            if remaining_needed < len(remaining):
                positions = np.linspace(0, len(remaining) - 1, remaining_needed, dtype=int).tolist()
                down = [remaining[p] for p in positions]
                selected_indices = sorted(set(necessity_indices) | set(down))
            else:
                selected_indices = sorted(set(necessity_indices) | set(remaining))
        # Fetch frames
        if len(selected_indices) == 0:
            selected_indices = [seg_start]
        frames = vr.get_batch(selected_indices).asnumpy()
        # timestamps per selected frame
        timestamps = [idx / actual_fps for idx in selected_indices]
        # https://github.com/dmlc/decord/issues/208
        vr.seek(0)
        return frames, selected_indices, timestamps, actual_fps

    def process_image(self, image_file: str, resize_method: str = None):
        """Read image(s) and preprocess via image_processor."""
        if not os.path.exists(image_file):
            print(f"File {image_file} does not exist!")
            raise FileNotFoundError(f"Image file {image_file} not found")
        image = Image.open(image_file).convert("RGB")
        processor = self.data_args.image_processor
        if processor is None:
            raise ValueError("image_processor is required for processing images")
        out = processor.preprocess(image, return_tensors="pt")["pixel_values"]
        return out, image.size

    def getitem(self, i) -> Dict[str, torch.Tensor]:
        sample = self.samples[i]
        image_inputs = None
        video_inputs = None
        
        if sample['format'] == 'llava':
            entry = sample['data']
            sources = [entry]
            conversation = None
            if 'video' in entry:
                video_file = entry['video']
                if not os.path.exists(video_file):
                    raise FileNotFoundError(f"Video file {video_file} not found")
                if 'start_frame' in entry:
                    start_frame = entry['start_frame']
                    end_frame = entry['end_frame']
                    current_observation_frame = entry.get('current_observation_frame', None)
                    video, frame_indices, frame_timestamps, fps = self.process_video(video_file, start_frame, end_frame, current_observation_frame)
                else:
                    video, frame_indices, frame_timestamps, fps = self.process_video(video_file)
                if video.shape[0] == 0:
                    raise Exception(f"Video is empty for sample {i}")
                if self.data_args.enable_time_prompt:
                    frames_per_input = max(1, int(self.data_args.frames_per_input))
                    all_videos, time_texts = self._build_time_prompt_segments_from_indices(
                        video_frames=video,
                        frame_indices=frame_indices,
                        frames_per_input=frames_per_input,
                        frame_timestamps=frame_timestamps,
                        sample_fps=fps,
                    )
                    conversation = preprocess_llava_conversation([e["conversations"] for e in sources], has_video=True, has_image=False)
                    # Apply segment reconstruction (partial shuffle controlled by degree) if enabled for this sample
                    if self._should_apply_segment_reconstruction(sample) and len(all_videos) > 1:
                        seed = (
                            self.data_args.segment_reconstruction_seed
                            if self.data_args.segment_reconstruction_seed is not None
                            else self.data_args.sampling_seed
                        )
                        rng = np.random.default_rng(seed)
                        degree = getattr(self.data_args, 'segment_reconstruction_degree', 1.0)
                        clip_number = getattr(self.data_args, 'segment_reconstruction_clip_number', None)
                        perm = self._compute_partial_permutation(len(all_videos), degree, rng, clip_number)
                        # Inject time prompts
                        conversation = self._inject_time_prompts_into_conversation(conversation, all_videos, time_texts)
                        # Compute target order and always insert QA (even if identity permutation)
                        is_changed = any(idx != i for i, idx in enumerate(perm))
                        if is_changed:
                            video_inputs = [all_videos[p] for p in perm]
                        else:
                            video_inputs = all_videos
                        correct_texts = [time_texts[p] for p in perm]
                        question = self._choose_reconstruction_prompt(rng)
                        joiner = getattr(self.data_args, 'segment_reconstruction_answer_joiner', ' | ') or ' | '
                        answer_template = getattr(self.data_args, 'segment_reconstruction_answer_template', "Correct timestamps (in displayed order): {timestamps_joined}")
                        answer = answer_template.format(timestamps_joined=joiner.join(correct_texts))
                        conversation = self._prepend_reconstruction_qa_llava(conversation, question, answer)
                    else:
                        conversation = self._inject_time_prompts_into_conversation(conversation, all_videos, time_texts)
                        video_inputs = all_videos
                else:
                    conversation = preprocess_llava_conversation([e["conversations"] for e in sources], has_video=True, has_image=False)
                    video_inputs = [video]
            elif 'image' in entry:
                image_file = entry['image']
                if isinstance(image_file, list):
                    image = [self.process_image(f) for f in image_file]
                    if len(image_file) > 1:
                        image = [self.process_image(f, "pad") for f in image_file]
                        image = [[im[0], im[1], "image"] for im in image]
                else:
                    image = [self.process_image(image_file)]
                image_inputs = image
                conversation = preprocess_llava_conversation([e["conversations"] for e in sources], has_video=False, has_image=True)
            else:
                conversation = preprocess_llava_conversation([e["conversations"] for e in sources], has_video=False, has_image=False)
        else:
            # Qwen2VL format: conversation is already in messages/content schema
            conversation_raw = self._read_qwen_conversation(sample)
            conversation, image_inputs, video_inputs = self._process_qwen_conversation(conversation_raw, sample)
        texts = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False, return_tensors='pt')
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        )
        input_ids = inputs.input_ids
        
        labels = torch.full_like(input_ids, fill_value=-100, dtype=input_ids.dtype)
        im_start_idxs = (input_ids == self.im_start_id).nonzero()
        im_end_idxs = (input_ids == self.im_end_id).nonzero()
        for (sample_idx, im_start_idx), (sample_idx, im_end_idx) in zip(im_start_idxs, im_end_idxs):
            if input_ids[sample_idx, im_start_idx + 1] == self.assistant_id:
                labels[sample_idx, im_start_idx+3:im_end_idx+1] = input_ids[sample_idx, im_start_idx+3:im_end_idx+1]
        inputs['labels'] = labels
        inputs['original_idx'] = torch.tensor(i, dtype=torch.long)
        return inputs

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # return self.getitem(i)
        max_try = 10
        for _ in range(max_try):
            try:
                return self.getitem(i)
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                i = (i + 1) % len(self.samples)
        raise Exception(f"Failed to process sample {i} after {max_try} tries")
        
    
    def data_collator(self, batched_inputs, **kwargs):
        assert len(batched_inputs) == 1
        return batched_inputs[0]

    def _setup_chat_template(self):
        """
        Modify processor's chat_template to preserve content order.
        Only applies to LLaVA-OneVision processor variants.
        """
        try:
            processor_class = self.processor.__class__.__name__
        except Exception:
            processor_class = ""
        if 'LlavaOnevision' not in processor_class:
            logger.info(f'Skipping chat_template update for {processor_class} (only for LlavaOnevisionProcessor)')
            return
        # Template that preserves content order exactly as provided
        new_template = """{% for message in messages %}{{ '<|im_start|>' + message['role'] + ' ' }}{% for content in message['content'] %}{% if content['type'] == 'image' %}{{ '<image>' }}{% elif content['type'] == 'video' %}{{ '<video>' }}{% elif content['type'] == 'text' %}{% if message['role'] != 'assistant' %}{{ '\\n' + content['text'] }}{% else %}{% generation %}{{ '\\n' + content['text'] }}{% endgeneration %}{% endif %}{% endif %}{% endfor %}{{ '<|im_end|>' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"""
        if hasattr(self.processor, 'tokenizer'):
            try:
                _ = self.processor.tokenizer.chat_template
            except Exception:
                pass
            self.processor.tokenizer.chat_template = new_template
        # Some processors also read from processor.chat_template
        setattr(self.processor, 'chat_template', new_template)

    def _compute_partial_permutation(
        self,
        length: int,
        degree: float,
        rng: Optional[np.random.Generator] = None,
        clip_number: Optional[int] = None,
    ) -> List[int]:
        """Return a permutation list for partial shuffling based on degree in [0,1].
        - degree=0.0 -> identity
        - degree=1.0 -> full random permutation
        - 0<degree<1 -> choose ~degree*length positions and permute among themselves
        - clip_number>1 -> keep intra-clip order, shuffle across clips only
        """
        try:
            degree = float(degree)
        except Exception:
            degree = 1.0
        degree = max(0.0, min(1.0, degree))
        if rng is None:
            rng = np.random.default_rng(getattr(self.data_args, 'sampling_seed', None))

        def _partial_perm(n: int) -> List[int]:
            if n <= 1 or degree <= 0.0:
                return list(range(n))
            if degree >= 1.0:
                return rng.permutation(n).tolist()
            k = int(round(degree * n))
            if k < 2:
                return list(range(n))
            k = min(k, n)
            selected_positions = rng.choice(n, size=k, replace=False).tolist()
            selected_positions.sort()
            permuted_positions = rng.permutation(selected_positions).tolist()
            if permuted_positions == selected_positions and k > 1:
                # ensure a change when possible
                permuted_positions = selected_positions[1:] + selected_positions[:1]
            perm_local = list(range(n))
            for new_pos, old_pos in zip(selected_positions, permuted_positions):
                perm_local[new_pos] = old_pos
            return perm_local

        if length <= 1:
            return list(range(length))

        # Optional clip grouping: keep order inside each clip, shuffle at clip level.
        try:
            clip_number_val = int(clip_number) if clip_number is not None else 1
        except Exception:
            clip_number_val = 1
        if clip_number_val is not None and clip_number_val > 1:
            clip_number_val = max(1, min(clip_number_val, length))
            group_size = int(math.ceil(length / clip_number_val))
            groups: List[List[int]] = []
            for start in range(0, length, group_size):
                groups.append(list(range(start, min(start + group_size, length))))
            group_perm = _partial_perm(len(groups))
            perm_grouped: List[int] = []
            for g_idx in group_perm:
                perm_grouped.extend(groups[g_idx])
            return perm_grouped

        return _partial_perm(length)

    def _build_time_prompt_segments_from_indices(
        self, 
        video_frames: np.ndarray, 
        frame_indices: List[int], 
        frames_per_input: int, 
        frame_timestamps: List[float],
        sample_fps: float
    ):
        """
        Split video frames into segments of size frames_per_input and generate accurate time prompts.
        Uses actual frame indices and fps to compute timestamps.
        
        Args:
            video_frames: (N, H, W, C) array of frames returned by process_video
            frame_indices: list of N integers, actual frame indices in the original video
            frames_per_input: number of frames per segment
            frame_timestamps: list of N floats, actual frame timestamps in the original video
            sample_fps: actual fps of the sampled video
        Returns:
            (all_videos, time_texts): 
                all_videos: list of np.ndarray segments
                time_texts: list of str time prompts like "Time=0000000.0-0000003.0s"
        """
        total_frames = int(video_frames.shape[0])
        all_videos: List[np.ndarray] = []
        time_texts: List[str] = []
        if total_frames <= 0 or len(frame_indices) != total_frames:
            return all_videos, time_texts
        
        # Add random offset to start time between 0-300s in discrete steps of 1s
        random_offset = np.random.randint(0, 301)
        num_segments = (total_frames + frames_per_input - 1) // frames_per_input
        for seg_idx in range(num_segments):
            seg_start = seg_idx * frames_per_input
            seg_end = min((seg_idx + 1) * frames_per_input, total_frames)
            segment_frames = video_frames[seg_start:seg_end]
            # Use actual frame indices for timestamps
            timestamp_seg = frame_timestamps[seg_start:seg_end+1] if seg_end+1 <= total_frames else frame_timestamps[seg_start:]

            time_text = f"Time={timestamp_seg[0]:07.1f}-{timestamp_seg[-1]:07.1f}s"
            time_texts.append(time_text)
            all_videos.append(segment_frames)
        return all_videos, time_texts

    def _inject_time_prompts_into_conversation(self, conversations: List[List[Dict[str, Any]]], all_videos: List[np.ndarray], time_texts: List[str]):
        """Replace a single 'video' slot in the first user turn with [time_text, video] pairs."""
        if not conversations:
            return conversations
        conv = conversations[0]
        expanded = False
        for msg in conv:
            if msg.get('role') == 'user' and isinstance(msg.get('content'), list):
                contents = msg['content']
                has_video_marker = any((isinstance(c, dict) and c.get('type') == 'video') for c in contents)
                if not has_video_marker:
                    continue
                new_contents: List[Dict[str, Any]] = []
                for c in contents:
                    if isinstance(c, dict) and c.get('type') == 'video':
                        for t in time_texts:
                            new_contents.append({'type': 'text', 'text': t})
                            new_contents.append({'type': 'video'})
                    else:
                        new_contents.append(c)
                msg['content'] = new_contents
                expanded = True
                break
        if not expanded:
            logger.warning("Time-prompt injection found no video slot to expand; leaving conversation unchanged.")
        return conversations
    
    def _read_qwen_conversation(self, sample: Dict[str, Any]):
        """Read a single Qwen2VL-format conversation sample from memory or by seek."""
        if 'data' in sample:
            return sample['data']
        path = sample['path']
        seek = int(sample['seek'])
        with open(path, 'r') as f:
            f.seek(seek)
            line = f.readline()
        try:
            data = json.loads(line)
        except Exception as e:
            raise ValueError(f"Failed to parse JSONL at seek {seek} in {path}: {e}")
        return data

    def _process_qwen_conversation(self, conversation_raw: List[Dict[str, Any]], sample: Dict[str, Any]):
        """Normalize Qwen2VL messages to LLaVA-OV style content and collect media inputs.
        - Preserves order of content items
        - Expands videos into time-prompted segments when enabled
        Returns: (conversation, image_inputs, video_inputs)
        """
        image_inputs: List[Any] = []
        video_inputs: List[np.ndarray] = []
        conversation: List[Dict[str, Any]] = []
        applied_reconstruction: bool = False
        accumulated_correct_texts: List[str] = []
        # RNG for deterministic behavior if seed provided
        seed = (
            self.data_args.segment_reconstruction_seed
            if getattr(self.data_args, 'segment_reconstruction_seed', None) is not None
            else getattr(self.data_args, 'sampling_seed', None)
        )
        rng = np.random.default_rng(seed)
        if not conversation_raw or not isinstance(conversation_raw, list):
            raise ValueError("Qwen2VL conversation must be a list of messages")
        # Ensure system message exists first
        if conversation_raw[0].get('role') != 'system':
            conversation.append({
                'role': 'system',
                'content': [{'type': 'text', 'text': 'You are a helpful assistant.'}],
            })
        for msg in conversation_raw:
            role = msg.get('role')
            content = msg.get('content', [])
            if role not in ('user', 'assistant', 'system'):
                continue
            new_msg = {'role': role, 'content': []}
            # For system messages, allow passthrough of any text
            if role == 'system':
                for elem in content:
                    if isinstance(elem, dict) and elem.get('type') == 'text' and 'text' in elem:
                        new_msg['content'].append({'type': 'text', 'text': elem['text']})
                conversation.append(new_msg)
                continue
            # Process user/assistant content
            for elem in content:
                if not isinstance(elem, dict):
                    continue
                elem_type = elem.get('type')
                if elem_type == 'text':
                    text_val = elem.get('text') or elem.get('text_stream') or ''
                    if isinstance(text_val, str) and text_val:
                        new_msg['content'].append({'type': 'text', 'text': text_val})
                elif elem_type == 'image':
                    image_ref = elem.get('image')
                    if isinstance(image_ref, str) and os.path.exists(image_ref):
                        try:
                            img = Image.open(image_ref).convert('RGB')
                            image_inputs.append(img)
                            new_msg['content'].append({'type': 'image'})
                        except Exception:
                            logger.warning(f"Failed to open image: {image_ref}")
                    else:
                        # Unsupported image modality reference; keep placeholder if any
                        new_msg['content'].append({'type': 'image'})
                elif elem_type == 'video':
                    video_ref = elem.get('video')
                    # Handle path-based videos; other forms are not supported here
                    segment_videos: List[np.ndarray] = []
                    if isinstance(video_ref, str) and os.path.exists(video_ref):
                        # Parse optional video_start, video_end, nframes from elem
                        video_start_time = elem.get('video_start', None)
                        video_end_time = elem.get('video_end', None)
                        nframes_cap = elem.get('nframes', None)
                        
                        # get necessity indices
                        if 'key_frame_idx' in elem:
                            necessity_indices = elem['key_frame_idx']
                        else:
                            necessity_indices = None
                        
                        # Convert time-based ranges to frame indices
                        start_frame_arg = None
                        end_frame_arg = None
                        if video_start_time is not None or video_end_time is not None:
                            # Read video metadata to convert timestamps to frames
                            from decord import VideoReader, cpu
                            vr_temp = VideoReader(video_ref, ctx=cpu(0), num_threads=1)
                            actual_fps = float(vr_temp.get_avg_fps())
                            total_frames = len(vr_temp)
                            vr_temp.seek(0)
                            
                            if video_start_time is not None:
                                start_frame_arg = int(float(video_start_time) * actual_fps)
                            else:
                                start_frame_arg = 0
                            if video_end_time is not None:
                                end_frame_arg = int(float(video_end_time) * actual_fps)
                            else:
                                end_frame_arg = total_frames
                        
                        # Temporarily override frames_upbound if nframes is specified
                        original_frames_upbound = self.data_args.frames_upbound
                        if nframes_cap is not None:
                            self.data_args.frames_upbound = int(nframes_cap)
                            self.data_args.force_sample = True

                        try:
                            video, frame_indices, frame_timestamps, fps = self.process_video(
                                video_ref, 
                                start_frame=start_frame_arg, 
                                end_frame=end_frame_arg,
                                necessity_indices=necessity_indices
                            )
                        finally:
                            # Restore original settings
                            if nframes_cap is not None:
                                self.data_args.frames_upbound = original_frames_upbound
                                self.data_args.force_sample = False
                        
                        if video.shape[0] == 0:
                            continue
                        if self.data_args.enable_time_prompt:
                            frames_per_input = max(1, int(self.data_args.frames_per_input))
                            segment_videos, time_texts = self._build_time_prompt_segments_from_indices(
                                video_frames=video,
                                frame_indices=frame_indices,
                                frames_per_input=frames_per_input,
                                frame_timestamps=frame_timestamps,
                                sample_fps=fps,
                            )
                            # Append time texts (original order) and corresponding video markers
                            for t in time_texts:
                                new_msg['content'].append({'type': 'text', 'text': t})
                                new_msg['content'].append({'type': 'video'})
                            # Optionally shuffle actual video tensors to mismatch timestamps
                            if self._should_apply_segment_reconstruction(sample) and len(segment_videos) > 1:
                                degree = getattr(self.data_args, 'segment_reconstruction_degree', 1.0)
                                clip_number = getattr(self.data_args, 'segment_reconstruction_clip_number', None)
                                perm = self._compute_partial_permutation(len(segment_videos), degree, rng, clip_number)
                                is_changed = any(idx != i for i, idx in enumerate(perm))
                                if is_changed:
                                    shuffled_videos = [segment_videos[p] for p in perm]
                                    video_inputs.extend(shuffled_videos)
                                else:
                                    video_inputs.extend(segment_videos)
                                correct_texts = [time_texts[p] for p in perm]
                                accumulated_correct_texts.extend(correct_texts)
                                applied_reconstruction = True
                            else:
                                video_inputs.extend(segment_videos)
                        else:
                            new_msg['content'].append({'type': 'video'})
                            video_inputs.append(video)
                    else:
                        # Unsupported video reference; keep placeholder to maintain structure
                        new_msg['content'].append({'type': 'video'})
                else:
                    # Unknown content types are ignored
                    continue
            conversation.append(new_msg)
        # If reconstruction applied anywhere, insert QA after the last user message with video segments
        if applied_reconstruction and len(accumulated_correct_texts) > 0:
            question = self._choose_reconstruction_prompt(rng)
            joiner = getattr(self.data_args, 'segment_reconstruction_answer_joiner', ' | ') or ' | '
            answer_template = getattr(self.data_args, 'segment_reconstruction_answer_template', "Correct timestamps (in displayed order): {timestamps_joined}")
            answer = answer_template.format(timestamps_joined=joiner.join(accumulated_correct_texts))
            # Insert QA right after the last user message that contains a video token
            # Find first user message with video and its corresponding assistant response
            conversation = self._prepend_reconstruction_qa_llava([conversation], question, answer)
            conversation = conversation[0]
        return conversation, (image_inputs or None), (video_inputs or None)

    def _should_apply_segment_reconstruction(self, sample: Dict[str, Any]) -> bool:
        if not getattr(self.data_args, 'enable_time_prompt', False):
            return False
        if not getattr(self.data_args, 'enable_segment_reconstruction', False):
            return False
        try:
            src = sample.get('source')
        except Exception:
            src = None
        paths = getattr(self.data_args, 'segment_reconstruction_paths', []) or []
        # Check if sample is in the specified paths
        if src not in paths:
            return False
        
        # Apply random threshold if probability < 1.0
        prob = getattr(self.data_args, 'segment_reconstruction_prob', 1.0)
        if prob >= 1.0:
            return True
        if prob <= 0.0:
            return False
        
        # Generate deterministic random number based on sample identity
        try:
            # Create a unique identifier for this sample
            sample_id = str(src)
            if 'seek' in sample:
                sample_id += f"_{sample['seek']}"
            elif 'data' in sample:
                # Use a hash of the data for deterministic behavior
                data_str = json.dumps(sample['data'], sort_keys=True)
                data_hash = hashlib.md5(data_str.encode()).hexdigest()[:8]
                sample_id += f"_{data_hash}"
            
            # Use seed for deterministic randomness
            seed = (
                self.data_args.segment_reconstruction_seed
                if getattr(self.data_args, 'segment_reconstruction_seed', None) is not None
                else getattr(self.data_args, 'sampling_seed', 42)
            )
            # Create RNG based on seed and sample_id
            combined_seed = hash(sample_id) % (2**31) + seed
            rng = np.random.default_rng(combined_seed)
            # Generate random value and compare with probability
            return rng.random() < prob
        except Exception as e:
            logger.warning(f"Error in random threshold check for segment reconstruction: {e}, defaulting to True")
            return True

    def _choose_reconstruction_prompt(self, rng: Optional[np.random.Generator] = None) -> str:
        prompts = [
            "These video segments are shuffled. List each segment's true time range.",
            "Segments have been randomly reordered. Output the correct timestamps (start–end) for each.",
            "The segment order is randomized. Provide the real time span (start–end) for each segment.",
            "Order and timestamps do not match. Write the true timestamps in chronological order.",
            "Displayed timestamps may be wrong. Output the correct time range list.",
            "Segments are shuffled. Infer and write each segment's correct timestamp.",
            "Treat shown timestamps as distractors. Provide the real start–end times for each segment.",
            "Recover the true temporal order and provide each segment's time range.",
            "Current segment order is random. Output timestamps in actual occurrence order.",
            "Time prompts do not match the videos. Provide each segment's correct time span."
        ]
        if rng is None:
            idx = np.random.randint(0, len(prompts))
        else:
            idx = int(rng.integers(0, len(prompts)))
        return prompts[idx]

    def _prepend_reconstruction_qa_llava(self, conversations, question: str, answer: str):
        """Insert QA immediately after the first user message that contains a video token.
        Accepts either a single conversation (list[dict]) or a list containing one conversation.
        """
        if not conversations or not isinstance(conversations, list):
            return conversations
        # Normalize to (conv_container, conv)
        if conversations and isinstance(conversations[0], dict) and 'role' in conversations[0]:
            conv_container = None
            conv = conversations
        else:
            conv_container = conversations
            conv = conversations[0] if conversations else []
        if not conv or not isinstance(conv, list):
            return conversations
        # Find last user msg with a video content
        first_video_idx = None
        for idx, msg in enumerate(conv):
            if msg.get('role') == 'user' and isinstance(msg.get('content'), list):
                if any((isinstance(c, dict) and c.get('type') == 'video') for c in msg['content']):
                    first_video_idx = idx
                    break
        
        if first_video_idx is None:
            first_video_idx = 1 if (len(conv) > 0 and conv[0].get('role') == 'system') else 0
        
        # Store original QA pair
        original_user = conv[first_video_idx]
        original_assistant = conv[first_video_idx + 1]
        
        originl_first_question = original_user['content'][-1]['text']
        original_user['content'][-1]['text'] = question
        
        # replace the original QA pair with the new QA pair
        # Replace with new QA pair
        conv[first_video_idx] = original_user
        conv[first_video_idx + 1] = {'role': 'assistant', 'content': [{'type': 'text', 'text': answer}]}
        
        # Insert original QA pair after
        conv.insert(first_video_idx + 2, {'role': 'user', 'content': [{'type': 'text', 'text': originl_first_question}]})
        conv.insert(first_video_idx + 3, original_assistant)
        return conv_container
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test LLaVAOVDataset construction and preprocessing")
    parser.add_argument(
        "--path",
        type=str,
        default="/2024233235/.cache/huggingface/hub/datasets--lmms-lab--EgoIT-99K/snapshots/a57f1f2078a7b01ea87014050fdb3afe169e54f1/datasets/EgoIT_process.json",
        help="Path to the JSON dataset file"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="llava-hf/llava-onevision-qwen2-7b-ov-hf",
        help="Tokenizer model name to use for testing"
    )
    args = parser.parse_args()

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if getattr(tokenizer, "pad_token_id", None) is None:
        # Fallback to using eos as pad if missing (e.g., some causal LMs)
        if getattr(tokenizer, "eos_token_id", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise ValueError("Tokenizer has no pad_token_id and no eos_token_id; please choose a different tokenizer")

    
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(args.model_name)
    
    data_args = DataArguments(annotation_paths=[args.path], 
                              processor=processor,
                              enable_time_prompt=True,
                            #   enable_segment_reconstruction=True,
                            #   segment_reconstruction_paths=[
                            #       "/2024233235/videollm-online/EyeWO2/data/llava_video_178k_with_seeks_sample_valid.jsonl",
                            #       "/2024233235/videollm-online/EyeWO2/data/cof_qwen2vl_timestamp.jsonl",
                            #   ],
                              )

    print(f"Constructing dataset from: {args.path}")
    dataset = LLaVAOVDataset(
        annotation_paths=[
            # "/2024233235/.cache/huggingface/hub/datasets--lmms-lab--EgoIT-99K/snapshots/a57f1f2078a7b01ea87014050fdb3afe169e54f1/datasets/EgoIT_min448frames.json", 
            '/2024233235/videollm-online/EyeWO2/data/llava_video_178k_with_seeks_sample_valid.jsonl',
            # '/2024233235/videollm-online/EyeWO2/data/qwen_format_split_videoQA.jsonl',
            # '/2024233235/videollm-online/EyeWO2/data/qwen_format_random_videoQA.jsonl'
            # "/2024233235/videollm-online/EyeWO2/data/etbench_qwen2vl_timestamp.jsonl",
            # "/2024233235/videollm-online/EyeWO2/data/cof_qwen2vl_timestamp.jsonl",
            
            ],
        tokenizer=tokenizer,
        data_args=data_args,
    )
    print(f"Dataset loaded with {len(dataset)} samples")

    if len(dataset) == 0:
        print("Dataset is empty; nothing to preprocess.")
        raise SystemExit(0)

    out = dataset[0]

    # print("Preprocessing complete.")
    # print(f"input_ids shape: {tuple(out['input_ids'].shape)}")
    # print(f"labels shape: {tuple(out['labels'].shape)}")

    # print("Building a test batch with the data collator...")
    # from torch.utils.data import DataLoader
    # # Only process samples 3110-3120
    # subset_indices = list(range(3110, 3121))
    # subset_dataset = torch.utils.data.Subset(dataset, subset_indices)
    # dataloader = DataLoader(subset_dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=dataset.data_collator)
    
    # for i, batch in enumerate(dataloader, start=3110):
    #     print(f"Processing sample {i}")
        