import os
import json
import hashlib
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from livecc_utils import _read_video_decord_plus, _spatial_resize_video

logger = logging.getLogger(__name__)


@dataclass
class DataArguments:
    annotation_paths: List[str] = field(default_factory=list)
    # Core
    processor: Optional[Any] = None
    image_processor: Optional[Any] = None
    # Time-prompt settings
    enable_time_prompt: bool = True
    frames_per_input: int = 2
    # Segment reconstruction settings
    enable_segment_reconstruction: bool = False
    segment_reconstruction_paths: List[str] = field(default_factory=list)
    segment_reconstruction_seed: Optional[int] = None
    segment_reconstruction_prob: float = 1
    # Group segments into this many clips; shuffle clips, keep intra-clip order
    segment_reconstruction_clip_number: Optional[int] = None
    segment_reconstruction_answer_joiner: str = " | "
    segment_reconstruction_answer_template: str = (
        "Correct timestamps (in displayed order): {timestamps_joined}"
    )
    # Degree control: True -> randomly pick degree in [0,1); False -> force 1.0 (shuffle all)
    segment_reconstruction_degree: bool = False


def split_text(text: str, keywords: Sequence[str]) -> List[str]:
    import re

    pattern = "(" + "|".join(map(re.escape, keywords)) + ")"
    parts = re.split(pattern, text)
    parts = [part for part in parts if part]
    return parts


def preprocess_llava_conversation(
    sources: Sequence[Sequence[Dict[str, Any]]],
    has_speech: bool = False,
    has_video: bool = False,
    has_image: bool = False,
) -> List[List[Dict[str, Any]]]:
    conversations: List[List[Dict[str, Any]]] = []
    roles = {"human": "user", "gpt": "assistant"}

    for source in sources:
        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a helpful assistant."},
                ],
            }
        ]
        for sentence in source:
            role = roles.get(sentence.get("from", "human"), "user")
            content = sentence.get("value", "")
            conversation.append({"role": role, "content": []})

            sorted_split_sentence = sorted(
                split_text(content, ["<video>\n", "<image>\n"]),
                key=lambda x: "<image>\n"
                if x == "<image>\n"
                else "<video>\n"
                if x == "<video>\n"
                else "text",
                reverse=False,
            )
            for part in sorted_split_sentence:
                if part in ["<video>\n", "<image>\n"]:
                    if has_video and part == "<video>\n":
                        conversation[-1]["content"].append({"type": "video"})
                    if has_image and part == "<image>\n":
                        conversation[-1]["content"].append({"type": "image"})
                else:
                    conversation[-1]["content"].append({"type": "text", "text": part})
        conversations.append(conversation)
    return conversations


class QwenVLDataset(Dataset):
    """Dataset matching LLaVA-OV features (time-prompt + segment reconstruction),
    but loading videos in LMMDataset style via _read_video_decord_plus.
    Supports both LLaVA-style JSON and Qwen2VL-style JSONL.
    """

    def __init__(
        self,
        annotation_paths: List[str],
        tokenizer: PreTrainedTokenizer,
        data_args: DataArguments,
    ) -> None:
        super().__init__()
        logger.warning("Loading data...")
        self.samples: List[Dict[str, Any]] = []
        for annotation_path in annotation_paths:
            if annotation_path.endswith(".jsonl"):
                seek_path = annotation_path.replace(".jsonl", ".seek.json")
                if os.path.exists(seek_path):
                    seeks = json.load(open(seek_path))
                    for s in seeks:
                        self.samples.append(
                            {
                                "format": "qwen",
                                "path": annotation_path,
                                "seek": int(s),
                                "source": annotation_path,
                            }
                        )
                else:
                    with open(annotation_path, "r") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                data = json.loads(line)
                            except Exception:
                                continue
                            self.samples.append(
                                {"format": "qwen", "data": data, "source": annotation_path}
                            )
            else:
                data = json.load(open(annotation_path, "r"))
                if isinstance(data, list):
                    for item in data:
                        self.samples.append(
                            {"format": "llava", "data": item, "source": annotation_path}
                        )
                elif isinstance(data, dict):
                    self.samples.append(
                        {"format": "llava", "data": data, "source": annotation_path}
                    )
                else:
                    raise ValueError(
                        f"Unsupported JSON structure in {annotation_path}"
                    )

        logger.warning("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.video_info = [1 for _ in range(len(self.samples))]

        # Processor and tokenizer ids (Qwen-style)
        self.processor = data_args.processor
        if hasattr(self.processor, "tokenizer"):
            ids = self.processor.tokenizer("<|im_start|>assistant\n<|im_end|>").input_ids
            # Expect 4 ids
            self.im_start_id, self.assistant_id, self.newline_id, self.im_end_id = ids
        else:
            raise ValueError("processor with tokenizer is required")


    def __len__(self) -> int:
        return len(self.samples)

    # -------------- Core media IO --------------
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
            rng = np.random.default_rng(getattr(self.data_args, "sampling_seed", None))

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
                permuted_positions = selected_positions[1:] + selected_positions[:1]
            perm_local = list(range(n))
            for new_pos, old_pos in zip(selected_positions, permuted_positions):
                perm_local[new_pos] = old_pos
            return perm_local

        # Optional clip grouping: keep order inside each clip, shuffle at clip level.
        try:
            clip_number_val = int(clip_number) if clip_number is not None else 1
        except Exception:
            clip_number_val = 1

        if length <= 1:
            return list(range(length))

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
    def process_video(
        self,
        video_file: str,
        start_time_s: Optional[float] = None,
        end_time_s: Optional[float] = None,
    ) -> Tuple[torch.Tensor, List[int], List[float]]:
        """Load a video via _read_video_decord_plus and optionally slice by time window.
        Returns: (clip[T,C,H,W], frame_indices[list[int]], timestamps[list[float]])
        Note: Do NOT cap frames here; Qwen manages frame counts elsewhere.
        """
        if not os.path.exists(video_file):
            raise FileNotFoundError(f"Video file {video_file} not found")

        # Build minimal element for the reader
        element = {"video": video_file}
        clip, _, clip_pts = _read_video_decord_plus(element, return_pts=True, strict_fps=True)
        clip = _spatial_resize_video(clip)
        # Ensure shape (T,C,H,W)
        if isinstance(clip, np.ndarray):
            clip = torch.from_numpy(clip)
        if clip.dim() == 4 and clip.shape[-1] in (1, 3):
            # (T, H, W, C) -> (T, C, H, W)
            clip = clip.permute(0, 3, 1, 2).contiguous()
        elif clip.dim() != 4:
            raise ValueError(f"Unexpected video tensor shape: {tuple(clip.shape)}")

        timestamps = [float(t) for t in clip_pts]
        total_frames = clip.shape[0]

        if start_time_s is None and end_time_s is None:
            frame_indices = list(range(total_frames))
            return clip, frame_indices, timestamps

        # Select frames within time window
        st = float(start_time_s) if start_time_s is not None else float("-inf")
        et = float(end_time_s) if end_time_s is not None else float("inf")
        selected: List[int] = [i for i, ts in enumerate(timestamps) if st <= ts <= et]
        if len(selected) == 0:
            # fallback, keep closest
            closest = min(range(total_frames), key=lambda i: abs(timestamps[i] - (st if st != float("-inf") else 0.0)))
            selected = [closest]
        clip_sel = clip.index_select(0, torch.tensor(selected, dtype=torch.long))
        ts_sel = [timestamps[i] for i in selected]
        return clip_sel, selected, ts_sel

    def process_image(self, image_file: str):
        if not os.path.exists(image_file):
            raise FileNotFoundError(f"Image file {image_file} not found")
        return Image.open(image_file).convert("RGB")

    # -------------- Item building --------------
    def getitem(self, i: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[i]
        image_inputs = None
        video_inputs = None

        if sample["format"] == "llava":
            entry = sample["data"]
            sources = [entry]
            conversation = None
            if "video" in entry:
                video_file = entry["video"]
                if not os.path.exists(video_file):
                    raise FileNotFoundError(f"Video file {video_file} not found")

                # Optional frame/time range (seconds) not provided by LLaVA JSON normally
                start_time_s = entry.get("video_start", None)
                end_time_s = entry.get("video_end", None)
                video, frame_indices, frame_timestamps = self.process_video(
                    video_file, start_time_s, end_time_s
                )

                if video.shape[0] == 0:
                    raise RuntimeError(f"Empty video after loading for sample {i}")

                if self.data_args.enable_time_prompt:
                    frames_per_input = max(1, int(self.data_args.frames_per_input))
                    all_videos, time_texts = self._build_time_prompt_segments_from_indices(
                        video_frames=video,
                        frame_indices=frame_indices,
                        frames_per_input=frames_per_input,
                        frame_timestamps=frame_timestamps,
                    )
                    conversation = preprocess_llava_conversation(
                        [e["conversations"] for e in sources], has_video=True, has_image=False
                    )

                    if self._should_apply_segment_reconstruction(sample) and len(all_videos) > 1:
                        rng = np.random.default_rng(
                            self.data_args.segment_reconstruction_seed
                            if self.data_args.segment_reconstruction_seed is not None
                            else self.data_args.__dict__.get("sampling_seed", None)
                        )
                        degree_flag = getattr(self.data_args, "segment_reconstruction_degree", False)
                        degree = float(rng.random()) if degree_flag else 1.0
                        clip_number = getattr(self.data_args, "segment_reconstruction_clip_number", None)
                        perm = self._compute_partial_permutation(
                            len(all_videos), degree, rng, clip_number
                        )
                        conversation = self._inject_time_prompts_into_conversation(
                            conversation, all_videos, time_texts
                        )
                        is_changed = any(idx != i for i, idx in enumerate(perm))
                        if is_changed:
                            video_inputs = [all_videos[p] for p in perm]
                        else:
                            video_inputs = all_videos
                        correct_texts = [time_texts[p] for p in perm]
                        question = self._choose_reconstruction_prompt(rng)
                        joiner = (
                            getattr(self.data_args, "segment_reconstruction_answer_joiner", " | ")
                            or " | "
                        )
                        answer_template = getattr(
                            self.data_args,
                            "segment_reconstruction_answer_template",
                            "Correct timestamps (in displayed order): {timestamps_joined}",
                        )
                        answer = answer_template.format(
                            timestamps_joined=joiner.join(correct_texts)
                        )
                        conversation = self._prepend_reconstruction_qa_llava(
                            conversation, question, answer
                        )
                    else:
                        conversation = self._inject_time_prompts_into_conversation(
                            conversation, all_videos, time_texts
                        )
                        video_inputs = all_videos
                else:
                    conversation = preprocess_llava_conversation(
                        [e["conversations"] for e in sources], has_video=True, has_image=False
                    )
                    video_inputs = [video]
            elif "image" in entry:
                image_file = entry["image"]
                if isinstance(image_file, list):
                    images = [self.process_image(f) for f in image_file]
                else:
                    images = [self.process_image(image_file)]
                image_inputs = images
                conversation = preprocess_llava_conversation(
                    [e["conversations"] for e in sources], has_video=False, has_image=True
                )
            else:
                conversation = preprocess_llava_conversation(
                    [e["conversations"] for e in sources], has_video=False, has_image=False
                )
        else:
            conversation_raw = self._read_qwen_conversation(sample)
            conversation, image_inputs, video_inputs = self._process_qwen_conversation(
                conversation_raw, sample
            )

        texts = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False, return_tensors="pt"
        )
        inputs = self.processor(
            text=texts, images=image_inputs, videos=video_inputs, return_tensors="pt"
        )
        input_ids = inputs.input_ids
        labels = torch.full_like(input_ids, fill_value=-100, dtype=input_ids.dtype)
        im_start_idxs = (input_ids == self.im_start_id).nonzero()
        im_end_idxs = (input_ids == self.im_end_id).nonzero()
        for (sample_idx, im_start_idx), (sample_idx2, im_end_idx) in zip(
            im_start_idxs, im_end_idxs
        ):
            if sample_idx != sample_idx2:
                continue
            if input_ids[sample_idx, im_start_idx + 1] == self.assistant_id:
                labels[sample_idx, im_start_idx + 3 : im_end_idx + 1] = input_ids[
                    sample_idx, im_start_idx + 3 : im_end_idx + 1
                ]
        inputs["labels"] = labels
        
        return inputs

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
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

    # -------------- Helpers --------------

    def _build_time_prompt_segments_from_indices(
        self,
        video_frames: torch.Tensor,
        frame_indices: List[int],
        frames_per_input: int,
        frame_timestamps: List[float],
    ) -> Tuple[List[torch.Tensor], List[str]]:
        total_frames = int(video_frames.shape[0])
        all_videos: List[torch.Tensor] = []
        time_texts: List[str] = []
        if total_frames <= 0 or len(frame_indices) != total_frames:
            return all_videos, time_texts

        num_segments = (total_frames + frames_per_input - 1) // frames_per_input
        
        for seg_idx in range(num_segments):
            seg_start = seg_idx * frames_per_input
            seg_end = min((seg_idx + 1) * frames_per_input, total_frames)
            segment_frames = video_frames[seg_start:seg_end]
            ts_seg = (
                frame_timestamps[seg_start : seg_end+1]
                if seg_end+1 <= total_frames
                else frame_timestamps[seg_start:]
            )
            time_text = f"Time={ts_seg[0]:07.1f}-{ts_seg[-1]:07.1f}s"
            time_texts.append(time_text)
            all_videos.append(segment_frames)
        return all_videos, time_texts

    def _inject_time_prompts_into_conversation(
        self,
        conversations: List[List[Dict[str, Any]]],
        all_videos: List[torch.Tensor],
        time_texts: List[str],
    ) -> List[List[Dict[str, Any]]]:
        if not conversations:
            return conversations
        conv = conversations[0]
        expanded = False
        for msg in conv:
            if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                contents = msg["content"]
                has_video_marker = any(
                    (isinstance(c, dict) and c.get("type") == "video") for c in contents
                )
                if not has_video_marker:
                    continue
                new_contents: List[Dict[str, Any]] = []
                for c in contents:
                    if isinstance(c, dict) and c.get("type") == "video":
                        for t in time_texts:
                            new_contents.append({"type": "text", "text": t})
                            new_contents.append({"type": "video"})
                    else:
                        new_contents.append(c)
                msg["content"] = new_contents
                expanded = True
                break
        if not expanded:
            logger.warning(
                "Time-prompt injection found no video slot to expand; leaving conversation unchanged."
            )
        return conversations

    def _read_qwen_conversation(self, sample: Dict[str, Any]):
        if "data" in sample:
            return sample["data"]
        path = sample["path"]
        seek = int(sample["seek"])
        with open(path, "r") as f:
            f.seek(seek)
            line = f.readline()
        try:
            data = json.loads(line)
        except Exception as e:
            raise ValueError(f"Failed to parse JSONL at seek {seek} in {path}: {e}")
        return data

    def _process_qwen_conversation(
        self, conversation_raw: List[Dict[str, Any]], sample: Dict[str, Any]
    ):
        image_inputs: List[Any] = []
        video_inputs: List[torch.Tensor] = []
        conversation: List[Dict[str, Any]] = []
        applied_reconstruction: bool = False
        accumulated_correct_texts: List[str] = []

        seed = (
            self.data_args.segment_reconstruction_seed
            if getattr(self.data_args, "segment_reconstruction_seed", None) is not None
            else getattr(self.data_args, "sampling_seed", None)
        )
        rng = np.random.default_rng(seed)
        if not conversation_raw or not isinstance(conversation_raw, list):
            raise ValueError("Qwen2VL conversation must be a list of messages")

        for msg in conversation_raw:
            role = msg.get("role")
            content = msg.get("content", [])
            if role not in ("user", "assistant", "system"):
                continue
            new_msg = {"role": role, "content": []}
            if role == "system":
                for elem in content:
                    if (
                        isinstance(elem, dict)
                        and elem.get("type") == "text"
                        and "text" in elem
                    ):
                        new_msg["content"].append({"type": "text", "text": elem["text"]})
                conversation.append(new_msg)
                continue
            for elem in content:
                if not isinstance(elem, dict):
                    continue
                elem_type = elem.get("type")
                if elem_type == "text":
                    text_val = elem.get("text") or elem.get("text_stream") or ""
                    if isinstance(text_val, str) and text_val:
                        new_msg["content"].append({"type": "text", "text": text_val})
                elif elem_type == "image":
                    image_ref = elem.get("image")
                    if isinstance(image_ref, str) and os.path.exists(image_ref):
                        try:
                            img = Image.open(image_ref).convert("RGB")
                            image_inputs.append(img)
                            new_msg["content"].append({"type": "image"})
                        except Exception:
                            logger.warning(f"Failed to open image: {image_ref}")
                    else:
                        new_msg["content"].append({"type": "image"})
                elif elem_type == "video":
                    video_ref = elem.get("video")
                    segment_videos: List[torch.Tensor] = []
                    if isinstance(video_ref, str) and os.path.exists(video_ref):
                        video_start_time = elem.get("video_start", None)
                        video_end_time = elem.get("video_end", None)
                        # Load entire clip and optionally slice by seconds
                        video, frame_indices, frame_timestamps = self.process_video(
                            video_ref, start_time_s=video_start_time, end_time_s=video_end_time
                        )
                        
                        if video.shape[0] == 0:
                            continue
                        if self.data_args.enable_time_prompt:
                            frames_per_input = max(1, int(self.data_args.frames_per_input))
                            segment_videos, time_texts = self._build_time_prompt_segments_from_indices(
                                video_frames=video,
                                frame_indices=frame_indices,
                                frames_per_input=frames_per_input,
                                frame_timestamps=frame_timestamps,
                            )
                            
                            for t in time_texts:
                                new_msg["content"].append({"type": "text", "text": t})
                                new_msg["content"].append({"type": "video"})
                            if self._should_apply_segment_reconstruction(sample) and len(segment_videos) > 1:
                                degree_flag = getattr(self.data_args, "segment_reconstruction_degree", False)
                                degree = float(rng.random()) if degree_flag else 1.0
                                clip_number = getattr(self.data_args, "segment_reconstruction_clip_number", None)
                                perm = self._compute_partial_permutation(
                                    len(segment_videos), degree, rng, clip_number
                                )
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
                            new_msg["content"].append({"type": "video"})
                            video_inputs.append(video)
                    else:
                        new_msg["content"].append({"type": "video"})
                else:
                    continue
            conversation.append(new_msg)

        if applied_reconstruction and len(accumulated_correct_texts) > 0:
            question = self._choose_reconstruction_prompt(rng)
            joiner = (
                getattr(self.data_args, "segment_reconstruction_answer_joiner", " | ")
                or " | "
            )
            answer_template = getattr(
                self.data_args,
                "segment_reconstruction_answer_template",
                "Correct timestamps (in displayed order): {timestamps_joined}",
            )
            answer = answer_template.format(
                timestamps_joined=joiner.join(accumulated_correct_texts)
            )
            conversation = self._prepend_reconstruction_qa_llava([conversation], question, answer)
            conversation = conversation[0]
        return conversation, (image_inputs or None), (video_inputs or None)

    def _should_apply_segment_reconstruction(self, sample: Dict[str, Any]) -> bool:
        if not getattr(self.data_args, "enable_time_prompt", False):
            return False
        if not getattr(self.data_args, "enable_segment_reconstruction", False):
            return False
        try:
            src = sample.get("source")
        except Exception:
            src = None
        paths = getattr(self.data_args, "segment_reconstruction_paths", []) or []
        if src not in paths:
            return False
        prob = getattr(self.data_args, "segment_reconstruction_prob", 1.0)
        if prob >= 1.0:
            return True
        if prob <= 0.0:
            return False
        try:
            sample_id = str(src)
            if "seek" in sample:
                sample_id += f"_{sample['seek']}"
            elif "data" in sample:
                data_str = json.dumps(sample["data"], sort_keys=True)
                data_hash = hashlib.md5(data_str.encode()).hexdigest()[:8]
                sample_id += f"_{data_hash}"
            seed = (
                self.data_args.segment_reconstruction_seed
                if getattr(self.data_args, "segment_reconstruction_seed", None) is not None
                else getattr(self.data_args, "sampling_seed", 42)
            )
            combined_seed = hash(sample_id) % (2**31) + seed
            rng = np.random.default_rng(combined_seed)
            return bool(rng.random() < prob)
        except Exception as e:
            logger.warning(
                f"Error in random threshold check for segment reconstruction: {e}, defaulting to True"
            )
            return True

    def _choose_reconstruction_prompt(
        self, rng: Optional[np.random.Generator] = None
    ) -> str:
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
            "Time prompts do not match the videos. Provide each segment's correct time span.",
        ]
        if rng is None:
            idx = np.random.randint(0, len(prompts))
        else:
            idx = int(rng.integers(0, len(prompts)))
        return prompts[idx]

    def _prepend_reconstruction_qa_llava(
        self, conversations, question: str, answer: str
    ):
        if not conversations or not isinstance(conversations, list):
            return conversations
        if conversations and isinstance(conversations[0], dict) and "role" in conversations[0]:
            conv_container = None
            conv = conversations
        else:
            conv_container = conversations
            conv = conversations[0] if conversations else []
        if not conv or not isinstance(conv, list):
            return conversations
        first_video_idx = None
        for idx, msg in enumerate(conv):
            if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                if any((isinstance(c, dict) and c.get("type") == "video") for c in msg["content"]):
                    first_video_idx = idx
                    break
        if first_video_idx is None:
            first_video_idx = 1 if (len(conv) > 0 and conv[0].get("role") == "system") else 0
        original_user = conv[first_video_idx]
        original_assistant = conv[first_video_idx + 1]
        originl_first_question = original_user["content"][-1]["text"]
        original_user["content"][-1]["text"] = question
        conv[first_video_idx] = original_user
        conv[first_video_idx + 1] = {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        conv.insert(first_video_idx + 2, {"role": "user", "content": [{"type": "text", "text": originl_first_question}]})
        conv.insert(first_video_idx + 3, original_assistant)
        return conv_container


if __name__ == "__main__":
    import argparse
    from transformers import AutoProcessor

    parser = argparse.ArgumentParser(description="Test QwenVLDataset construction and preprocessing")
    parser.add_argument(
        "--path",
        type=str,
        default="/2024233235/videollm-online/EyeWO2/data/cof_qwen2vl.jsonl",
        help="Path to the dataset file (JSON or JSONL)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2-VL-7B-Instruct",
        help="Processor/Tokenizer model name to use for testing",
    )
    args = parser.parse_args()

    print(f"Loading processor: {args.model_name}")
    processor = AutoProcessor.from_pretrained(args.model_name, padding_side="right")
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("Processor must expose a tokenizer for QwenVLDataset test")
    if getattr(tokenizer, "pad_token_id", None) is None:
        if getattr(tokenizer, "eos_token_id", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise ValueError("Tokenizer has no pad_token_id and no eos_token_id; please choose a different tokenizer")

    data_args = DataArguments(
        annotation_paths=[
            # "/2024233235/.cache/huggingface/hub/datasets--lmms-lab--EgoIT-99K/snapshots/a57f1f2078a7b01ea87014050fdb3afe169e54f1/datasets/EgoIT_min448frames.json", 
            '/2024233235/videollm-online/EyeWO2/data/llava_video_178k_with_seeks_sample_valid.jsonl',
            # '/2024233235/videollm-online/EyeWO2/data/qwen_format_split_videoQA.jsonl',
            # '/2024233235/videollm-online/EyeWO2/data/qwen_format_random_videoQA.jsonl'
            # "/2024233235/videollm-online/EyeWO2/data/etbench_qwen2vl_timestamp.jsonl",
            # "/2024233235/videollm-online/EyeWO2/data/cof_qwen2vl_timestamp.jsonl",
            
            ],
        processor=processor,
        enable_time_prompt=True,
        enable_segment_reconstruction=True,
        segment_reconstruction_paths=[
            "/2024233235/videollm-online/EyeWO2/data/llava_video_178k_with_seeks_sample_valid.jsonl",
            "/2024233235/videollm-online/EyeWO2/data/cof_qwen2vl.jsonl",
        ],
        segment_reconstruction_clip_number=4,
    )

    print(f"Constructing dataset from: {args.path}")
    dataset = QwenVLDataset(
        annotation_paths=data_args.annotation_paths,
        tokenizer=tokenizer,
        data_args=data_args,
    )
    print(f"Dataset loaded with {len(dataset)} samples")
    if len(dataset) == 0:
        print("Dataset is empty; nothing to preprocess.")
        raise SystemExit(0)

    out = dataset[0]
    print("First sample processed successfully.")

