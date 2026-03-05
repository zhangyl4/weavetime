import ast
import copy
import json
import logging
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
    frames_upbound: Optional[int] = field(default=30)
    force_sample: bool = False
    image_processor: Optional[Any] = None


# --------- some utils copy from https://github1s.com/EvolvingLMMs-Lab/EgoLife/blob/main/EgoGPT/egogpt/train/train_audio.py#L551-L627---------

CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
SPEECH_TOKEN_INDEX = -200
DEFAULT_SPEECH_TOKEN = "<speech>"
IMAGE_TOKEN_INDEX = -300
DEFAULT_IMAGE_TOKEN = "<image>"



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

def preprocess_qwen(
    sources,
    tokenizer,
    has_speech: bool = False,
    has_image: bool = False,
    max_len=2048,
    system_message: str = "You are a helpful assistant.",
) -> Dict:
    def split_text(text, keywords):
        pattern = "(" + "|".join(map(re.escape, keywords)) + ")"
        parts = re.split(pattern, text)
        parts = [part for part in parts if part]
        return parts

    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}

    # im_start, im_end = tokenizer.additional_special_tokens_ids

    im_start = tokenizer("<|im_start|>").input_ids[0]
    im_end = tokenizer("<|im_end|>").input_ids[0]
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []
        system = (
            [im_start]
            + _system
            + tokenizer(system_message).input_ids
            + [im_end]
            + nl_tokens
        )
        input_id += system
        target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens
        assert len(input_id) == len(target)
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            splited_sentence = split_text(sentence["value"], ["<speech>", "<image>"])
            _input_id = []
            for part in splited_sentence:
                _input_id += tokenizer(role).input_ids + nl_tokens  # add prefix
                if "<speech>" == part:
                    _input_id += [SPEECH_TOKEN_INDEX]
                elif "<image>" == part:
                    _input_id += [IMAGE_TOKEN_INDEX]
                else:
                    _input_id += tokenizer(part).input_ids
            _input_id += [im_end] + nl_tokens  # add suffix
            input_id += _input_id
            if role == "<|im_start|>user":
                _target = (
                    [im_start]
                    + [IGNORE_INDEX] * (len(_input_id) - 3)
                    + [im_end]
                    + nl_tokens
                )
            elif role == "<|im_start|>assistant":
                _target = (
                    [im_start]
                    + [IGNORE_INDEX] * len(tokenizer(role).input_ids)
                    + _input_id[len(tokenizer(role).input_ids) + 1 : -2]
                    + [im_end]
                    + nl_tokens
                )
            else:
                raise NotImplementedError
            target += _target
        assert len(input_id) == len(target)
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    breakpoint()
    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )
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
        list_data_dict = []
        for annotation_path in annotation_paths:
            list_data_dict.extend(json.load(open(annotation_path, "r")))
        
        logging.warning("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        # Basic stride info for sampler compatibility
        self.video_info = [1 for _ in range(len(self.list_data_dict))]

    def __len__(self):
        return len(self.list_data_dict)

    def process_video(self, video_file, start_frame=None, end_frame=None, current_observation_frame=None):
        """Process video frames using decord helpers above."""
        if not os.path.exists(video_file):
            print(f"File {video_file} does not exist!")
            raise FileNotFoundError(f"Video file {video_file} not found")
        if start_frame is not None and end_frame is not None:
            video = process_video_with_decord_byframe(
                video_file,
                start_frame,
                end_frame,
                self.data_args,
                current_observation_frame,
            )
        else:
            video, video_time, frame_time, num_frames = process_video_with_decord(
                video_file, self.data_args
            )
        return video

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

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"

        image = None
        # if "image" in sources[0]:
        #     image_file = self.list_data_dict[i]["image"]
        #     if isinstance(image_file, list):
        #         image = [self.process_image(f) for f in image_file]
        #         if len(image_file) > 1:
        #             image = [self.process_image(f, "pad") for f in image_file]
        #             image = [[im[0], im[1], "image"] for im in image]
        #     else:
        #         image = [self.process_image(image_file)]

        if ("video" in sources[0]):
            video_file = self.list_data_dict[i]["video"]
            if not os.path.exists(video_file):
                print("File {} not exist!".format(video_file))
            if "start_frame" in self.list_data_dict[i]:
                start_frame = self.list_data_dict[i]["start_frame"]
                end_frame = self.list_data_dict[i]["end_frame"]
                if self.list_data_dict[i].get("current_observation_frame", None):
                    current_observation_frame = self.list_data_dict[i]["current_observation_frame"]
                else:
                    current_observation_frame = None
                video = self.process_video(
                    video_file,
                    start_frame,
                    end_frame,
                    current_observation_frame,
                )
            else:
                video, video_time, frame_time, num_frames = process_video_with_decord(
                    video_file, self.data_args
                )
            processor = self.data_args.image_processor
            # breakpoint()
            processed_video = processor.preprocess(video, return_tensors="pt")["pixel_values_videos"]
            image = [(processed_video, Image.fromarray(video[0]).size, "video")]

        # conversations
        sources_proc = preprocess_multimodal(
            copy.deepcopy([e["conversations"] for e in sources]), self.data_args
        )
        data_dict = preprocess_qwen(
            sources_proc, self.tokenizer
        )
        if isinstance(i, int):
            data_dict = dict(
                input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0]
            )

        if image is not None:
            data_dict["image"] = image
        return data_dict

@dataclass 
class DataCollatorForLLaVAOV:
    """Collate examples for supervised fine-tuning."""

    tokenizer: PreTrainedTokenizer

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=batch_first, padding_value=padding_value
        )
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = [
            _input_ids[: self.tokenizer.model_max_length] for _input_ids in input_ids
        ]
        labels = [_labels[: self.tokenizer.model_max_length] for _labels in labels]
        if self.tokenizer.pad_token_id is None:
            if "qwen" in self.tokenizer.name_or_path.lower():
                # print("Setting pad token to bos token for qwen model.")
                self.tokenizer.pad_token_id = 151643
            else:
                self.tokenizer.pad_token_id = (
                    self.tokenizer.eos_token_id
                )  # FIXME: this could only be triggered for llama3 model.
        input_ids = self.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = self.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        if "speech" in instances[0]:
            speeches = [instance["speech"] for instance in instances]
            speeches_lengths = [instance["speech_lengths"] for instance in instances]
            batch["speech"] = [au for audio_list in speeches for au in audio_list]

            batch["speech_lengths"] = [
                au for audio_list in speeches_lengths for au in audio_list
            ]
            batch["speech_lengths"] = torch.stack(batch["speech_lengths"])

            if all(
                x is not None and x.shape == speeches[0][0].shape
                for x in batch["speech"]
            ):
                batch["speech"] = torch.stack(batch["speech"])

        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]

            batch["image_sizes"] = [im[1] for im_list in images for im in im_list]
            batch["modalities"] = [im[2] for im_list in images for im in im_list]
            images = [im[0] for im_list in images for im in im_list]

            # if all(x is not None and x.shape == images[0].shape for x in images):
            # Image: (N, P, C, H, W)
            # Video: (N, F, C, H, W)
            #     batch["images"] = torch.stack(images)
            # else:
            batch["images"] = images
        return batch

    

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
    
    data_args = DataArguments(annotation_paths=[args.path], image_processor=processor.video_processor)

    print(f"Constructing dataset from: {args.path}")
    dataset = LLaVAOVDataset(
        annotation_paths=[args.path],
        tokenizer=tokenizer,
        data_args=data_args,
    )
    print(f"Dataset loaded with {len(dataset)} samples")

    if len(dataset) == 0:
        print("Dataset is empty; nothing to preprocess.")
        raise SystemExit(0)

    out = dataset[0]

    print("Preprocessing complete.")
    print(f"input_ids shape: {tuple(out['input_ids'].shape)}")
    print(f"labels shape: {tuple(out['labels'].shape)}")

    print("Building a test batch with the data collator...")
    collator = DataCollatorForLLaVAOV(tokenizer=tokenizer)
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=collator)
    i = 0
    for batch in dataloader:
        print("Batch contents:")
        breakpoint()
        for key, value in batch.items():
            if hasattr(value, 'shape'):
                print(f"{key}: shape {tuple(value.shape)}")
            else:
                print(f"{key}: {type(value)}")
        i+=1
        if i % 10 == 0:
            print(f"Batch {i}")
        if i > 100:
            break
        