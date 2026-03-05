from dataclasses import dataclass, field
from typing import Optional
import json, torch, random, tqdm, io, functools, os
from PIL import Image
from torch.utils.data import Dataset
from transformers import logging, AutoProcessor
from torchvision.transforms.functional import pil_to_tensor

from livecc_utils import _read_video_decord_plus, _spatial_resize_video
from qwen_vl_utils.vision_process import smart_nframes, process_vision_info, FPS, VIDEO_TOTAL_PIXELS, VIDEO_MIN_PIXELS, FPS_MAX_FRAMES, FORCE_QWENVL_VIDEO_READER

logger = logging.get_logger(__name__)

logger.warning(f'{__name__}: {FORCE_QWENVL_VIDEO_READER=}, {FPS_MAX_FRAMES=}, {VIDEO_MIN_PIXELS=}, {VIDEO_TOTAL_PIXELS=}')

@dataclass
class DataArguments:
    annotation_paths: list[str] = field(default_factory=list)
    initial_fps_frames: int = int(FPS) * 3
    streaming_fps_frames: int = int(FPS)
    with_context: bool = False
    # HACK: max clip frames for streaming
    max_clip_frames: Optional[int] = 300

# --- some utils ---
def readlastline(path: str):
    with open(path, "rb") as f:
        f.seek(-2, 2) # avoid last \n
        while f.read(1) != b"\n":  
            f.seek(-2, 1)
        return f.readline()

def bytes_to_pil(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode == 'P':
        image = image.convert('RGBA')
    return image.convert('RGB')

def get_phrase_before_timestamp(text_stream, timestamp, start_from: int = 0, multi_turn: bool = False):
    phrase = '' if not multi_turn else []
    # FIX BUG: if the last word is finished, the pointer will move forward, lead to last element be used repeatedly
    pointer = 0
    for i, (ws, we, word) in enumerate(text_stream[start_from:]):
        if timestamp >= we:
            pointer += 1
            phrase += ' ' + word.strip() if not multi_turn else [word.strip()]
        else:
            break
    return phrase.strip() if not multi_turn else phrase, start_from + pointer
# --- some utils ---

class LMMDataset(Dataset):
    def __init__(
        self, *, annotation_paths: list[str], processor: AutoProcessor, 
        initial_fps_frames: int = DataArguments.initial_fps_frames, streaming_fps_frames: int = DataArguments.streaming_fps_frames, 
        with_context: str = DataArguments.with_context, 
        max_clip_frames: int = DataArguments.max_clip_frames,
        **kwargs
    ):
        super().__init__()
        self.handles = []
        # HACK: save temp file for fast start up
        self.video_info = []
        for annotation_path in annotation_paths:
            assert annotation_path.endswith('.jsonl'), f"Please organize the annotations in JSONL format, with each data sample on a separate line, and the last line stores the seek indices"
            logger.warning(f'Load {annotation_path}. Please ensure its last line stores the seek indices...')
            # seeks = json.loads(readlastline(annotation_path))
            if not os.path.exists(f'{annotation_path.replace(".jsonl", ".seek.json")}'):
                seeks = json.loads(readlastline(annotation_path))
                json.dump(seeks, open(f'{annotation_path.replace(".jsonl", ".seek.json")}', 'w'))
            else:
                seeks = json.load(open(f'{annotation_path.replace(".jsonl", ".seek.json")}'))
            self.video_info.extend(json.load(open(annotation_path.replace('.jsonl', '_video_info.json'))))
            self.handles.extend(zip([annotation_path] * len(seeks), seeks))
            print("seek len",len(seeks))
            print("seek max",max(seeks))
            print("video info len",len(self.video_info))
            print("video info max",max(self.video_info))
            logger.warning(f'Successfully loaded {annotation_path}')
        if 'Qwen2VL' in processor.__class__.__name__:
            self.im_start_id, self.assistant_id, self.newline_id, self.im_end_id = processor.tokenizer(
                '<|im_start|>assistant\n<|im_end|>').input_ids
        else:
            raise NotImplementedError(f"Video preprocess not implemented for {processor.__class__.__name__}")
        self.processor = processor
        self.with_context = with_context
        self.initial_fps_frames = initial_fps_frames
        self.streaming_fps_frames = streaming_fps_frames
        # HACK: max clip frames for streaming
        self.max_clip_frames = max_clip_frames
        
    def load_conversation(self, index):
        try:
            annotation_path, seek = self.handles[index]
        except:
            print("error", index)
            exit()
        with open(annotation_path) as f:
            f.seek(seek)
            line = f.readline()
        line = json.loads(line)
        return line

    def preprocess_image(self, element: dict):
        if hasattr(self, 'remote_loader'):
            return Image.open(self.remote_loader(element['image']))
        return element['image']
    
    def preprocess_video(self, element: dict):
        if 'pos' in element: # for sharegpt. implement smart_nframes and smart_resize for pil images video
            positions = [0] + element['pos']
            nframes = smart_nframes(element, total_frames=len(positions) - 1, video_fps=FPS)
            sampler = torch.linspace(0, len(positions) - 2, nframes).round().long()
            data_bytes = self.remote_loader(element['video'], length_check=True, return_io=False)
            video = torch.stack([pil_to_tensor(bytes_to_pil(data_bytes[positions[i]:positions[i+1]])) for i in sampler])
            video = _spatial_resize_video(video)
            return video
        return element['video']

    def preprocess_text(self, element: str):
        if self.with_context and ('title' in element or 'previous' in element):
            previous = element.get('previous', '')
            if previous:
                title = ''
            else:
                title = element.get('title', '')
            return (element['text'] + f"\n{title}\n{previous}").strip()
        return element['text']

    def preprocess_conversation_stream(self, conversation: list):
        user_message, assistant_message = conversation
        user_content, assistant_content = user_message['content'], assistant_message['content']
        user_video_dict, user_query_dict = user_content
        assert 'video' in user_video_dict, 'please check your data, ensure the video info in the first user content'
        assistant_text_stream = assistant_message['content'][0]['text_stream']
        
        # load video in strict fps
        clip, _, clip_pts = _read_video_decord_plus(user_video_dict, return_pts=True, strict_fps=True)
        clip = _spatial_resize_video(clip)

        # make conversation
        start_timestamp, end_timestamp = 0, self.initial_fps_frames / FPS
        
        phrase, next_start_from = get_phrase_before_timestamp(assistant_text_stream, clip_pts[self.initial_fps_frames - 1], multi_turn='LLaVA' in user_video_dict['video'])
        
        # HACK：multi user query
        if "text_stream" in user_query_dict:
            user_phrase, user_next_start_from = get_phrase_before_timestamp(user_query_dict['text_stream'], clip_pts[self.initial_fps_frames - 1], multi_turn='LLaVA' in user_video_dict['video'])
        else:
            user_phrase, user_next_start_from = user_query_dict['text'], None
        
        conversation = [
            {
                'role': 'user', 'content': [
                    {'type': 'text', 'text': f'Time={start_timestamp:.1f}-{end_timestamp:.1f}s'}, 
                    {'type': 'video', 'video': clip[:self.initial_fps_frames]},
                ],
            },
            {'role': 'assistant', 'content': [{'type': 'text', 'text':  ' ...'}]} # ' ...' denotes the streaming is not ended
        ]
        
        if isinstance(user_phrase, list) and isinstance(phrase, list):
            assert len(user_phrase) == len(phrase), 'please check your data, ensure the user query and assistant response are the same length'
            for ii, (user_phrase_i, phrase_i) in enumerate(zip(user_phrase, phrase)):
                conversation[-2]['content'].append({'type': 'text', 'text': user_phrase_i})
                conversation[-1]['content'] = [{'type': 'text', 'text':  phrase_i}]
        elif user_phrase != '' or phrase != '':
            if user_phrase != '':            
                conversation[-2]['content'].append({'type': 'text', 'text': user_phrase})
            if phrase != '':
                conversation[-1]['content'] = [{'type': 'text', 'text':  phrase}]

        frames_list = [clip[:self.initial_fps_frames]]
        for i in range(self.initial_fps_frames, len(clip), self.streaming_fps_frames):
            start_timestamp, end_timestamp = i / FPS, (i + self.streaming_fps_frames) / FPS
            phrase, next_start_from = get_phrase_before_timestamp(assistant_text_stream, clip_pts[i + self.streaming_fps_frames-1], start_from=next_start_from, multi_turn='LLaVA' in user_video_dict['video'])
            # HACK：multi user query
            if "text_stream" in user_query_dict:
                user_phrase, user_next_start_from = get_phrase_before_timestamp(user_query_dict['text_stream'], clip_pts[i + self.streaming_fps_frames-1], start_from=user_next_start_from, multi_turn='LLaVA' in user_video_dict['video'])
            else:
                user_phrase, user_next_start_from = '', None
            frames = clip[i:i + self.streaming_fps_frames]
            
            conversation.extend([
                {
                    'role': 'user', 'content': [
                        {'type': 'text', 'text': f'Time={start_timestamp:.1f}-{end_timestamp:.1f}s'}, 
                        {'type': 'video', 'video': frames},
                    ]
                },
                {'role': 'assistant', 'content': [{'type': 'text', 'text':  ' ...'}]} 
            ])
            
            
            if isinstance(user_phrase, list) and isinstance(phrase, list):
                assert len(user_phrase) == len(phrase), 'please check your data, ensure the user query and assistant response are the same length'
                for ii, (user_phrase_i, phrase_i) in enumerate(zip(user_phrase, phrase)):
                    conversation[-2]['content'].append({'type': 'text', 'text': user_phrase_i})
                    conversation[-1]['content'] = [{'type': 'text', 'text':  phrase_i}]
            elif user_phrase != '' or phrase != '':
                if user_phrase != '':            
                    conversation[-2]['content'].append({'type': 'text', 'text': user_phrase})
                if phrase != '':
                    conversation[-1]['content'] = [{'type': 'text', 'text':  phrase}]
            frames_list.append(frames)
        # remove the last with no phrase
        
        # HACK: prcocess video info error case
        if next_start_from == 0 or user_next_start_from == 0:
            phrase, next_start_from = get_phrase_before_timestamp(assistant_text_stream, 10000, start_from=next_start_from, multi_turn='LLaVA' in user_video_dict['video'])
            # HACK：multi user query
            if "text_stream" in user_query_dict:
                user_phrase, user_next_start_from = get_phrase_before_timestamp(user_query_dict['text_stream'], 10000, start_from=user_next_start_from, multi_turn='LLaVA' in user_video_dict['video'])
            else:
                user_phrase, user_next_start_from = '', None
            if isinstance(user_phrase, list) and isinstance(phrase, list):
                assert len(user_phrase) == len(phrase), 'please check your data, ensure the user query and assistant response are the same length'
                for ii, (user_phrase_i, phrase_i) in enumerate(zip(user_phrase, phrase)):
                    conversation[-2]['content'].append({'type': 'text', 'text': user_phrase_i})
                    conversation[-1]['content'] = [{'type': 'text', 'text':  phrase_i}]
            elif user_phrase != '':            
                conversation[-2]['content'].append({'type': 'text', 'text': user_phrase})
        try:
            while conversation[-1]['content'][0]['text'] == ' ...':
                conversation = conversation[:-2]
                frames_list = frames_list[:-1]
        except Exception as e:
            breakpoint()
            print(e)
            print(clip.shape)
            print(clip_pts)
            print(assistant_text_stream)
            print(user_query_dict)
            print(user_video_dict)
            exit()
        # HACK: max clip frames for streaming
        # if self.max_clip_frames is not None:
        #     conversation, frames_list = self.max_frames_clip(conversation, frames_list)
        return conversation, frames_list


    # HACK: max clip frames for streaming
    def max_frames_clip(self, conversation: list[dict], frames_list: list[torch.Tensor]):
        """
        conversation = [
            {
                'role': 'user', 'content': [
                    {'type': 'text', 'text': f'Time={start_timestamp:.1f}-{end_timestamp:.1f}s'}, 
                    {'type': 'video', 'video': clip[:self.initial_fps_frames]},
                    user_text_dict,
                ]
            },
            {'role': 'assistant', 'content': [{'type': 'text', 'text':  phrase + ' ...'}]} # ' ...' denotes the streaming is not ended,
            ...
        ]
        """
        cum_num_frames = 0
        cum_num_frame_padd = 0
        is_clip = False
        for i, message in enumerate(conversation):
            if message['role'] == 'user':
                for element in message['content']:
                    if element['type'] == 'video':
                        if cum_num_frames + element['video'].shape[0] > self.max_clip_frames:
                            conversation = conversation[:i] # remove the last message
                            is_clip = True
                            break
                        cum_num_frames += element['video'].shape[0]
                        cum_num_frame_padd += 1
            if is_clip:
                break
        assert conversation[-1]['role'] == 'assistant', 'please check your data, ensure the streaming is not ended'
        return conversation, frames_list[:cum_num_frame_padd]
    
    def getitem(self, index):
        conversation = self.load_conversation(index)

        special_process_for_stream, image_inputs, video_inputs = False, None, None
        for message in conversation:
            if message['role'] == 'user':
                for element in message['content']:
                    if 'text_stream' in element:
                        special_process_for_stream = 'text_stream' in element
                        break
                    if hasattr(self, 'remote_loader'):
                        element['remote_loader'] = self.remote_loader
                    modal = element['type']
                    element[modal] = getattr(self, f'preprocess_{modal}')(element)
                    if isinstance(element[modal], torch.Tensor):
                        if video_inputs is None:
                            video_inputs = [element[modal]]
                        else:
                            video_inputs.append(element[modal])
            else:
                for element in message['content']:
                    special_process_for_stream = 'text_stream' in element
                    break
        if special_process_for_stream:
            conversation, video_inputs = self.preprocess_conversation_stream(conversation)
            image_inputs = None
        else:
            if not video_inputs and not image_inputs:
                image_inputs, video_inputs = process_vision_info(conversation)
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
        return inputs

    def __getitem__(self, index):
        max_tries = 100
        for _ in range(max_tries):
            # try:
            return self.getitem(index)
            # except Exception as e:
            #     logger.warning(f"Failed {_}-th try to get item {index}: {e}")
            #     index = random.randint(0, self.__len__() - 1)
            #     logger.warning(f"Retrying to get item {index}")
        raise Exception(f"Failed to get item after {max_tries} retries")

    def data_collator(self, batched_inputs, **kwargs):
        assert len(batched_inputs) == 1
        return batched_inputs[0]

    def __len__(self):
        return len(self.handles)

if __name__ == "__main__":
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained('Qwen/Qwen2-VL-7B-Instruct', padding_side='right') 
    # model = Qwen2VLForConditionalGeneration.from_pretrained('Qwen/Qwen2-VL-7B', torch_dtype='auto', attn_implementation='flash_attention_2', device_map='cuda')
    # model.to('cuda')
    # print(processor.tokenizer.encode('<|im_start|>assistant\n ...<|im_end|>'))
    dataset = LMMDataset(
        annotation_paths=[
            # 'live_whisperx_526k_with_seeks.jsonl', 
            # '/2022233235/videollm-online/EyeWO2/data/llava_video_178k_with_seeks_sample_stream.jsonl', 
            # '/2022233235/videollm-online/EyeWO2/data/llava_video_178k_with_seeks_sample_valid_stream.jsonl',
            # '/2022233235/videollm-online/EyeWO2/data/qwen_format_split_videoQA.jsonl',
            # '/2022233235/videollm-online/EyeWO2/data/qwen_format_random_videoQA.jsonl'
            # 'llava_hound_video_with_seeks.jsonl', 
            # 'llava_ov_multi_image_with_seeks.jsonl', 
            # 'llava_ov_single_image_text_mix_with_seeks.jsonl'
            # "/2022233235/videollm-online/EyeWO2/data/etbench_qwen2vl.jsonl",
            "/2024233235/videollm-online/EyeWO2/data/cof_qwen2vl.jsonl",
            # '/2022233235/videollm-online/EyeWO2/data/estp_it_cqa_converted_withseeks.jsonl',
            # '/2022233235/videollm-online/EyeWO2/data/estp_it_sqa_converted_withseeks.jsonl'
        ], 
        processor=processor,
        with_context=False,
    )
    dataset[10000]
    # a = random.sample(range(len(dataset)), 100)
    # # a[13]
    # for i in tqdm.tqdm(random.sample(range(len(dataset)), 100)): # FPS_MAX_FRAMES=768 
    #     dataset[i]
    exit()
    

    
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=16, collate_fn=dataset.data_collator)
    
    for batch in tqdm.tqdm(dataloader):
        # dataset.processor.tokenizer.decode([785,    293,  14725,  15625,    702,    264,  22360,   3705,])
        pass
    # for i in tqdm.tqdm(range(len(dataset))):
    #     conversation = dataset.__getitem__(i)
        # inputs.to('cuda')
        # with torch.inference_mode():
        #     model(**inputs)