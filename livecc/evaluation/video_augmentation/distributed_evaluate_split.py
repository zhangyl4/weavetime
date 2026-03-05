import json, os, torch, functools, tqdm, random, sys
import numpy as np
import decord
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
    return clip, sample_fps

def create_video_with_middle_crop(video: torch.Tensor, crop_ratio: float = 0.3):
    """
    Create a video with middle part cropped out to simulate discontinuity.
    
    Args:
        video (torch.Tensor): Input video tensor with shape (T, C, H, W)
        crop_ratio (float): Ratio of frames to crop from the middle (0.0-1.0)
    
    Returns:
        torch.Tensor: Video with middle part cropped out
    """
    if  crop_ratio == 0:
        return video
    
    T, C, H, W = video.shape
    crop_start = int(T * (1 - crop_ratio) / 2)
    crop_end = int(T * (1 + crop_ratio) / 2)
    
    # Take first part and last part, skip middle
    first_part = video[:crop_start]
    last_part = video[crop_end:]
    
    # Concatenate first and last parts
    cropped_video = torch.cat([first_part, last_part], dim=0)
    
    return cropped_video, first_part.shape[0]

def save_function_print(function: callable, save_path: str, *args, **kwargs):
    original_stdout = sys.stdout
    try:
        with open(save_path, 'w') as f:
            sys.stdout = f  
            function(*args, **kwargs)          
    finally:
        sys.stdout = original_stdout 



class VideoContinuityDataset(Dataset):
    """
    Dataset for video continuity detection task.
    Reads videos and creates samples with middle crop to test if model can detect discontinuity.
    """
    def __init__(self, video_paths, remote_loader=None, crop_ratio=0.3, question_prefix="", question_postfix="\nPlease answer Yes or No.", answer_prefix="Answer:", sample=None, wo_stamp=False):
        """
        Args:
            video_paths (list): List of video file paths
            remote_loader (callable): Remote loader function for videos
            crop_ratio (float): Ratio of frames to crop from middle (0.0-1.0)
            question_prefix (str): Prefix for the question
            question_postfix (str): Postfix for the question
            answer_prefix (str): Prefix for the answer
            sample (int): Number of samples to use (for testing)
        """
        self.video_paths = video_paths
        if sample is not None:
            random.seed(42)
            self.video_paths = random.sample(self.video_paths, min(sample, len(self.video_paths)))
        
        self.remote_loader = remote_loader
        self.crop_ratio = crop_ratio
        self.question_prefix = question_prefix
        self.question_postfix = question_postfix
        self.answer_prefix = answer_prefix
        self.wo_stamp = wo_stamp
        
        # Create data samples with labels
        self.datums = []
        for i, video_path in enumerate(self.video_paths):
            # Create both continuous (original) and discontinuous (cropped) versions
            self.datums.append({
                'id': f"{i}_continuous",
                'video_path': video_path,
                'is_continuous': True,
                'crop_ratio': 0.0  # No crop for continuous
            })
            self.datums.append({
                'id': f"{i}_discontinuous", 
                'video_path': video_path,
                'is_continuous': False,
                'crop_ratio': self.crop_ratio  # Crop for discontinuous
            })
        print(f"Created {len(self.datums)} samples")
        
    def __len__(self):
        return len(self.datums)

    def __getitem__(self, i):
        datum = self.datums[i]
        conversation = [{"role": "user", "content": []}]
        
        # Read the full video
        video, _ = _read_may1fps_video_decord({
            'video': datum['video_path'], 
            'video_start':0, 'video_end': 240,
            'remote_loader': self.remote_loader
        })
        T_original = video.shape[0]
        duration = T_original / FPS
        
        # Apply middle crop if needed
        crop_index = None
        if datum['crop_ratio'] > 0:
            video, crop_index  = create_video_with_middle_crop(video, datum['crop_ratio'])
        
        # Resize video
        video = _spatial_resize_video(video)
        
        # Create question
        query = self.question_prefix + "Is this video continuous without any cuts or jumps in the middle?" + self.question_postfix
        
        # Add video with timestamps and text to conversation
        if self.wo_stamp:
            video_inputs = []
            streaming_fps_frames = int(FPS)
            T = len(video)
            crop_frame = crop_index if crop_index is not None else None
            crop_time = datum['crop_ratio'] * duration if crop_frame is not None else 0
            for idx in range(0, T, streaming_fps_frames):
                start_timestamp, end_timestamp = idx / FPS, min((idx + streaming_fps_frames), T) / FPS
                # 判断当前clip是否跨越断裂点
                if crop_frame is not None and idx < crop_frame < idx + streaming_fps_frames:
                    # 前半段
                    pre_end = crop_frame
                    pre_start_timestamp, pre_end_timestamp = idx / FPS, pre_end / FPS
                    conversation[0]['content'].append({'type': 'text', 'text': f'Time={pre_start_timestamp:.1f}-{pre_end_timestamp:.1f}s'})
                    conversation[0]['content'].append({'type': 'video', 'video': video[idx:pre_end]})
                    video_inputs.append(video[idx:pre_end])
                    # 后半段，时间戳整体右移
                    post_start = crop_frame
                    post_end = min(idx + streaming_fps_frames, T)
                    post_start_timestamp = post_start / FPS + crop_time
                    post_end_timestamp = post_end / FPS + crop_time
                    conversation[0]['content'].append({'type': 'text', 'text': f'Time={post_start_timestamp:.1f}-{post_end_timestamp:.1f}s'})
                    conversation[0]['content'].append({'type': 'video', 'video': video[post_start:post_end]})
                    video_inputs.append(video[post_start:post_end])
                elif crop_frame is not None and idx >= crop_frame:
                    # 断裂后所有时间戳右移
                    adj_start = start_timestamp + crop_time
                    adj_end = end_timestamp + crop_time
                    conversation[0]['content'].append({'type': 'text', 'text': f'Time={adj_start:.1f}-{adj_end:.1f}s'})
                    conversation[0]['content'].append({'type': 'video', 'video': video[idx:min(idx+streaming_fps_frames, T)]})
                    video_inputs.append(video[idx:min(idx+streaming_fps_frames, T)])
                else:
                    conversation[0]['content'].append({'type': 'text', 'text': f'Time={start_timestamp:.1f}-{end_timestamp:.1f}s'})
                    conversation[0]['content'].append({'type': 'video', 'video': video[idx:min(idx+streaming_fps_frames, T)]})
                    video_inputs.append(video[idx:min(idx+streaming_fps_frames, T)])
            conversation[0]['content'].append({'type': 'text', 'text': query})
        else:
            conversation[0]['content'].append({'type': 'video', 'video': video})
            conversation[0]['content'].append({'type': 'text', 'text': query})
            video_inputs = [video]
            # video_inputs = None
        
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
        return inputs


def preprocess_logits_for_metrics(logits, labels, strict_option_ids): 
    return torch.stack([logit[(logit[:, 0] != -100).nonzero().squeeze()[-1], strict_option_ids] for logit in logits]).argmax(dim=-1)

def video_continuity_predict(
    model, 
    processor, 
    video_paths: list, 
    remote_loader: callable = None,
    crop_ratio: float = 0.3,
    question_prefix: str = '', 
    question_postfix: str = '\nPlease answer Yes or No.', 
    answer_prefix: str = 'Answer:', 
    yes_no_previous_str: str = ' ',
    use_liger_kernel: bool = True,
    per_device_eval_batch_size: int = 1,
    dataloader_num_workers: int = 2,
    sample: int = None,
    wo_stamp:bool = False,
):
    """
    Predict video continuity using Yes/No classification.
    
    Args:
        model: The model to use for prediction
        processor: The processor for the model
        video_paths: List of video file paths
        remote_loader: Remote loader function for videos
        crop_ratio: Ratio of frames to crop from middle
        question_prefix: Prefix for the question
        question_postfix: Postfix for the question
        answer_prefix: Prefix for the answer
        yes_no_previous_str: String before Yes/No options
        use_liger_kernel: Whether to use liger kernel
        per_device_eval_batch_size: Batch size per device
        dataloader_num_workers: Number of workers for dataloader
        sample: Number of samples to use (for testing)
    
    Returns:
        predictions: Model predictions (0 for No, 1 for Yes)
        dataset: The dataset used
        process_index: Process index for distributed training
    """
    options = ['No', 'Yes']  # 0 for No (discontinuous), 1 for Yes (continuous)
    strict_option_ids = [processor.tokenizer(f'{yes_no_previous_str}{_}').input_ids[-1] for _ in options] 
    
    dataset = VideoContinuityDataset(
        video_paths=video_paths,
        remote_loader=remote_loader,
        crop_ratio=crop_ratio,
        question_prefix=question_prefix,
        question_postfix=question_postfix,
        answer_prefix=answer_prefix,
        sample=sample,
        wo_stamp=wo_stamp,
    )
    
    trainer = Trainer(
        model=model, 
        args=TrainingArguments(
            output_dir='outputs/', 
            do_predict=True, 
            per_device_eval_batch_size=per_device_eval_batch_size, 
            dataloader_num_workers=dataloader_num_workers, 
            report_to='none', 
            use_liger_kernel=use_liger_kernel
        ), 
        data_collator=functools.partial(dataset.data_collator, processor=processor),
        processing_class=processor,
        preprocess_logits_for_metrics=functools.partial(preprocess_logits_for_metrics, strict_option_ids=strict_option_ids),
    )
    
    letter_idxs_predictions = trainer.predict(dataset, ignore_keys=['past_key_values', 'hidden_states', 'attentions', 'rope_deltas']).predictions
    return letter_idxs_predictions, dataset, trainer.args.process_index



def evaluate_video_continuity_results(predictions, dataset):
    """
    Evaluate video continuity detection results.
    
    Args:
        predictions: Model predictions (0 for No, 1 for Yes)
        dataset: The dataset used for evaluation
    
    Returns:
        dict: Evaluation metrics
    """
    correct = 0
    total = 0
    continuous_correct = 0
    continuous_total = 0
    discontinuous_correct = 0
    discontinuous_total = 0
    
    for i, (prediction, datum) in enumerate(zip(predictions, dataset.datums)):
        total += 1
        predicted_continuous = prediction == 1  # 1 for Yes (continuous)
        actual_continuous = datum['is_continuous']
        datum['predicted_continuous'] = str(predicted_continuous)
        
        if predicted_continuous == actual_continuous:
            correct += 1
            
        if actual_continuous:
            continuous_total += 1
            if predicted_continuous:
                continuous_correct += 1
        else:
            discontinuous_total += 1
            if not predicted_continuous:
                discontinuous_correct += 1
    
    metrics = {
        'overall_accuracy': correct / total if total > 0 else 0,
        'continuous_accuracy': continuous_correct / continuous_total if continuous_total > 0 else 0,
        'discontinuous_accuracy': discontinuous_correct / discontinuous_total if discontinuous_total > 0 else 0,
        'total_samples': total,
        'continuous_samples': continuous_total,
        'discontinuous_samples': discontinuous_total
    }
    
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"Continuous Videos Accuracy: {metrics['continuous_accuracy']:.4f} ({continuous_correct}/{continuous_total})")
    print(f"Discontinuous Videos Accuracy: {metrics['discontinuous_accuracy']:.4f} ({discontinuous_correct}/{discontinuous_total})")
    
    return metrics, dataset.datums

if __name__ == '__main__':
    model_path = "chenjoya/LiveCC-7B-Instruct"
    model_path = "/2022233235/videollm-online/livecc/outputs/livecc_sft_24k480x100_llava178kSampleStream_split_random_lora_lr1e-5/checkpoint-1494"
    wo_stamp = False
    if "Qwen2.5" in model_path:
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", attn_implementation='flash_attention_2')
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", attn_implementation='flash_attention_2')
    processor = AutoProcessor.from_pretrained(model_path, padding_side='left')
    
    # Example usage for video continuity detection
    src_video_dir = os.path.dirname("/2022233235/.cache/huggingface/hub/datasets--JoeLeelyf--OVO-Bench/snapshots/fec29e3385747b5642d995370143ba92d2819bd2/src_videos/")
    video_paths = json.load(open('/2022233235/videollm-online/livecc/evaluation/video_augmentation/sampled_video_paths.json'))
    video_paths = [os.path.join(src_video_dir, p) for p in video_paths]
    
    
    predictions, dataset, process_index = video_continuity_predict(
        model=model, 
        processor=processor, 
        video_paths=video_paths,
        crop_ratio=0.3,
        answer_prefix='The answer is:\n',
        yes_no_previous_str='\n',
        remote_loader=None,
        # sample=100  # For testing with 10 samplesa
        wo_stamp=wo_stamp,
    )
    
    if process_index == 0:
        metrics, datums = evaluate_video_continuity_results(predictions, dataset)
        save_json_path = f'evaluation/video_augmentation/results/{os.path.basename(model_path)}_continuity_results_stamp{wo_stamp}.json'
        os.makedirs(os.path.dirname(save_json_path), exist_ok=True)
        json.dump(datums, open(save_json_path, 'w'))
        save_txt_path = save_json_path.replace('.json', '.txt')
        print_json = lambda x: print(json.dumps(x, indent=4))
        save_function_print(print_json, save_txt_path, metrics)

# torchrun --standalone --nproc_per_node=8 evaluation/video_augmentation/distributed_evaluate_split.py