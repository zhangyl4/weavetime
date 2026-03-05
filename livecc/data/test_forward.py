import torch
import json
import tqdm
import os
import argparse
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, HfArgumentParser, TrainingArguments, AutoConfig, logging
from transformers import LlavaForConditionalGeneration
from peft import LoraConfig, get_peft_model, PeftModel

from dataclasses import dataclass, field
from typing import Optional

# Import from the project
import sys
sys.path.append('/2024233235/videollm-online/livecc')
from data.llava_ov_dataset import LLaVAOVDataset, DataArguments

logger = logging.get_logger(__name__)

@dataclass
class ModelLoRAArguments:
    pretrained_model_name_or_path: str = ''
    freeze_modules: list[str] = field(default_factory=lambda: [])
    lora_modules: str = "language_model.model.layers.*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|language_model.lm_head$"
    lora_r: int = 128
    lora_alpha: int = 256
    finetune_modules: list[str] = field(default_factory=lambda: ['connector'])
    adapter_model: Optional[str] = None

def BatchForCausalLMLoss(
    logits, labels, vocab_size: int, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs
):
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()
    labels = labels.to(logits.device)
    # Shift so that tokens < n predict n
    labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    logits = logits.view(-1, vocab_size)
    loss = nn.functional.cross_entropy(logits, shift_labels.view(-1).to(logits.device), ignore_index=ignore_index, reduction='none')
    loss = loss.view_as(shift_labels).sum(dim=-1) / (shift_labels > 0).sum(dim=-1)
    return loss

def _format_suffix(degree: float, clip_number: Optional[int]) -> str:
    try:
        deg_str = f"{float(degree):g}".replace(".", "p")
    except Exception:
        deg_str = "unknown"
    if clip_number is None:
        clip_str = "none"
    else:
        try:
            clip_str = str(int(clip_number))
        except Exception:
            clip_str = "unknown"
    return f"_deg{deg_str}_clip{clip_str}"


def test_forward_debug(
    data_path: str,
    model_path: str,
    lora_path: str = None,
    debug: bool = False,
    max_samples: int = 5,
    segment_reconstruction_degree: float = 1.0,
    segment_reconstruction_clip_number: Optional[int] = None,
    save_suffix: str = "",
):
    """
    Single GPU forward pass with debug mode
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load LLaVA OV model (following train_llava_ov_lora.py pattern)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2'
    )
    
    # Configure LoRA if specified
    if lora_path:
        print(f"Loading LoRA weights from: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
    # else:
    #     # Configure LoRA (following train_llava_ov_lora.py pattern)
    #     lora_config = LoraConfig(
    #         r=128,
    #         lora_alpha=256,
    #         target_modules="language_model.model.layers.*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|language_model.lm_head$",
    #         lora_dropout=0.05,
    #         task_type="CAUSAL_LM",
    #         modules_to_save=['connector'],
    #         inference_mode=False,
    #     )
    #     model = get_peft_model(model, lora_config)
    
    model.eval()
    model.to(device)
    model.loss_function = BatchForCausalLMLoss
    
    # Load processor (following train_llava_ov_lora.py pattern)
    processor = AutoProcessor.from_pretrained(
        model_path,
        padding_side='right'
    )
    
    # Setup data arguments (following train_llava_ov_lora.py pattern)
    data_args = DataArguments(
        annotation_paths=[data_path],
        lazy_preprocess=False,
        is_multimodal=True,
        image_aspect_ratio="square",
        video_fps=1,
        frames_upbound=32,
        enable_time_prompt=True,
        frames_per_input=1,
        enable_segment_reconstruction=True,
        segment_reconstruction_paths=[
            "/2024233235/videollm-online/EyeWO2/data/llava_video_178k_with_seeks_sample_valid.jsonl",
        ],
        
    )
    # Set segment reconstruction controls
    data_args.segment_reconstruction_degree = float(segment_reconstruction_degree)
    data_args.segment_reconstruction_clip_number = (
        int(segment_reconstruction_clip_number)
        if segment_reconstruction_clip_number is not None
        else None
    )
    
    # Wire processor into dataset args
    data_args.processor = processor
    
    # Create dataset (following train_llava_ov_lora.py pattern)
    dataset = LLaVAOVDataset(
        annotation_paths=data_args.annotation_paths,
        tokenizer=processor.tokenizer,
        data_args=data_args,
    )
    
    if debug:
        print(f"Dataset size: {len(dataset)}")
        print(f"Max samples to process: {max_samples}")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=True, 
        collate_fn=dataset.data_collator,
        num_workers=1
    )
    
    outputs = []
    for i, batch in enumerate(tqdm.tqdm(dataloader, desc="Processing samples")):
        if debug:
            print(f"\n=== Processing sample {i+1} ===")
            print(f"Batch keys: {batch.keys()}")
            if 'input_ids' in batch:
                print(f"Input shape: {batch['input_ids'].shape}")
            if 'labels' in batch:
                print(f"Labels shape: {batch['labels'].shape}")
                
        if i >= max_samples:
            break
        
        try:
            # Extract original_idx before moving to device
            original_idx = batch.get('original_idx', torch.tensor(i))
            if isinstance(original_idx, torch.Tensor):
                original_idx = original_idx.item()
            
            # Move batch to device (skip original_idx as it's metadata)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) and k != 'original_idx' else v for k, v in batch.items()}
            # Forward pass
            with torch.inference_mode():
                outputs_model = model(**batch)
                
                loss = outputs_model.loss
                logits = outputs_model.logits
                
                # Get predictions
                pred_ids = torch.argmax(logits, dim=-1)
                
                # Decode labels and predictions
                labels = batch['labels']
                label_tokens = labels[labels != -100]
                pred_tokens = pred_ids[labels != -100]
                
                label_text = processor.tokenizer.decode(label_tokens, skip_special_tokens=True)
                pred_text = processor.tokenizer.decode(pred_tokens, skip_special_tokens=True)
                
                # Store results
                result = {
                    'sample_id': i,
                    'original_idx': int(original_idx),
                    'loss': loss.item(),
                    'label': label_text,
                    'prediction': pred_text,
                }
                outputs.append(result)
                
                # Print comparison
                print(f"\n=== Sample {i+1} (original_idx: {result['original_idx']}) ===")
                print(f"Loss: {result['loss']:.4f}")
                print(f"Label: {result['label']}")
                print(f"Prediction: {result['prediction']}")
                print("=" * 50)
                
                if debug:
                    print(f"Logits shape: {logits.shape}")
                    print(f"Prediction tokens: {pred_tokens[:10]}...")  # Show first 10 tokens
                
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            if debug:
                import traceback
                traceback.print_exc()
            exit()
    
    # Save outputs
    output_dir = os.path.splitext(data_path)[0] + '_test_forward_results'
    os.makedirs(output_dir, exist_ok=True)
    auto_suffix = _format_suffix(segment_reconstruction_degree, segment_reconstruction_clip_number)
    extra_suffix = save_suffix or ""
    if extra_suffix and not extra_suffix.startswith("_"):
        extra_suffix = "_" + extra_suffix
    suffix = f"{auto_suffix}{extra_suffix}"
    output_file = (
        f'{output_dir}/results_debug{suffix}.json'
        if debug
        else f'{output_dir}/results{suffix}.json'
    )
    json.dump(outputs, open(output_file, 'w'), indent=2)
    
    print(f"\nResults saved to: {output_file}")
    print(f"Processed {len(outputs)} samples successfully")
    
    return outputs


# python /2024233235/videollm-online/livecc/data/test_forward.py --data_path /2024233235/videollm-online/EyeWO2/data/llava_video_178k_with_seeks_sample_valid.jsonl --model_path llava-hf/llava-onevision-qwen2-7b-ov-hf --lora_path /2024233235/videollm-online/livecc/outputs/llavaov_sft_timestamp_shuffle_et_llava_lr5e-5/checkpoint-249/ --debug
# python /2024233235/ReKV/tools/analyze_results_debug.py /2024233235/videollm-online/EyeWO2/data/llava_video_178k_with_seeks_sample_valid_test_forward_results/results_debug_deg0p5_clipnone.json
# python /2024233235/ReKV/tools/plot_overlap_distribution.py  --input 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test forward pass for LLaVA OV model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to data JSONL file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model directory')
    parser.add_argument('--lora_path', type=str, default=None, help='Path to LoRA weights (optional)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--max_samples', type=int, default=500, help='Maximum samples to process in debug mode')
    parser.add_argument(
        '--segment_reconstruction_degrees',
        type=str,
        default='1.0',
        help='Comma-separated list of segment_reconstruction_degree values, e.g. "1.0,0.5,0.25"',
    )
    parser.add_argument(
        '--segment_reconstruction_clip_numbers',
        type=str,
        default='4',
        help='Comma-separated list of clip numbers, e.g. "none,2,4". Use "none" to disable clip grouping.',
    )
    parser.add_argument(
        '--save_suffix',
        type=str,
        default='',
        help='Optional extra suffix appended after the auto suffix, e.g. "runA" -> "..._deg1_clip4_runA.json".',
    )
    
    args = parser.parse_args()

    # Parse lists
    degrees: list[float] = []
    for part in (args.segment_reconstruction_degrees or '').split(','):
        part = part.strip()
        if not part:
            continue
        degrees.append(float(part))
    if not degrees:
        degrees = [1.0]

    clip_numbers: list[Optional[int]] = []
    for part in (args.segment_reconstruction_clip_numbers or '').split(','):
        part = part.strip().lower()
        if not part:
            continue
        if part in ('none', 'null', 'nil', 'off', 'disable', 'disabled'):
            clip_numbers.append(None)
        else:
            clip_numbers.append(int(part))
    if not clip_numbers:
        clip_numbers = [None]

    # Run forward for each (degree, clip_number) combination
    for deg in degrees:
        for clip in clip_numbers:
            print(f"\n=== Running: segment_reconstruction_degree={deg}, segment_reconstruction_clip_number={clip} ===")
            outputs = test_forward_debug(
                data_path=args.data_path,
                model_path=args.model_path,
                lora_path=args.lora_path,
                debug=args.debug,
                max_samples=args.max_samples,
                segment_reconstruction_degree=deg,
                segment_reconstruction_clip_number=clip,
                save_suffix=args.save_suffix,
            )

    print("\nForward pass completed!")
