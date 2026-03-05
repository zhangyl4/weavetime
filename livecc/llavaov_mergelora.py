import torch
import argparse
import os
from transformers import LlavaForConditionalGeneration, AutoProcessor
from peft import PeftModel


def merge_lora_weights(
    base_model_path: str,
    adapter_model_path: str,
    output_dir: str = None,
):
    """
    合并 LoRA 权重到基础模型
    
    Args:
        base_model_path: 基础模型路径
        adapter_model_path: LoRA adapter 路径
        output_dir: 输出目录，默认为 outputs/{adapter_name}_mergelora
    """
    # 设置默认输出目录
    if output_dir is None:
        adapter_name = os.path.basename(adapter_model_path.rstrip('/'))
        output_dir = f"outputs/{adapter_name}_mergelora"
    
    print(f"Loading base model from: {base_model_path}")
    # 加载基础模型
    base_model = LlavaForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2'
    )
    
    print(f"Loading LoRA adapter from: {adapter_model_path}")
    # 加载 LoRA adapter
    model = PeftModel.from_pretrained(
        base_model,
        adapter_model_path,
        is_trainable=False
    )
    
    print("Merging LoRA weights into base model...")
    # 合并 LoRA 权重
    merged_model = model.merge_and_unload()
    
    print(f"Saving merged model to: {output_dir}")
    # 保存合并后的模型
    os.makedirs(output_dir, exist_ok=True)
    merged_model.save_pretrained(output_dir)
    
    # 同时保存 processor
    print(f"Saving processor to: {output_dir}")
    processor = AutoProcessor.from_pretrained(base_model_path)
    processor.save_pretrained(output_dir)
    
    print("Merge completed successfully!")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA weights into base LLaVA-OV model")
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="llava-hf/llava-onevision-qwen2-7b-ov-hf",
        help="Path to the base LLaVA-OV model"
    )
    parser.add_argument(
        "--adapter_model_path",
        type=str,
        required=True,
        help="Path to the LoRA adapter checkpoint"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for merged model (default: outputs/{adapter_name}_mergelora)"
    )
    
    args = parser.parse_args()
    
    merge_lora_weights(
        base_model_path=args.base_model_path,
        adapter_model_path=args.adapter_model_path,
        output_dir=args.output_dir
    )

