import transformers
from transformers import Trainer, AutoProcessor, HfArgumentParser, TrainingArguments, AutoConfig, logging

from dataclasses import dataclass, field
from data.qwen_vl_dataset import (
    DataArguments as QwenDataArguments,
    QwenVLDataset,
)
from peft import LoraConfig, get_peft_model, PeftModel

from torch.utils.data import Sampler, Dataset
from typing import Any, Dict, List, Optional, Union
import torch

logger = logging.get_logger(__name__)


class TrainerWithPerformanceAnalysis(Trainer):
    pass
    # def _get_train_sampler(self, dataset) -> Optional[torch.utils.data.Sampler]:
    #     """
    #     Override parent method to use custom Sampler when needed.
    #     (Currently falls back to default sampler.)
    #     """
    #     return super()._get_train_sampler(dataset)


@dataclass
class ModelLoRAArguments:
    pretrained_model_name_or_path: str = ''
    freeze_modules: list[str] = field(default_factory=lambda: [])
    lora_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ]
    )
    lora_r: int = 128
    lora_alpha: int = 256
    finetune_modules: list[str] = field(default_factory=lambda: ['connector'])
    adapter_model: Optional[str] = None


if __name__ == "__main__":
    training_args, model_args, data_args = HfArgumentParser((TrainingArguments, ModelLoRAArguments, QwenDataArguments)).parse_args_into_dataclasses()

    # Load base model (generic path like train_lora.py to work across Qwen variants)
    config = AutoConfig.from_pretrained(model_args.pretrained_model_name_or_path)
    model_cls_name = config.architectures[0]
    model = getattr(transformers, model_cls_name).from_pretrained(
        model_args.pretrained_model_name_or_path,
        torch_dtype="auto",
        attn_implementation='flash_attention_2'
    )

    # Configure LoRA
    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        target_modules=model_args.lora_modules,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        modules_to_save=model_args.finetune_modules,
        inference_mode=False,
    )
    if model_args.adapter_model:
        model = PeftModel.from_pretrained(model, model_args.adapter_model, is_trainable=True, config=lora_config)
    else:
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Freeze specified modules
    # for m in model_args.freeze_modules:
    #     logger.warning(f"Freezing module {m}")
    #     getattr(model, m).requires_grad_(False)

    # Enable gradient checkpointing and input gradients
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # Load processor (special-case Qwen2VL naming like train_lora.py)
    if 'Qwen2VL' in model.config.architectures[0]:
        processor = AutoProcessor.from_pretrained('Qwen/Qwen2-VL-7B-Instruct', padding_side='right')
    else:
        processor = AutoProcessor.from_pretrained(model_args.pretrained_model_name_or_path, padding_side='right')

    # Wire tokenizer/processor into dataset args
    data_args.processor = processor

    # Build dataset
    train_dataset = QwenVLDataset(
        annotation_paths=data_args.annotation_paths,
        tokenizer=processor.tokenizer,
        data_args=data_args,
    )

    # Resume training if specified
    if training_args.resume_from_checkpoint:
        resume_from_checkpoint = training_args.resume_from_checkpoint
    else:
        resume_from_checkpoint = not training_args.overwrite_output_dir

    # Initialize trainer and start training
    TrainerWithPerformanceAnalysis(
        model=model,
        args=training_args,
        data_collator=train_dataset.data_collator,
        train_dataset=train_dataset,
        processing_class=processor
    ).train(resume_from_checkpoint=resume_from_checkpoint)



