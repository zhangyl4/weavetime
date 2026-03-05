from dataclasses import asdict
import transformers
from transformers import Trainer, AutoProcessor, HfArgumentParser, TrainingArguments, AutoConfig, logging

from models import ModelLoRAArguments
from data.lmm_dataset import DataArguments, LMMDataset
from peft import LoraConfig, get_peft_model, PeftModel

logger = logging.get_logger(__name__)


from torch.utils.data import Sampler, Dataset
from typing import Any, Dict, List, Optional, Union
import torch, math, random
import os

class StrideGroupedSampler(Sampler):
    """Group """

    @staticmethod
    def _compute_turns(dataset: Dataset) -> List[int]:
        return dataset.video_info
    
    @staticmethod
    def _compute_frame_numbers(dataset: Dataset) -> List[int]:
        frame_numbers = []
        for info in dataset.video_info:
            frame_numbers.append(1000)
        return frame_numbers
    
    
    def __init__(
        self,
        batch_size: int,
        group: str,
        sort: Optional[str] = None,
        dataset: Optional[Dataset] = None,
        lengths: Optional[List[int]] = None,
    ):
        print("init len",len(dataset))
        
        # 1. get lengths
        if dataset is None:
            raise ValueError("One of dataset and lengths must be provided.")
        
        if group is None:
            raise ValueError("Group cannot be None!")

        if lengths is None:
            lengths = StrideGroupedSampler._compute_turns(dataset)
            frame_numbers = StrideGroupedSampler._compute_frame_numbers(dataset)
        elif isinstance(lengths, torch.Tensor):
            logger.info(
                "If lengths is a torch.Tensor, LengthGroupedSampler will be slow. Converting lengths to List[int]..."
            )
            lengths = lengths.tolist()

        # 2. compute index and stride pairs
        # 2.1 compute indices
        indices = list(range(len(lengths)))

        # 2.2 get number of strides for each data
        num_strides = []
        for length in lengths:
            num_stride = math.ceil(length)
            num_strides.append(num_stride)

        print("midd len",len(indices))
        print("midd max",max(indices))
        
        indice_stride_pairs = list(zip(indices, num_strides, frame_numbers))
        # NOTE: shuffle the indices in advance, otherwise the randomness may be lost when all num_strides are equal
        random.shuffle(indice_stride_pairs)

        # 2.3 sort data according to the number of strides
        indice_stride_pairs = sorted(indice_stride_pairs, key=lambda x: (x[1], x[2]))
        # indice_stride_pairs = sorted(indice_stride_pairs, key=lambda x: x[1])

        # 3. group data instances with the same number of strides into the same batch
        batches = []
        batch = []
        prev_num_stride = None
        for index, num_stride, frame_number in indice_stride_pairs:
            if len(batch) > 0  and num_stride != prev_num_stride:
                # in strict mode, all instances in the batch are forced to have the same number of strides
                if group == "strict":
                    # If stride doesn't match, randomly sample from current batch to fill it use previous batch's stride
                    lack_num = batch_size - len(batch)
                    try:
                        sampled_index = random.choices(batch, k=lack_num)
                    except:
                        print(batch)
                        breakpoint()
                    batch.extend(sampled_index)
                    
                    batches.append((batch.copy(), prev_num_stride)) 
                    batch.clear()
                    
                    # Create new index with updated stride count
                    batch.append(index)
                    prev_num_stride = num_stride
                    
                    continue
                elif group == "relaxed":
                    pass
                else:
                    raise ValueError(f"Group method {group} must be in None, strict, relaxed!")

            batch.append(index)
            prev_num_stride = num_stride

            if len(batch) == batch_size:
                # random.shuffle(batch)
                batches.append((batch.copy(), num_stride))
                batch.clear()

        if len(batch) and group == "relaxed":
            # random.shuffle(batch)
            batches.append((batch.copy(), num_stride))
        
        if sort is None:
            random.shuffle(batches)
        elif sort == "ascend":
            batches = sorted(batches, key=lambda x: x[1])
        elif sort == "descend":
            batches = sorted(batches, key=lambda x: x[1], reverse=True)
        else:
            raise ValueError(f"Sort method {sort} must be in None, ascend, descend!")

        batches = [x[0] for x in batches]
        
         # Remove problematic batch at step 82 (batch_idx = 81 since 0-based)
        def delete_batch(batches, step):
            batch_size = 1
            grad_accum = 8
            samples_per_step = batch_size * grad_accum
            pop_idxs = range(step * samples_per_step, (step+1) * samples_per_step)
            for pop_idx in pop_idxs:
                if pop_idx < len(batches):
                    batches.pop(pop_idx)
        delete_batch(batches, 344)
        delete_batch(batches, 451)
        delete_batch(batches, 625)
        delete_batch(batches, 804)
        delete_batch(batches, 868)
        delete_batch(batches, 869)
        delete_batch(batches, 871)
        delete_batch(batches, 872)
        delete_batch(batches, 873)
        
        self.indices = sum(batches, [])
        print("final len",len(self.indices))
        print("final max",max(self.indices))

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        return iter(self.indices)


logger = logging.get_logger(__name__)

# 假设 StrideGroupedSampler 已经从其他地方导入
# from .sampler import StrideGroupedSampler 

class TrainerWithPerformanceAnalysis(Trainer):

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        """
        重写父类的方法以使用自定义的 Sampler。
        """
        # Build the sampler.
        print(f"Using StrideGroupedSampler with total batch size: {self.args.train_batch_size * self.args.world_size}")
        if True: # 保持你原来的逻辑
             return StrideGroupedSampler(
                 # NOTE: multiply world size to get the total number of training instances across devices
                 batch_size=self.args.train_batch_size * self.args.world_size,
                 dataset=self.train_dataset,
                 group="relaxed",
             )
        else:
             return super()._get_train_sampler()


if __name__ == "__main__":
    training_args, model_args, data_args = HfArgumentParser((TrainingArguments, ModelLoRAArguments, DataArguments)).parse_args_into_dataclasses()
    config = AutoConfig.from_pretrained(model_args.pretrained_model_name_or_path)
    model = getattr(transformers, config.architectures[0]).from_pretrained(
        model_args.pretrained_model_name_or_path, 
        torch_dtype="auto", attn_implementation='flash_attention_2'
    )
    # # HACK: add lora modules
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
        model = PeftModel.from_pretrained(model, model_args.adapter_model, is_trainable=True, config=lora_config) # model = get_peft_model(model, config.adapter_model, lora_config)
    else:
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    for m in model_args.freeze_modules:
        logger.warning(f"Freezing module {m}")
        getattr(model, m).requires_grad_(False)
    
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    if 'Qwen2VL' in model.config.architectures[0]:
        processor = AutoProcessor.from_pretrained('Qwen/Qwen2-VL-7B-Instruct', padding_side='right') # Qwen2vl-base processor has some bugs. otherwise we do not need this
    else:
        processor = AutoProcessor.from_pretrained(model_args.pretrained_model_name_or_path, padding_side='right')
    train_dataset = LMMDataset(**asdict(data_args), **asdict(training_args), **asdict(model_args), processor=processor)
    if training_args.resume_from_checkpoint:
        resume_from_checkpoint = training_args.resume_from_checkpoint
    else:
        resume_from_checkpoint = not training_args.overwrite_output_dir
    TrainerWithPerformanceAnalysis(
        model=model, args=training_args, 
        data_collator=train_dataset.data_collator, 
        train_dataset=train_dataset, processing_class=processor
    ).train(resume_from_checkpoint=resume_from_checkpoint)
    
    