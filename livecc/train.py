from dataclasses import asdict
import transformers
from transformers import Trainer, AutoProcessor, HfArgumentParser, TrainingArguments, AutoConfig, logging

from models import ModelArguments
from data.lmm_dataset import DataArguments, LMMDataset

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
        
        self.indices = sum(batches, [])

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        return iter(self.indices)

import torch
import time
from torch import nn
from typing import Optional, Dict, Any, Union
from transformers import Trainer
from transformers.utils import (
    is_sagemaker_mp_enabled, 
    is_torch_tpu_available,
    is_torch_xpu_available,
    is_torch_mlu_available,
    is_torch_musa_available,
    is_torch_npu_available,
    is_torch_mps_available,
    is_torch_hpu_available,
    is_accelerate_available,
    logging,
)
# from transformers.trainer_pt_utils import smp_forward_backward
from transformers.utils.import_utils import is_apex_available
from transformers.training_args import OptimizerNames

if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.state import AcceleratorState
    from accelerate.utils import (
        AutocastKwargs,
        DistributedDataParallelKwargs,
        DistributedType,
        load_fsdp_model,
        load_fsdp_optimizer,
        save_fsdp_model,
        save_fsdp_optimizer,
    )

if is_apex_available():
    from apex import amp

logger = logging.get_logger(__name__)

# 假设 StrideGroupedSampler 已经从其他地方导入
# from .sampler import StrideGroupedSampler 

class TrainerWithPerformanceAnalysis(Trainer):
    """
    这个 Trainer 继承自原始的 Trainer，并重写了 training_step 方法，
    以便在 rank 0 进程上记录关键操作的时间戳，用于性能分析。
    """
    
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     # 初始化一个字典来存储各个阶段的累计时间
    #     self.timers = {
    #         'data_loading': 0.0,
    #         'data_to_gpu': 0.0,
    #         'forward_pass': 0.0,
    #         'backward_pass': 0.0,
    #         'total_step_time': 0.0
    #     }
    #     self.step_count = 0

    # def get_batch_samples(self, epoch_iterator, num_batches, device):
    #     """
    #     Fetches a number of batches from the data iterator and performs timing.
    #     This is a custom function to be called from a custom training loop.
    #     """
    #     is_rank_0 = self.args.local_rank == -1 or self.args.process_index == 0
    #     if is_rank_0:
    #         # This function is mostly CPU-bound, but synchronizing is good practice.
    #         torch.cuda.synchronize()
    #         start_time = time.time()

    #     batch_samples = []
    #     num_items_in_batch = None
    #     for _ in range(num_batches):
    #         try:
    #             batch_samples += [next(epoch_iterator)]
    #         except StopIteration:
    #             break

    #     if len(batch_samples) > 0 and "labels" in batch_samples[0]:
    #         # For now we don't support object detection
    #         try:
    #             num_items_in_batch = sum([(batch["labels"].ne(-100)).sum() for batch in batch_samples])
    #         except (TypeError, AttributeError):
    #             pass

    #     if num_items_in_batch is not None:
    #         if self.args.average_tokens_across_devices:
    #             num_items_in_batch = self.accelerator.gather(num_items_in_batch).sum()

    #         if torch.is_tensor(num_items_in_batch):
    #             num_items_in_batch = num_items_in_batch.to(device)
        
    #     if is_rank_0:
    #         # Synchronize again in case of device operations at the end (like .to(device))
    #         torch.cuda.synchronize()
    #         end_time = time.time()
    #         # Add the total time for this operation to the cumulative timer.
    #         # The averaging in training_step will correctly prorate this over micro-steps.
    #         self.timers['data_loading'] += (end_time - start_time)

    #     return batch_samples, num_items_in_batch

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

    # def training_step(
    #     self, model: nn.Module, inputs: dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    # ) -> torch.Tensor:
    #     """
    #     重写核心的 training_step 方法来添加计时逻辑，并与新接口对齐。
    #     只有在 rank 0 上才会打印日志。
    #     """
    #     # 只有在 rank 0 进程上才执行计时和打印
    #     is_rank_0 = self.args.local_rank == -1 or self.args.process_index == 0
        
    #     if is_rank_0:
    #         torch.cuda.synchronize()
    #         step_start_time = time.time()

    #     model.train()
    #     if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
    #         self.optimizer.train()

    #     # 1. 数据移动到 GPU
    #     inputs = self._prepare_inputs(inputs)
        
    #     if is_rank_0:
    #         torch.cuda.synchronize()
    #         time_after_data_move = time.time()
    #         self.timers['data_to_gpu'] += (time_after_data_move - step_start_time)

    #     # SageMaker Model Parallelism special handling
    #     if is_sagemaker_mp_enabled():
    #         loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
    #         # Timers for SageMaker are harder to inject without more complex logic.
    #         # We return early as per the original implementation.
    #         return loss_mb.reduce_mean().detach().to(self.args.device)

    #     # 2. 模型前向传播 (Forward Pass)
    #     with self.compute_loss_context_manager():
    #         loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

    #     if is_rank_0:
    #         torch.cuda.synchronize()
    #         time_after_forward = time.time()
    #         self.timers['forward_pass'] += (time_after_forward - time_after_data_move)
        
    #     # Free up memory
    #     del inputs
    #     if (
    #         self.args.torch_empty_cache_steps is not None
    #         and self.state.global_step % self.args.torch_empty_cache_steps == 0
    #     ):
    #         if is_torch_xpu_available():
    #             torch.xpu.empty_cache()
    #         elif is_torch_mlu_available():
    #             torch.mlu.empty_cache()
    #         elif is_torch_musa_available():
    #             torch.musa.empty_cache()
    #         elif is_torch_npu_available():
    #             torch.npu.empty_cache()
    #         elif is_torch_mps_available(min_version="2.0"):
    #             torch.mps.empty_cache()
    #         elif is_torch_hpu_available():
    #             logger.warning(
    #                 "`torch_empty_cache_steps` is set but HPU device/backend does not support empty_cache()."
    #             )
    #         else:
    #             torch.cuda.empty_cache()

    #     # 3. 反向传播 (Backward Pass)
    #     kwargs = {}
    #     if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
    #         kwargs["learning_rate"] = self._get_learning_rate()

    #     if self.args.n_gpu > 1:
    #         loss = loss.mean()

    #     if self.use_apex:
    #         with amp.scale_loss(loss, self.optimizer) as scaled_loss:
    #             scaled_loss.backward()
    #     else:
    #         if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
    #             loss_to_backward = loss / self.args.gradient_accumulation_steps
    #         else:
    #             loss_to_backward = loss

    #         if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
    #             kwargs["scale_wrt_gas"] = False
            
    #         self.accelerator.backward(loss_to_backward, **kwargs)

    #     if is_rank_0:
    #         torch.cuda.synchronize()
    #         time_after_backward = time.time()
    #         self.timers['backward_pass'] += (time_after_backward - time_after_forward)
            
    #         # 记录总时间并更新计数器
    #         total_time_for_step = time_after_backward - step_start_time
    #         self.timers['total_step_time'] += total_time_for_step
    #         self.step_count += 1

    #         # 每隔N步打印一次平均耗时，避免日志刷屏
    #         if True:
    #             avg_data_loading = self.timers['data_loading'] / self.step_count
    #             avg_data_to_gpu = self.timers['data_to_gpu'] / self.step_count
    #             avg_forward = self.timers['forward_pass'] / self.step_count
    #             avg_backward = self.timers['backward_pass'] / self.step_count
    #             avg_total = self.timers['total_step_time'] / self.step_count
                
    #             print("\n--- Performance Analysis (Average Time per Micro-Step) ---")
    #             print(f" rank: {self.args.local_rank}, Global Step:      {self.state.global_step}, local step {self.step_count}")
    #             print(f" rank: {self.args.local_rank}  Data Loading:     {avg_data_loading:.4f} s")
    #             print(f" rank: {self.args.local_rank}  Data to GPU:      {avg_data_to_gpu:.4f} s")
    #             print(f" rank: {self.args.local_rank}  Forward Pass:     {avg_forward:.4f} s")
    #             print(f" rank: {self.args.local_rank}  Backward Pass:    {avg_backward:.4f} s")
    #             print(f" rank: {self.args.local_rank}  Total Step Time:  {avg_total:.4f} s")
    #             print("----------------------------------------------------------\n")

        return loss.detach()


if __name__ == "__main__":
    training_args, model_args, data_args = HfArgumentParser((TrainingArguments, ModelArguments, DataArguments)).parse_args_into_dataclasses()
    config = AutoConfig.from_pretrained(model_args.pretrained_model_name_or_path)
    model = getattr(transformers, config.architectures[0]).from_pretrained(
        model_args.pretrained_model_name_or_path, 
        torch_dtype="auto", attn_implementation='flash_attention_2'
    )
    for m in model_args.freeze_modules:
        logger.warning(f"Freezing module {m}")
        getattr(model, m).requires_grad_(False)
    if 'Qwen2VL' in model.config.architectures[0]:
        processor = AutoProcessor.from_pretrained('Qwen/Qwen2-VL-7B-Instruct', padding_side='right') # Qwen2vl-base processor has some bugs. otherwise we do not need this
    else:
        processor = AutoProcessor.from_pretrained(model_args.pretrained_model_name_or_path, padding_side='right')
    train_dataset = LMMDataset(**asdict(data_args), **asdict(training_args), **asdict(model_args), processor=processor)

    trainer = TrainerWithPerformanceAnalysis(
        model=model,
        args=training_args,
        data_collator=train_dataset.data_collator,
        train_dataset=train_dataset,
        processing_class=processor
    )
    trainer.train(resume_from_checkpoint=not training_args.overwrite_output_dir)
