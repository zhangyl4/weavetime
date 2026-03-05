#!/usr/bin/env python3
import torch
from logzero import logger
from typing import Optional, List, Dict, Any
import os
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from model.patch import patch_hf
from model.abstract_rekv import Abstract_ReKV

from torchvision import transforms
os.environ['VIDEO_MAX_PIXELS'] = str(int(24576 / 4 * 28 * 28)) # increase this for streaming. 24576 * 28 * 28 = 19267584
import qwen_vl_utils.vision_process
qwen_vl_utils.vision_process.VIDEO_MIN_PIXELS = int(os.environ.get('VIDEO_MIN_PIXELS', 100 * 28 * 28)) # follow qwen2vl paper
qwen_vl_utils.vision_process.FPS_MAX_FRAMES = int(os.environ.get('FPS_MAX_FRAMES', 120)) 
from qwen_vl_utils.vision_process import (
    FORCE_QWENVL_VIDEO_READER, VIDEO_TOTAL_PIXELS, FPS_MAX_FRAMES, VIDEO_MIN_PIXELS, VIDEO_MAX_PIXELS, FRAME_FACTOR, IMAGE_FACTOR, FPS,
    smart_nframes, smart_resize
)

from model.qwen2vl.patch_model import convert_qwen2vl_to_streaming


class Qwen2VL_ReKV(Qwen2VLForConditionalGeneration, Abstract_ReKV):
    def __init__(self, config, n_frame_tokens, init_prompt_ids=None, n_local=None, topk=64, chunk_size=1, processor=None):
        Qwen2VLForConditionalGeneration.__init__(self, config)
        Abstract_ReKV.__init__(self, processor, n_frame_tokens, init_prompt_ids, n_local, topk, chunk_size)
        
        # 添加固定分辨率支持
        self._fixed_video_resolution = None  # (height, width) 由第一个chunk确定
        self._current_video_id = None        # 用于跟踪当前处理的视频

    @property
    def device(self):
        try:
            return self.model.device
        except Exception:
            return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def reset_video_session(self, video_id=None):
        """
        重置视频会话，开始处理新视频时调用
        """
        # Only reset resolution if this is a different video
        if video_id != self._current_video_id:
            self._fixed_video_resolution = None
            # logger.info(f"开始处理新视频: {video_id}")
        else:
            logger.debug(f"继续处理同一视频: {video_id}，保持分辨率: {self._fixed_video_resolution}")
        self._current_video_id = video_id

    def get_prompt(self, query, mc=False):
        if self.time_prompt:
             prompt =  f"\n{query}<|im_end|><|im_start|>assistant\n"
        else:
            prompt =  f"<|vision_end|>\n{query}<|im_end|><|im_start|>assistant\n"
        if mc:
            prompt += 'Best option: ('
        return prompt

    @torch.inference_mode()
    def _encode_video_chunk(self, video_chunk, video_id=None, chunk_idx=None):
        """
        Encode a chunk of video by invoking the full Qwen2-VL model to update the
        language model KV cache. This avoids relying on internal vision feature APIs.
        Accepts video_chunk as a numpy array or tensor with shape (T, H, W, 3) or (T, C, H, W).
        
        Args:
            video_chunk: 视频chunk数据
            video_id: 视频标识符，用于跟踪是否为同一视频
            chunk_idx: chunk索引，用于调试
        """
        # 检查是否为新视频
        if video_id is not None and video_id != self._current_video_id:
            self.reset_video_session(video_id)
        
        # Normalize to TCHW torch tensor
        if not isinstance(video_chunk, torch.Tensor):
            videos = torch.from_numpy(video_chunk)
        else:
            videos = video_chunk
        if videos.ndim == 4:
            # THWC -> TCHW
            if videos.shape[-1] == 3:
                videos = videos.permute(0, 3, 1, 2).contiguous()
        
        # 应用固定分辨率调整
        videos = self._apply_fixed_resolution_resize(videos, chunk_idx)
        # Ensure the frame count is divisible by temporal_patch_size (default 2)
        temporal_patch = getattr(getattr(self, 'processor', None), 'image_processor', None)
        temporal_patch_size = 2
        if temporal_patch is not None and hasattr(temporal_patch, 'temporal_patch_size'):
            temporal_patch_size = temporal_patch.temporal_patch_size
        if videos.shape[0] % temporal_patch_size != 0:
            # pad by repeating the last frame to satisfy divisibility
            pad_count = temporal_patch_size - (videos.shape[0] % temporal_patch_size)
            last_frame = videos[-1:].expand(pad_count, -1, -1, -1)
            videos = torch.cat([videos, last_frame], dim=0)
        
        chat_text = "<|video_pad|>"
        try:
            inputs = self.processor(text=[chat_text], images=None, videos=[videos], return_tensors="pt")
        except Exception as e:
            logger.error(f"处理视频chunk时出错: {e}")
            breakpoint()
        
        # Move tensors to device
        inputs.pop('attention_mask', None)
        for k, v in list(inputs.items()):
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)
        inputs['pixel_values_videos'] = inputs['pixel_values_videos'].to(self.device, self.dtype)

        # Derive dynamic block size from video_grid_thw if available: block_size = H_tokens * W_tokens per temporal slice
        dynamic_block_size = None
        if "video_grid_thw" in inputs:
            grid = inputs["video_grid_thw"][0]  # (T, H, W) in tokens
            # H and W here are token grid sizes already considering merge_size in processor
            dynamic_block_size = int(grid[1].item() / 2 * grid[2].item() / 2)
        # Fallback: use default n_frame_tokens
        if dynamic_block_size is None or dynamic_block_size <= 0:
            dynamic_block_size = self.n_frame_tokens
        
        # Propagate dynamic sizes to each attention module so the KV cache manager uses them
        if hasattr(self, 'model') and hasattr(self.model, 'layers'):
            for layer in self.model.layers:
                if hasattr(layer, 'self_attn'):
                    attn = layer.self_attn
                    setattr(attn, "dynamic_block_size", dynamic_block_size)
                    setattr(attn, "dynamic_exc_block_size", dynamic_block_size)
        elif hasattr(self, 'language_model') and hasattr(self.language_model.model, 'layers'):
            for layer in self.language_model.model.layers:
                if hasattr(layer, 'self_attn'):
                    attn = layer.self_attn
                    setattr(attn, "dynamic_block_size", dynamic_block_size)
                    setattr(attn, "dynamic_exc_block_size", dynamic_block_size)
        if hasattr(self, 'kv_cache') and self.kv_cache is not None:
            if not isinstance(self.kv_cache, tuple):
                self.kv_cache.set_block_size(dynamic_block_size)
        
        # Provide mRoPE config/runtime to cache (Qwen2-VL path)
        try:
            if hasattr(self, 'kv_cache') and self.kv_cache is not None and not isinstance(self.kv_cache, tuple):
                cfg = getattr(self, 'config', None)
                if cfg is not None and hasattr(cfg, 'vision_config'):
                    spatial_merge_size = getattr(cfg.vision_config, 'spatial_merge_size', 2)
                else:
                    spatial_merge_size = 2
                image_token_id = getattr(cfg, 'image_token_id', None) if cfg is not None else None
                video_token_id = getattr(cfg, 'video_token_id', None) if cfg is not None else None
                vision_start_token_id = getattr(cfg, 'vision_start_token_id', None) if cfg is not None else None
                vision_end_token_id = getattr(cfg, 'vision_end_token_id', None) if cfg is not None else None
                if (image_token_id is not None and video_token_id is not None and vision_start_token_id is not None and vision_end_token_id is not None):
                    self.kv_cache.set_mrope_config(
                        image_token_id=int(image_token_id),
                        video_token_id=int(video_token_id),
                        vision_start_token_id=int(vision_start_token_id),
                        vision_end_token_id=int(vision_end_token_id),
                        spatial_merge_size=int(spatial_merge_size),
                        is_qwen2vl=True,
                    )
                # Non-time-prompt path here; runtime grids
                image_grid = inputs['image_grid_thw'] if 'image_grid_thw' in inputs else None
                video_grid = inputs['video_grid_thw'] if 'video_grid_thw' in inputs else None
                self.kv_cache.set_mrope_runtime(
                    image_grid_thw=image_grid,
                    video_grid_thw=video_grid,
                    is_time_prompt=False,
                    num_time_tokens=0,
                )
        except Exception as e:
            logger.warning(f"Setting mRoPE config/runtime in wrapper failed: {e}")
        
        outputs = self(
            **inputs,
            use_cache=True,
            return_dict=True,
            past_key_values=self.kv_cache,
        )
        self.kv_cache = outputs.past_key_values
        
        # 记录第一个chunk的block_size信息
        if chunk_idx == 0 or chunk_idx is None:
            logger.info(f"第一个chunk的dynamic_block_size: {dynamic_block_size}")
        
        return dynamic_block_size

    def _apply_fixed_resolution_resize(self, videos, chunk_idx=None):
        """
        应用固定分辨率调整：第一个chunk确定分辨率，后续chunks使用相同分辨率
        
        Args:
            videos: 视频tensor (T, C, H, W)
            chunk_idx: chunk索引
        
        Returns:
            调整后的视频tensor
        """
        nframes, _, height, width = videos.shape
        nframes = max(64, nframes)
        
        if self._fixed_video_resolution is None:
            # 第一个chunk：计算并保存固定分辨率
            max_pixels = max(min(VIDEO_MAX_PIXELS, VIDEO_TOTAL_PIXELS / nframes * FRAME_FACTOR), int(VIDEO_MIN_PIXELS * 1.05))
            resized_height, resized_width = smart_resize( 
                height,
                width,
                factor=IMAGE_FACTOR,
                min_pixels=VIDEO_MIN_PIXELS,
                max_pixels=max_pixels,
            )
            self._fixed_video_resolution = (resized_height, resized_width)
            logger.info(f"确定固定视频分辨率: {resized_height}x{resized_width} (原始: {height}x{width})")
        else:
            # 后续chunks：使用已确定的固定分辨率
            resized_height, resized_width = self._fixed_video_resolution
            if chunk_idx is not None:
                logger.debug(f"Chunk {chunk_idx}: 使用固定分辨率 {resized_height}x{resized_width}")
        
        # 应用分辨率调整
        if (height, width) != (resized_height, resized_width):
            videos = transforms.functional.resize(
                videos,
                [resized_height, resized_width],
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ).float()
        
        return videos


def load_model(
    model_path: str = 'qwen-vl/Qwen2-VL-7B-Instruct',
    n_init: Optional[int] = None,
    n_local: Optional[int] = None,
    topk: int = 64,
    chunk_size: int = 1,
    use_hybrid_similarity: bool = True,
    convert_to_streaming=False,
    **kwargs,
):
    """
    Load Qwen2VL under the ReKV attention patch.
    """
    import os
    import glob

    time_prompt = kwargs.pop('time_prompt', False)
    if isinstance(time_prompt, str):
        time_prompt = time_prompt.lower() in ('true', '1', 'yes', 'y')
    
    # 处理训练输出目录的情况 - 自动查找最新的检查点
    original_model_path = model_path
    processor_path = model_path
    is_lora_model = False
    base_model_path = None
    
    # 检查是否是训练输出目录（包含checkpoint子目录）
    if os.path.isdir(model_path):
        checkpoint_dirs = glob.glob(os.path.join(model_path, 'checkpoint-*'))
        if checkpoint_dirs:
            # 找到最新的检查点目录
            latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split('-')[-1]))
            print(f"检测到训练输出目录，使用最新检查点: {latest_checkpoint}")
            
            # 检查是否是LoRA/PEFT模型
            adapter_config_path = os.path.join(latest_checkpoint, 'adapter_config.json')
            if os.path.exists(adapter_config_path):
                import json
                with open(adapter_config_path, 'r') as f:
                    adapter_config = json.load(f)
                base_model_path = adapter_config.get('base_model_name_or_path', 'Qwen/Qwen2-VL-7B-Instruct')
                is_lora_model = True
                model_path = latest_checkpoint  # 使用检查点目录作为模型路径
                processor_path = base_model_path  # 使用基础模型路径加载processor
                print(f"检测到LoRA模型，基础模型: {base_model_path}")
            
            # 检查检查点目录是否包含必要的配置文件
            elif os.path.exists(os.path.join(latest_checkpoint, 'preprocessor_config.json')):
                processor_path = latest_checkpoint
                model_path = latest_checkpoint
                print(f"使用检查点目录: {latest_checkpoint}")
            else:
                # 如果检查点目录没有processor配置，尝试使用基础模型路径
                print(f"检查点目录缺少processor配置，使用默认基础模型")
                processor_path = 'Qwen/Qwen2-VL-7B-Instruct'
                model_path = original_model_path
    
    # Processor - 始终从基础模型或有效配置路径加载
    try:
        processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
        print(f"成功从 {processor_path} 加载processor")
    except Exception as e:
        print(f"从 {processor_path} 加载processor失败: {e}")
        # 降级到默认基础模型
        processor_path = 'Qwen/Qwen2-VL-7B-Instruct'
        processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
        print(f"降级使用默认基础模型加载processor: {processor_path}")
    
    tokenizer = processor.tokenizer

    # Init prompt for n_init sizing
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if time_prompt:
        init_prompt = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'
    else:
        init_prompt = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|>'
    print(init_prompt)
    init_prompt_ids = tokenizer(init_prompt, return_tensors="pt").input_ids.to(device)

    # ReKV config
    n_frame_tokens = 196  # not used directly for images; kept consistent with other wrappers
    inf_llm_config = {
        'n_init': init_prompt_ids.shape[1] if n_init is None else n_init,
        'n_local': n_local,
        'fattn': True,
        'block_size': n_frame_tokens,
        'topk': topk,
        'chunk_size': chunk_size,
        'max_cached_block': 128,
        'exc_block_size': n_frame_tokens,
        'pin_memory': True,
        'use_hybrid_similarity': use_hybrid_similarity,
    }

    # Model (avoid passing non-JSON-serializable objects via kwargs)
    if is_lora_model:
        print(f"加载LoRA模型，基础模型: {base_model_path}, 适配器: {model_path}")
        # 先加载基础模型
        model = Qwen2VL_ReKV.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            low_cpu_mem_usage=True,
            n_frame_tokens=n_frame_tokens,
            n_local=n_local,
            topk=topk,
            chunk_size=chunk_size,
            trust_remote_code=True,
        )
        
        # 然后加载LoRA适配器
        try:
            from peft import PeftModel
            peft_model = PeftModel.from_pretrained(model, model_path)
            try:
                # 合并LoRA权重并卸载适配器，返回基础模型
                model = peft_model.merge_and_unload()
                print(f"成功合并LoRA权重并卸载适配器: {model_path}")
            except Exception as merge_err:
                # 合并失败：保留适配器在线，直接使用 peft_model 进行推理
                model = peft_model
                print(f"合并LoRA权重失败({merge_err})，保留适配器在线: {model_path}")
            print(f"成功加载LoRA适配器: {model_path}")
        except Exception as e:
            print(f"加载LoRA适配器失败: {e}")
            print("继续使用基础模型...")
    else:
        model = Qwen2VL_ReKV.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            low_cpu_mem_usage=True,
            n_frame_tokens=n_frame_tokens,
            n_local=n_local,
            topk=topk,
            chunk_size=chunk_size,
            trust_remote_code=True,
        )

    # Assign processor and init prompt IDs post-load
    model.processor = processor
    model.init_prompt_ids = init_prompt_ids

    # Patch attention into ReKV style; for Qwen2VL, the text backbone is model.model
    if convert_to_streaming == 'true':
        model = convert_qwen2vl_to_streaming(model) 
        # model = convert_llavaov_to_streaming(model)
    elif convert_to_streaming == 'baseline':
        pass
    else:
        print('patch model')
        model = patch_hf(model, **inf_llm_config)
    # model.language_model = model.model  # unify attribute used by Abstract_ReKV
    try:
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            model.model.rekv_config = inf_llm_config
        model.time_prompt = time_prompt
    except Exception:
        logger.warning("Failed to attach rekv_config to language model; dynamic cache may use defaults.")
    
    
    for k, v in inf_llm_config.items():
        logger.info(f'{k}: {v}')
    logger.info(f'n_frame_tokens: {n_frame_tokens}')

    model.eval()
    return model, processor 