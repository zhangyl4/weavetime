import torch
import numpy as np
from logzero import logger
import random
import math
import torch.nn.functional as F
import json
import os


def calculate_original_votes(
    all_layer_indices,
    short_memory_layer,
    layer_weights=None,
    spread_stddev=1.5,
    spread_radius=5,
):
    index_votes = {}
    for layer_idx, layer_indices in all_layer_indices:
        layer_w = 1.0 if layer_idx >= short_memory_layer else 0.0
        if layer_weights and layer_idx < len(layer_weights):
            try:
                w = float(layer_weights[layer_idx])
                if layer_idx >= short_memory_layer:
                    layer_w = w
            except (ValueError, TypeError):
                if layer_idx >= short_memory_layer:
                    layer_w = 1.0

        if layer_w == 0:
            continue

        for idxs in layer_indices:
            for idx in idxs:
                index_votes[idx] = index_votes.get(idx, 0.0) + layer_w
    return index_votes


class TimePromptMixin:
    def _setup_chat_template(self):

            
        processor_class = self.qa_model.processor.__class__.__name__
        if 'LlavaOnevision' not in processor_class:
            logger.info(f'Skipping chat_template update for {processor_class} (only for LlavaOnevisionProcessor)')
            return
        new_template = """{% for message in messages %}{{ '<|im_start|>' + message['role'] + ' ' }}{% for content in message['content'] %}{% if content['type'] == 'image' %}{{ '<image>' }}{% elif content['type'] == 'video' %}{{ '<video>' }}{% elif content['type'] == 'text' %}{% if message['role'] != 'assistant' %}{{ '\n' + content['text'] }}{% else %}{% generation %}{{ '\n' + content['text'] }}{% endgeneration %}{% endif %}{% endif %}{% endfor %}{{ '<|im_end|>' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"""
        if hasattr(self.qa_model.processor, 'tokenizer'):
            self.qa_model.processor.tokenizer.chat_template = new_template
            self.qa_model.processor.chat_template = new_template

    def encode_video_with_time_prompts(self, video, video_id=None, encode_chunk_size=64):
        self.qa_model.reset_video_session(video_id)
        num_frames = video.shape[0]
        num_chunks = (num_frames + encode_chunk_size - 1) // encode_chunk_size
        logger.debug(f'Encoding {num_frames} frames in {num_chunks} chunks (chunk_size={encode_chunk_size}), with time prompts every {self.frames_per_input} frames')
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * encode_chunk_size
            chunk_end = min(chunk_start + encode_chunk_size, num_frames)
            chunk_frames = video[chunk_start:chunk_end]
            chunk_num_frames = len(chunk_frames)
            num_segments = (chunk_num_frames + self.frames_per_input - 1) // self.frames_per_input
            all_videos = []
            time_texts = []
            # Detect processor type
            processor_class = self.qa_model.processor.__class__.__name__ if hasattr(self.qa_model, 'processor') else ''
            is_qwen2vl = 'Qwen2VL' in processor_class
            # Qwen2-VL temporal patch size (if available)
            temporal_patch = getattr(getattr(self.qa_model, 'processor', None), 'image_processor', None)
            temporal_patch_size = None
            if is_qwen2vl and temporal_patch is not None and hasattr(temporal_patch, 'temporal_patch_size'):
                temporal_patch_size = temporal_patch.temporal_patch_size
            for seg_idx in range(num_segments):
                seg_start = seg_idx * self.frames_per_input
                seg_end = min((seg_idx + 1) * self.frames_per_input, chunk_num_frames)
                segment_frames = chunk_frames[seg_start:seg_end]
                abs_start = chunk_start + seg_start
                abs_end = chunk_start + seg_end
                start_timestamp = abs_start / self.sample_fps
                end_timestamp = abs_end / self.sample_fps
                time_text = f'Time={start_timestamp:07.1f}-{end_timestamp:07.1f}s'
                time_texts.append(time_text)
                # Convert to tensor and TCHW
                if not is_qwen2vl:
                    seg_tensor = segment_frames
                else:
                    if not isinstance(segment_frames, torch.Tensor):
                        seg_tensor = torch.from_numpy(segment_frames)
                    else:
                        seg_tensor = segment_frames
                    if seg_tensor.ndim == 4 and seg_tensor.shape[-1] == 3:
                        seg_tensor = seg_tensor.permute(0, 3, 1, 2).contiguous()
                    # Apply fixed resolution consistent across chunks if available
                    # breakpoint()
                    if temporal_patch_size is not None:
                        seg_tensor = self.qa_model._apply_fixed_resolution_resize(seg_tensor, chunk_idx=chunk_idx)
                    if temporal_patch_size is not None and seg_tensor.shape[0] % temporal_patch_size != 0:
                        pad_count = temporal_patch_size - (seg_tensor.shape[0] % temporal_patch_size)
                        last_frame = seg_tensor[-1:].expand(pad_count, -1, -1, -1)
                        seg_tensor = torch.cat([seg_tensor, last_frame], dim=0)
                all_videos.append(seg_tensor)
            conversation = [{'role': 'system', "content": [{"type": "text", "text": "You are a helpful assistant."}]}, {"role": "user", "content": []}]
            for i in range(num_segments):
                conversation[1]["content"].append({"type": "text", "text": time_texts[i]})
                conversation[1]["content"].append({"type": "video", "video": all_videos[i]})
            chat_text = self.qa_model.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
            init_prompt = self.qa_model.processor.tokenizer.decode(self.qa_model.init_prompt_ids[0])
            chat_text = chat_text.replace(init_prompt, '')
            if is_qwen2vl:
                chat_text = chat_text.replace('<|im_end|>\n', '')
            inputs = self.qa_model.processor(text=[chat_text], videos=all_videos, return_tensors="pt")
            tokenizer = self.qa_model.processor.tokenizer
            im_end_token_id = tokenizer.encode('<|im_end|>', add_special_tokens=False)
            unwanted_token_ids = set()
            unwanted_token_ids.update(im_end_token_id)
            if 'input_ids' in inputs:
                mask = ~torch.isin(inputs['input_ids'], torch.tensor(list(unwanted_token_ids), device=inputs['input_ids'].device))
                filtered_input_ids = []
                for row, m in zip(inputs['input_ids'], mask):
                    filtered_row = row[m]
                    filtered_input_ids.append(filtered_row)
                from torch.nn.utils.rnn import pad_sequence
                inputs['input_ids'] = pad_sequence(filtered_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
                if 'attention_mask' in inputs:
                    filtered_attention_mask = []
                    for row, m in zip(inputs['attention_mask'], mask):
                        filtered_row = row[m]
                        filtered_attention_mask.append(filtered_row)
                    inputs['attention_mask'] = pad_sequence(filtered_attention_mask, batch_first=True, padding_value=0)

            # 移除可能被 processor 再次注入的 init prompt（避免与已缓存的 init prompt 重复）
            try:
                if 'input_ids' in inputs and hasattr(self.qa_model, 'init_prompt_ids') and self.qa_model.init_prompt_ids is not None:
                    init_ids = self.qa_model.init_prompt_ids[0]
                    if isinstance(init_ids, torch.Tensor):
                        init_ids = init_ids.tolist()
                    init_len = len(init_ids) if isinstance(init_ids, (list, tuple)) else 0
                    if init_len > 0:
                        trimmed_input_ids = []
                        trimmed_attention_masks = [] if 'attention_mask' in inputs else None
                        for i in range(inputs['input_ids'].shape[0]):
                            row = inputs['input_ids'][i]
                            row_list = row.tolist()
                            # 仅在前缀完全匹配时裁剪
                            if len(row_list) >= init_len and row_list[:init_len] == init_ids:
                                new_row = row[init_len:]
                                trimmed_input_ids.append(new_row)
                                if trimmed_attention_masks is not None:
                                    trimmed_attention_masks.append(inputs['attention_mask'][i][init_len:])
                            else:
                                trimmed_input_ids.append(row)
                                if trimmed_attention_masks is not None:
                                    trimmed_attention_masks.append(inputs['attention_mask'][i])
                        from torch.nn.utils.rnn import pad_sequence
                        inputs['input_ids'] = pad_sequence(trimmed_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
                        if trimmed_attention_masks is not None:
                            inputs['attention_mask'] = pad_sequence(trimmed_attention_masks, batch_first=True, padding_value=0)
            except Exception as e:
                logger.warning(f"Init prompt trimming skipped due to error: {e}")

            # 强制移除 attention_mask，使用纯因果掩码以避免 KV/mask 维度不一致
            inputs.pop('attention_mask', None)
            for k, v in list(inputs.items()):
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.qa_model.device)
            inputs['pixel_values_videos'] = inputs['pixel_values_videos'].to(self.qa_model.device, self.qa_model.dtype)

            if "video_grid_thw" in inputs:
                grid = inputs["video_grid_thw"][0]
                dynamic_block_size = int(grid[1].item() / 2 * grid[2].item() / 2)
            else:
                dynamic_block_size = self.qa_model.n_frame_tokens
            actual_time_text = time_texts[0]
            time_tokens = self.qa_model.processor.tokenizer(actual_time_text, add_special_tokens=False).input_ids
            num_time_tokens = len(time_tokens) + 1 + 1 #newline and \n, vision_start and vision end
            dynamic_block_size += num_time_tokens
            if hasattr(self.qa_model, 'model') and hasattr(self.qa_model.model, 'layers'):
                for layer in self.qa_model.model.layers:
                    if hasattr(layer, 'self_attn'):
                        attn = layer.self_attn
                        setattr(attn, "dynamic_block_size", dynamic_block_size)
                        setattr(attn, "dynamic_exc_block_size", dynamic_block_size)
            elif hasattr(self.qa_model, 'language_model') and hasattr(self.qa_model.language_model.model, 'layers'):
                for layer in self.qa_model.language_model.model.layers:
                    if hasattr(layer, 'self_attn'):
                        attn = layer.self_attn
                        setattr(attn, "dynamic_block_size", dynamic_block_size)
                        setattr(attn, "dynamic_exc_block_size", dynamic_block_size)
            if hasattr(self.qa_model, 'kv_cache') and self.qa_model.kv_cache is not None:
                if not isinstance(self.qa_model.kv_cache, tuple):
                    self.qa_model.kv_cache.set_block_size(dynamic_block_size)
                # For Qwen2-VL, provide mRoPE runtime and config so cache can compute 3D position ids
                try:
                    processor_class = self.qa_model.processor.__class__.__name__ if hasattr(self.qa_model, 'processor') else ''
                    is_qwen2vl = 'Qwen2VL' in processor_class
                    if is_qwen2vl and not isinstance(self.qa_model.kv_cache, tuple):
                        # Runtime shapes for this forward
                        image_grid = inputs['image_grid_thw'] if 'image_grid_thw' in inputs else None
                        video_grid = inputs['video_grid_thw'] if 'video_grid_thw' in inputs else None
                        # Time-prompt path here
                        self.qa_model.kv_cache.set_mrope_runtime(
                            image_grid_thw=image_grid,
                            video_grid_thw=video_grid,
                            is_time_prompt=True,
                            num_time_tokens=int(num_time_tokens),
                        )
                        # Static token IDs and merge size from config
                        cfg = getattr(self.qa_model, 'config', None)
                        if cfg is not None and hasattr(cfg, 'vision_config'):
                            spatial_merge_size = getattr(cfg.vision_config, 'spatial_merge_size', 2)
                            image_token_id = getattr(cfg, 'image_token_id', None)
                            video_token_id = getattr(cfg, 'video_token_id', None)
                            vision_start_token_id = getattr(cfg, 'vision_start_token_id', None)
                            vision_end_token_id = getattr(cfg, 'vision_end_token_id', None)
                            if (image_token_id is not None and video_token_id is not None and vision_start_token_id is not None):
                                self.qa_model.kv_cache.set_mrope_config(
                                    image_token_id=int(image_token_id),
                                    video_token_id=int(video_token_id),
                                    vision_start_token_id=int(vision_start_token_id),
                                    vision_end_token_id=int(vision_end_token_id),
                                    spatial_merge_size=int(spatial_merge_size),
                                    is_qwen2vl=True,
                                )
                except Exception as e:
                    logger.warning(f"Setting mRoPE runtime/config failed: {e}")
                
            outputs = self.qa_model(
                **inputs,
                use_cache=True,
                return_dict=True,
                past_key_values=self.qa_model.kv_cache,
            )
            self.qa_model.kv_cache = outputs.past_key_values
            logger.debug(f'Chunk {chunk_idx}: KV-Cache RAM usage: {self.qa_model.calc_memory_usage() / (1024**3):.1f} GB')

class RecentLayeredRetrievalMixin:
    """Utilities shared by recent-layered retrieval variants."""

    def get_recent_frame_indices(self, total_frames=None, retrieve_size=64):
        if total_frames is None:
            if hasattr(self.qa_model, 'kv_cache') and self.qa_model.kv_cache:
                first_layer = self.qa_model.kv_cache[0]
                if hasattr(first_layer, 'length') and first_layer.length > 0:
                    total_length = first_layer.length
                    estimated_frame_tokens = getattr(first_layer, 'block_size')
                    total_frames = max(1, total_length // estimated_frame_tokens)
                else:
                    total_frames = 1000
            else:
                total_frames = 1000
        if total_frames <= retrieve_size:
            return [list(range(total_frames))]
        else:
            start_idx = total_frames - retrieve_size if total_frames - retrieve_size > 0 else 0
            return [list(range(start_idx, total_frames))]

    def question_answering_with_layered_retrieval(self, input_text, max_new_tokens=128):
        device = self.qa_model.device
        query_text = input_text[self.query_type]
        if hasattr(self.qa_model, 'processor') and hasattr(self.qa_model.processor, 'tokenizer'):
            input_ids = self.qa_model.processor.tokenizer(query_text).input_ids
        else:
            input_ids = query_text
        input_ids = torch.as_tensor([input_ids], device=device)

        # prepare retrieval
        for layer_idx, layer_kv in enumerate(self.qa_model.kv_cache):
            layer_kv.set_retrieval()
            
        # first pass
        if hasattr(self.qa_model, 'language_model'):
            out = self.qa_model.language_model(input_ids=input_ids, use_cache=True, past_key_values=self.qa_model.kv_cache, output_hidden_states=True)
        else:
            out = self.qa_model(input_ids=input_ids, use_cache=True, past_key_values=self.qa_model.kv_cache, output_hidden_states=True)
        past_key_values = out.past_key_values

        retrieval_info = self._extract_retrieval_info()

        # NOTE : code for forward twice to get retrieval info, -1 is recent mode
        assert len(self.short_memory_layers) == 1, "short_memory_layers must be 1"
        short_memory_layer = self.short_memory_layers[0]
        recent_mode = True if short_memory_layer == -1 else False
        short_memory_layer = 0 if recent_mode else short_memory_layer
        
        # entropy per layer
        layer_entropies = {}
        if hasattr(out, 'hidden_states') and out.hidden_states:
            for layer_idx, hidden_state in enumerate(out.hidden_states):
                if hasattr(self.qa_model, 'language_model'):
                    layer_logits = self.qa_model.language_model.lm_head(hidden_state)
                else:
                    layer_logits = self.qa_model.lm_head(hidden_state)
                layer_logits = layer_logits[:, -1, :]
                layer_logits = layer_logits.float()
                top_k_scores, top_k_indices = torch.topk(layer_logits, 10)
                probabilities = torch.softmax(top_k_scores, dim=-1)
                entropy = torch.sum(-probabilities * torch.log(probabilities + 1e-10)) / np.log(10)
                layer_entropies[f'layer_{layer_idx}'] = float(entropy.item())

        dynamic_retrieve_size =  None
        if self.use_dynamic_size:
            try:
                last_layer_idx = len(self.qa_model.kv_cache) - 1 if hasattr(self.qa_model, 'kv_cache') and self.qa_model.kv_cache else None
                if last_layer_idx is None and layer_entropies:
                    last_layer_idx = max(int(str(k).split('_')[-1]) for k in layer_entropies.keys())
                last_entropy = layer_entropies.get(f'layer_{last_layer_idx}', None) if last_layer_idx is not None else None
                if isinstance(last_entropy, (int, float)):
                    norm = float(max(0.0, min(1.0, float(last_entropy))))
                    base_size = int(self.retrieve_size) // 2
                    kv_topk_limit = None
                    try:
                        if hasattr(self.qa_model, 'kv_cache') and self.qa_model.kv_cache:
                            kv_topk_limit = int(getattr(self.qa_model.kv_cache[0], 'topk', base_size))
                    except Exception:
                        kv_topk_limit = None
                    max_size = base_size if kv_topk_limit is None else min(base_size, kv_topk_limit)
                    min_size = max(int(self.chunk_size), max(1, base_size // 2))
                    dyn2 = int(round(min_size + (max_size - min_size) * math.exp(-2 * norm)))
                    dynamic_retrieve_size = max(min_size, min(max_size, dyn2))
            except Exception as e:
                logger.warning(f"Dynamic retrieve size (last-layer entropy) failed: {e}")

        for layer_kv in self.qa_model.kv_cache:
            layer_kv.reset_retrieval()
        # 每轮QA结束，关闭激活的base（恢复常规KV增长模式）
        if hasattr(self.qa_model, 'kv_cache') and self.qa_model.kv_cache is not None:
            for layer_kv in self.qa_model.kv_cache:
                if hasattr(layer_kv, 'deactivate_base'):
                    layer_kv.deactivate_base()
                    
        retrieval_info['dynamic_retrieve_size'] = dynamic_retrieve_size if dynamic_retrieve_size is not None else self.retrieve_size

        assert len(self.short_memory_layers) == 1, "short_memory_layers must be 1"
        short_memory_layer = self.short_memory_layers[0]

        all_layer_indices = []
        for layer_idx in range(len(self.qa_model.kv_cache)):
            layer_name = f'layer_{layer_idx}'
            if layer_name in retrieval_info['retrieval_records']:
                indices = retrieval_info['retrieval_records'][layer_name]['retrieved_indices']
                all_layer_indices.append((layer_idx, indices))

        target_retrieve_size = retrieval_info.get('dynamic_retrieve_size', self.retrieve_size)
        try:
            if hasattr(self.qa_model, 'kv_cache') and self.qa_model.kv_cache:
                kv_topk_limit = int(getattr(self.qa_model.kv_cache[0], 'topk', target_retrieve_size))
                target_retrieve_size = min(int(target_retrieve_size), kv_topk_limit)
        except Exception:
            target_retrieve_size = int(target_retrieve_size)

        index_votes = calculate_original_votes(
            all_layer_indices=[(layer_idx, indices) for layer_idx, indices in all_layer_indices],
            short_memory_layer=short_memory_layer,
            layer_weights=getattr(self, 'layer_weights', None),
            spread_stddev=3.5,
            spread_radius=5,
        )

        sorted_indices = sorted(index_votes.items(), key=lambda x: (x[1], random.random()), reverse=True)
        
        top_indices = [idx for idx, votes in sorted_indices[:target_retrieve_size]]
        if recent_mode:
            recent_indices = self.get_recent_frame_indices(retrieve_size=self.retrieve_size)
            top_indices.extend(recent_indices[0])
        top_indices = sorted(list(dict.fromkeys(top_indices)))

        for layer_idx, layer_kv in enumerate(self.qa_model.kv_cache):
            layer_kv.set_retrieval()
            layer_kv.set_retrieved_block_indices([top_indices])

        if hasattr(self.qa_model, 'language_model'):
            out = self.qa_model.language_model(input_ids=input_ids, use_cache=True, past_key_values=self.qa_model.kv_cache)
        else:
            out = self.qa_model(input_ids=input_ids, use_cache=True, past_key_values=self.qa_model.kv_cache)
        past_key_values = out.past_key_values

        retrieval_info = self._extract_retrieval_info()
        retrieval_info['layer_entropies'] = layer_entropies
        retrieval_info['dynamic_retrieve_size'] = dynamic_retrieve_size if dynamic_retrieve_size is not None else self.retrieve_size

        for layer_kv in self.qa_model.kv_cache:
            layer_kv.reset_retrieval()

        qa_result = self._continue_question_answering(input_text, max_new_tokens, past_key_values, retrieval_info)
        # 每轮QA结束，关闭激活的base（恢复常规KV增长模式）
        if hasattr(self.qa_model, 'kv_cache') and self.qa_model.kv_cache is not None:
            for layer_kv in self.qa_model.kv_cache:
                if hasattr(layer_kv, 'deactivate_base'):
                    layer_kv.deactivate_base()
        return qa_result, retrieval_info


class EntropyAdaptiveRecentRetrievalMixin(RecentLayeredRetrievalMixin):
    """Recent-first retrieval with entropy-adaptive fallback to auto retrieval.

    Workflow:
    1) First forward without retrieval to compute per-layer entropy, saving query_states.
    2) Compute the average entropy of the last K layers (default K=4).
    3) If avg >= threshold (default 0.6), enable retrieval using saved query_states;
       else reuse the first forward result.
    """

    def question_answering_with_layered_retrieval(self, input_text, max_new_tokens=128):
        import time
        time1 = time.time()
        device = self.qa_model.device
        query_text = input_text[self.query_type]
        if hasattr(self.qa_model, 'processor') and hasattr(self.qa_model.processor, 'tokenizer'):
            input_ids = self.qa_model.processor.tokenizer(query_text).input_ids
        else:
            input_ids = query_text
        input_ids = torch.as_tensor([input_ids], device=device)
        entropy_threshold = getattr(self, 'entropy_threshold', 0.6)
        entropy_window_layers = int(getattr(self, 'entropy_window_layers', 4))
        # print(f"entropy_threshold: {entropy_threshold}, entropy_window_layers: {entropy_window_layers}")
        # Activate base immediately after first pass so the question tokens
        # become the persistent base (avoid being offloaded and avoid re-prefill)
        if hasattr(self.qa_model, 'kv_cache') and self.qa_model.kv_cache is not None:
            for layer_kv in self.qa_model.kv_cache:
                if hasattr(layer_kv, 'get_sliding_window_kv') and hasattr(layer_kv, 'activate_base'):
                    try:
                        k_cur, v_cur = layer_kv.get_sliding_window_kv()
                        layer_kv.activate_base(k_cur, v_cur)
                    except Exception:
                        pass
        
        # 1. First pass: no retrieval, use cache directly, save query_states
        # Enable saving query_states for all layers
        if hasattr(self.qa_model.kv_cache, 'save_query_states'):
            self.qa_model.kv_cache.save_query_states(True)
        else:
            for layer_kv in self.qa_model.kv_cache:
                if hasattr(layer_kv, 'save_query_states'):
                    layer_kv.save_query_states(True)
        
        # First forward without retrieval (use cache directly)
        if hasattr(self.qa_model, 'language_model'):
            out_first = self.qa_model.language_model(
                input_ids=input_ids,
                use_cache=True,
                past_key_values=self.qa_model.kv_cache,
                output_hidden_states=True,
            )
        else:
            out_first = self.qa_model(
                input_ids=input_ids,
                use_cache=True,
                past_key_values=self.qa_model.kv_cache,
                output_hidden_states=True,
            )

        time2 = time.time()
        logger.debug(f'First pass time: {time2 - time1:.2f} seconds')
        
        # Compute entropy per layer (top-10 softmax entropy, base 10)
        layer_entropies = {}
        if hasattr(out_first, 'hidden_states') and out_first.hidden_states:
            for layer_idx, hidden_state in enumerate(out_first.hidden_states):
                if hasattr(self.qa_model, 'language_model'):
                    layer_logits = self.qa_model.language_model.lm_head(hidden_state)
                else:
                    layer_logits = self.qa_model.lm_head(hidden_state)
                layer_logits = layer_logits[:, -1, :].float()
                top_k_scores, _ = torch.topk(layer_logits, 10)
                probabilities = torch.softmax(top_k_scores, dim=-1)
                entropy = torch.sum(-probabilities * torch.log(probabilities + 1e-10)) / np.log(10)
                layer_entropies[f'layer_{layer_idx}'] = float(entropy.item())

        # Decide strategy based on last-K average entropy
        avg_entropy_lastk = None
        strategy = 'cache_only'
        if layer_entropies:
            layer_indices_sorted = sorted(
                [int(k.split('_')[-1]) for k in layer_entropies.keys()]
            )
            if layer_indices_sorted:
                k = min(entropy_window_layers, len(layer_indices_sorted))
                last_k_indices = layer_indices_sorted[-k:]
                vals = [layer_entropies[f'layer_{i}'] for i in last_k_indices]
                avg_entropy_lastk = float(sum(vals) / len(vals)) if vals else None
                if avg_entropy_lastk is not None and avg_entropy_lastk >= float(entropy_threshold):
                    strategy = 'retrieval'

        if strategy == 'cache_only':
            # Use first pass outputs directly; decode without re-prefill
            past_key_values = out_first.past_key_values
            first_step_logits = out_first.logits

            # Build retrieval_info for cache_only path
            retrieval_info = self._extract_retrieval_info()
            retrieval_info['strategy'] = strategy
            retrieval_info['avg_entropy_last4'] = avg_entropy_lastk if avg_entropy_lastk is not None else None
            retrieval_info['layer_entropies'] = layer_entropies

            # Disable/clear saved query states
            if hasattr(self.qa_model.kv_cache, 'save_query_states'):
                self.qa_model.kv_cache.save_query_states(False)
            else:
                for layer_kv in self.qa_model.kv_cache:
                    if hasattr(layer_kv, 'save_query_states'):
                        layer_kv.save_query_states(False)
            if hasattr(self.qa_model.kv_cache, 'use_saved_query_states'):
                self.qa_model.kv_cache.use_saved_query_states(False)
            else:
                for layer_kv in self.qa_model.kv_cache:
                    if hasattr(layer_kv, 'use_saved_query_states'):
                        layer_kv.use_saved_query_states(False)
            if hasattr(self.qa_model.kv_cache, 'clear_saved_query_states'):
                self.qa_model.kv_cache.clear_saved_query_states()
            else:
                for layer_kv in self.qa_model.kv_cache:
                    if hasattr(layer_kv, 'clear_saved_query_states'):
                        layer_kv.clear_saved_query_states()

            
            time3 = time.time()
            logger.debug(f'Cache only time: {time3 - time2:.2f} seconds')
            
            # Direct decoding without re-prefill
            tokenizer = self.qa_model.processor.tokenizer if hasattr(self.qa_model, 'processor') and hasattr(self.qa_model.processor, 'tokenizer') else None
            stop_token_ids = [tokenizer.eos_token_id] if tokenizer is not None and getattr(tokenizer, 'eos_token_id', None) is not None else []
            device = self.qa_model.device
            output_ids = []
            last_token_logits = first_step_logits[0, -1, :]
            _, indices = torch.topk(last_token_logits, 2)
            tokens = [int(index) for index in indices.tolist()]
            token = tokens[0]
            output_ids.append(token)
            stopped = token in stop_token_ids
            for _ in range(max_new_tokens - 1):
                if stopped:
                    break
                if hasattr(self.qa_model, 'language_model'):
                    out = self.qa_model.language_model(
                        input_ids=torch.as_tensor([[token]], device=device),
                        use_cache=True,
                        past_key_values=past_key_values,
                    )
                else:
                    out = self.qa_model(
                        input_ids=torch.as_tensor([[token]], device=device),
                        use_cache=True,
                        past_key_values=past_key_values,
                    )
                logits = out.logits
                past_key_values = out.past_key_values
                last_token_logits = logits[0, -1, :]
                _, indices = torch.topk(last_token_logits, 2)
                tokens = [int(index) for index in indices.tolist()]
                token = tokens[0]
                output_ids.append(token)
                stopped = token in stop_token_ids
                logger.debug(f'Cache only decode one time: {time.time() - time3} seconds')
            qa_result = self.qa_model.processor.tokenizer.decode(
                output_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
                clean_up_tokenization_spaces=True,
            ) if tokenizer is not None else ""

            # deactivte base
            if hasattr(self.qa_model, 'kv_cache') and self.qa_model.kv_cache is not None:
                for layer_kv in self.qa_model.kv_cache:
                    if hasattr(layer_kv, 'deactivate_base'):
                        layer_kv.deactivate_base()
            # breakpoint()
            time4 = time.time()
            logger.debug(f'Cache only decode time: {time4 - time3:.2f} seconds')
            return qa_result, retrieval_info
        # If entropy high: enable retrieval using saved query_states; else reuse first forward
        elif strategy == 'retrieval':
            # Disable saving query_states
            if hasattr(self.qa_model.kv_cache, 'save_query_states'):
                self.qa_model.kv_cache.save_query_states(False)
            else:
                for layer_kv in self.qa_model.kv_cache:
                    if hasattr(layer_kv, 'save_query_states'):
                        layer_kv.save_query_states(False)
            
            # Enable using saved query_states for retrieval
            if hasattr(self.qa_model.kv_cache, 'use_saved_query_states'):
                self.qa_model.kv_cache.use_saved_query_states(True)
            else:
                for layer_kv in self.qa_model.kv_cache:
                    if hasattr(layer_kv, 'use_saved_query_states'):
                        layer_kv.use_saved_query_states(True)
            
            # Enable retrieval
            for layer_kv in self.qa_model.kv_cache:
                layer_kv.set_retrieval()
            
            # Second forward with retrieval using saved query_states
            if hasattr(self.qa_model, 'language_model'):
                out_final = self.qa_model.language_model(
                    input_ids=input_ids,
                    use_cache=True,
                    past_key_values=self.qa_model.kv_cache,
                )
            else:
                out_final = self.qa_model(
                    input_ids=input_ids,
                    use_cache=True,
                    past_key_values=self.qa_model.kv_cache,
                )
            past_key_values = out_final.past_key_values


            # Collect retrieval info after the chosen strategy pass
            retrieval_info = self._extract_retrieval_info()
            retrieval_info['strategy'] = strategy
            retrieval_info['avg_entropy_last4'] = avg_entropy_lastk if avg_entropy_lastk is not None else None
            retrieval_info['layer_entropies'] = layer_entropies

            # Clean up flags
            # Disable saving query_states
            if hasattr(self.qa_model.kv_cache, 'save_query_states'):
                self.qa_model.kv_cache.save_query_states(False)
            else:
                for layer_kv in self.qa_model.kv_cache:
                    if hasattr(layer_kv, 'save_query_states'):
                        layer_kv.save_query_states(False)
            
            # Disable using saved query_states
            if hasattr(self.qa_model.kv_cache, 'use_saved_query_states'):
                self.qa_model.kv_cache.use_saved_query_states(False)
            else:
                for layer_kv in self.qa_model.kv_cache:
                    if hasattr(layer_kv, 'use_saved_query_states'):
                        layer_kv.use_saved_query_states(False)
            
            # Clear saved query_states
            if hasattr(self.qa_model.kv_cache, 'clear_saved_query_states'):
                self.qa_model.kv_cache.clear_saved_query_states()
            else:
                for layer_kv in self.qa_model.kv_cache:
                    if hasattr(layer_kv, 'clear_saved_query_states'):
                        layer_kv.clear_saved_query_states()
            
            # Reset retrieval flags
            for layer_kv in self.qa_model.kv_cache:
                layer_kv.reset_retrieval()

            qa_result = self._continue_question_answering(
                input_text, max_new_tokens, past_key_values, retrieval_info
            )
            
            # deactivte base
            if hasattr(self.qa_model, 'kv_cache') and self.qa_model.kv_cache is not None:
                for layer_kv in self.qa_model.kv_cache:
                    if hasattr(layer_kv, 'deactivate_base'):
                        layer_kv.deactivate_base()
            # breakpoint()
            return qa_result, retrieval_info


class WeightLoadingMixin:
    """Mixin for loading layer and head weights from JSON files."""
    
    def _load_layer_and_head_weights(self, layer_weight_path=None, head_weight_path=None):
        """
        Load layer and head weights from JSON files.
        
        Args:
            layer_weight_path: Path to JSON file containing layer weights
            head_weight_path: Path to JSON file containing head weights
        """
        self.layer_weights = None
        if layer_weight_path and os.path.isfile(layer_weight_path):
            try:
                with open(layer_weight_path, 'r') as f:
                    self.layer_weights = json.load(f)
                logger.info(f"Loaded layer weights from {layer_weight_path} ({len(self.layer_weights)})")
            except Exception as e:
                logger.warning(f"Failed to load layer weights: {e}")
        
        self.head_weights = None
        if head_weight_path and os.path.isfile(head_weight_path):
            try:
                with open(head_weight_path, 'r') as f:
                    self.head_weights = json.load(f)
                logger.info(f"Loaded head weights from {head_weight_path} (layers={len(self.head_weights)})")
            except Exception as e:
                logger.warning(f"Failed to load head weights: {e}")


class StandardRetrievalQAMixin:
    pass
