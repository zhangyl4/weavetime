import torch
import numpy as np
from logzero import logger
import random
import math
import torch.nn.functional as F


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
                all_videos.append(segment_frames)
            conversation = [{'role': 'system', "content": [{"type": "text", "text": "You are a helpful assistant."}]}, {"role": "user", "content": []}]
            for i in range(num_segments):
                conversation[1]["content"].append({"type": "text", "text": time_texts[i]})
                conversation[1]["content"].append({"type": "video", "video": all_videos[i]})
            chat_text = self.qa_model.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
            init_prompt = self.qa_model.processor.tokenizer.decode(self.qa_model.init_prompt_ids[0])
            chat_text = chat_text.replace(init_prompt, '')
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
            num_time_tokens = len(time_tokens) + 1 + 1
            dynamic_block_size += num_time_tokens
            if hasattr(self.qa_model, 'model') and hasattr(self.qa_model.model, 'layers'):
                for layer in self.qa_model.model.layers:
                    if hasattr(layer, 'self_attn'):
                        attn = layer.self_attn
                        setattr(attn, "dynamic_block_size", dynamic_block_size)
                        setattr(attn, "dynamic_exc_block_size", dynamic_block_size)
                        setattr(attn, "time_prompt_mode", True)
            elif hasattr(self.qa_model, 'language_model') and hasattr(self.qa_model.language_model.model, 'layers'):
                for layer in self.qa_model.language_model.model.layers:
                    if hasattr(layer, 'self_attn'):
                        attn = layer.self_attn
                        setattr(attn, "dynamic_block_size", dynamic_block_size)
                        setattr(attn, "dynamic_exc_block_size", dynamic_block_size)
                        setattr(attn, "time_prompt_mode", True)
            
            outputs = self.qa_model(
                **inputs,
                use_cache=True,
                return_dict=True,
                past_key_values=self.qa_model.kv_cache,
            )
            self.qa_model.kv_cache = outputs.past_key_values


class RecentLayeredRetrievalMixin:
    """Utilities shared by recent-layered retrieval variants."""
    def _get_question_last_hidden(self, first_pass_out):
        if not hasattr(first_pass_out, 'hidden_states') or not first_pass_out.hidden_states:
            return None
        last = first_pass_out.hidden_states[-1]
        if last.dim() == 3 and last.size(0) > 0:
            return last[0]
        if last.dim() == 2:
            return last
        return None

    def _build_present_word_prototypes(self, device):
        tokenizer = getattr(self.qa_model.processor, 'tokenizer', None)
        if tokenizer is None:
            return None
        # try:
        tok = tokenizer(
            present_words,
            add_special_tokens=False,
            padding=True,
            return_tensors='pt'
        )
        present_input_ids = tok['input_ids'].to(device)
        present_attn = tok['attention_mask'].to(device)
        if hasattr(self.qa_model, 'language_model'):
            pres_out = self.qa_model.language_model(
                input_ids=present_input_ids,
                attention_mask=present_attn,
                use_cache=True,
                output_hidden_states=True
            )
        else:
            pres_out = self.qa_model(
                input_ids=present_input_ids,
                attention_mask=present_attn,
                use_cache=True,
                output_hidden_states=True
            )
        pres_hidden_last = pres_out.hidden_states[-1]
        mask = present_attn.unsqueeze(-1).type_as(pres_hidden_last)
        summed = (pres_hidden_last * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1.0)
        prototypes = summed / lengths
        return prototypes
        # except Exception as e:
        #     logger.warning(f"Building present-word prototypes failed: {e}")
        #     return None

    def _compute_present_sigmoid_recent_prob(self, question_last_hidden, present_prototypes):
        try:
            if question_last_hidden is None or present_prototypes is None:
                return None, None
            H_q_n = F.normalize(question_last_hidden, dim=-1)
            P_pres_n = F.normalize(present_prototypes, dim=-1)

            S = torch.matmul(H_q_n, P_pres_n.t())
            s_present = S.max()
            scale = float(getattr(self, 'time_word_sigmoid_scale', 10.0))
            bias = float(getattr(self, 'time_word_sigmoid_bias', 0.3))
            p_recent = torch.sigmoid(scale * (s_present - bias))
            return float(s_present.item()), float(p_recent.item())
        except Exception as e:
            logger.warning(f"Present similarity computation failed: {e}")
            return None, None

    def _split_recent_past_counts(self, p_recent_value, target_retrieve_size):
        if p_recent_value is None:
            return None, None
        min_frac = float(getattr(self, 'min_recent_frac', 0.1))
        max_frac = float(getattr(self, 'max_recent_frac', 0.9))
        min_recent = int(round(min_frac * target_retrieve_size))
        max_recent = int(round(max_frac * target_retrieve_size))
        k_recent = int(round(p_recent_value * int(target_retrieve_size)))
        k_recent = max(min_recent, min(max_recent, k_recent))
        k_recent = max(0, min(int(target_retrieve_size), k_recent))
        k_past = int(target_retrieve_size) - k_recent
        return k_recent, k_past

    def _select_indices_recent_past(self, sorted_indices, k_recent, k_past):
        if k_recent is None or k_past is None:
            return None
        recent_indices = self.get_recent_frame_indices(retrieve_size=max(0, k_recent))
        recent_list = recent_indices[0] if recent_indices else []
        recent_set = set(recent_list)
        ranked_all = [idx for idx, _ in sorted_indices]
        past_candidates = [i for i in ranked_all if i not in recent_set]
        past_list = past_candidates[:max(0, k_past)]
        merged = list(dict.fromkeys(past_list + recent_list))
        merged.sort()
        return merged

    def _record_time_word_metrics(self, retrieval_info, s_present_value, p_recent_value, k_recent_value, k_past_value):
        retrieval_info['time_word'] = {
            's_present': s_present_value,
            'p_recent': p_recent_value,
            'k_recent': k_recent_value,
            'k_past': k_past_value,
        }
    def _load_layer_and_head_weights(self, layer_weight_path=None, head_weight_path=None):
        import os, json
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

    
    def get_time_word_future():
        pass
    
    
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
            if self.head_weights and layer_idx < len(self.head_weights):
                try:
                    w_list = self.head_weights[layer_idx]
                    w_tensor = torch.as_tensor(w_list, dtype=torch.float32, device=layer_kv.global_buffer[0].device if hasattr(layer_kv, 'global_buffer') else None)
                    if float(w_tensor.sum().item()) == 0.0:
                        w_tensor = torch.ones_like(w_tensor)
                    setattr(layer_kv, 'head_weights', w_tensor)
                except Exception as e:
                    logger.warning(f"Failed to set head weights for layer {layer_idx}: {e}")

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

        # get time word feature
        s_present_value = None
        p_recent_value = None
        k_recent_value = None
        k_past_value = None
        # if recent_mode:
        #     H_q = self._get_question_last_hidden(out)
        #     P_present = self._build_present_word_prototypes(H_q.device) if H_q is not None else None
        #     s_present_value, p_recent_value = self._compute_present_sigmoid_recent_prob(H_q, P_present)
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
        # Select frames using time-word driven recent/past split when in recent_mode
        if recent_mode and p_recent_value is not None:
            k_recent, k_past = self._split_recent_past_counts(p_recent_value, target_retrieve_size)
            k_recent_value = int(k_recent) if k_recent is not None else None
            k_past_value = int(k_past) if k_past is not None else None
            top_indices = self._select_indices_recent_past(sorted_indices, k_recent, k_past)
            if top_indices is None:
                top_indices = [idx for idx, votes in sorted_indices[:target_retrieve_size]]
                top_indices = sorted(list(dict.fromkeys(top_indices)))
        else:
            # Fallback: original behavior
            top_indices = [idx for idx, votes in sorted_indices[:target_retrieve_size]]
            if recent_mode:
                recent_indices = self.get_recent_frame_indices(retrieve_size=self.retrieve_size)
                top_indices.extend(recent_indices[0])
            top_indices = sorted(list(dict.fromkeys(top_indices)))

        if len(top_indices) < target_retrieve_size:
            total_frames = None
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
            start_idx = total_frames - (target_retrieve_size - len(top_indices))
            if start_idx < 0:
                start_idx = 0
            additional_indices = list(range(start_idx, total_frames))
            top_indices.extend([i for i in additional_indices if i not in top_indices])
            top_indices.sort()

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
        self._record_time_word_metrics(retrieval_info, s_present_value, p_recent_value, k_recent_value, k_past_value)

        for layer_kv in self.qa_model.kv_cache:
            layer_kv.reset_retrieval()

        qa_result = self._continue_question_answering(input_text, max_new_tokens, past_key_values, retrieval_info)
        return qa_result, retrieval_info
