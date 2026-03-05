import warnings
import random
import json
import os
import math
import argparse
import time
import torch.nn.functional as F
import numpy as np

import pandas as pd
import torch
from tqdm import tqdm
from decord import VideoReader, cpu
from transformers import (
    logging,
    LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor,
    VideoLlavaForConditionalGeneration, VideoLlavaProcessor
)
import logzero
from logzero import logger

# from model import llava_onevision_rekv, video_llava_rekv, longva_rekv


# MODELS = {
#     'llava_ov_0.5b': {
#         'load_func': llava_onevision_rekv.load_model,
#         'model_class': LlavaOnevisionForConditionalGeneration,
#         'processor_class': LlavaOnevisionProcessor,
#         'model_path': 'model_zoo/llava-onevision-qwen2-0.5b-ov-hf',
#     },
#     'llava_ov_7b': {
#         'load_func': llava_onevision_rekv.load_model,
#         'model_class': LlavaOnevisionForConditionalGeneration,
#         'processor_class': LlavaOnevisionProcessor,
#         'model_path': 'model_zoo/llava-onevision-qwen2-7b-ov-hf',
#     },
#     'llava_ov_72b': {
#         'load_func': llava_onevision_rekv.load_model,
#         'model_class': LlavaOnevisionForConditionalGeneration,
#         'processor_class': LlavaOnevisionProcessor,
#         'model_path': 'model_zoo/llava-onevision-qwen2-72b-ov-hf',
#     },
#     'video_llava_7b': {
#         'load_func': video_llava_rekv.load_model,
#         'model_class': VideoLlavaForConditionalGeneration,
#         'processor_class': VideoLlavaProcessor,
#         'model_path': 'model_zoo/Video-LLaVA-7B-hf',
#     },
#     'longva_7b': {
#         'load_func': longva_rekv.load_model,
#         'model_path': 'model_zoo/LongVA-7B',
#     },
# }


from model import llava_onevision_rekv
from model import qwen2vl_rekv
from video_qa.mixins import TimePromptMixin
from tools.frame_policies import (
    make_storage_strategy,
)


MODELS = {
    'llava_ov_0.5b': {
        'load_func': llava_onevision_rekv.load_model,
        'model_class': LlavaOnevisionForConditionalGeneration,
        'processor_class': LlavaOnevisionProcessor,
        'model_path': 'llava-hf/llava-onevision-qwen2-0.5b-ov-hf',
    },
    'llava_ov_7b': {
        'load_func': llava_onevision_rekv.load_model,
        'model_class': LlavaOnevisionForConditionalGeneration,
        'processor_class': LlavaOnevisionProcessor,
        'model_path': 'llava-hf/llava-onevision-qwen2-7b-ov-hf',
    },
    'llava_ov_72b': {
        'load_func': llava_onevision_rekv.load_model,
        'model_class': LlavaOnevisionForConditionalGeneration,
        'processor_class': LlavaOnevisionProcessor,
        'model_path': 'llava-hf/llava-onevision-qwen2-72b-ov-hf',
    },
    'qwen2vl_7b': {
        'load_func': qwen2vl_rekv.load_model,
        'model_class': None,
        'processor_class': None,
        'model_path': 'Qwen/Qwen2-VL-7B-Instruct',
    },
}

class BaseVQA:
    def __init__(self, anno, save_dir, sample_fps,
                 qa_model, qa_processor=None,
                 num_chunks=None, chunk_idx=None,
                 retrieve_size=64, chunk_size=1, query_type='question', use_dynamic_size=False,
                 prompt_builder_type=None, prompt_builder_kwargs=None,
                 entropy_threshold=None, entropy_window_layers=None,
                 precomputed_retrieval_file=None, **kwargs) -> None:
        
        self.sample_fps = sample_fps

        self.qa_model = qa_model
        self.qa_processor = qa_processor
        self.query_type = query_type
        self.use_dynamic_size = use_dynamic_size

        # Retrieval Hyperparams
        assert chunk_size <= retrieve_size, f'chunk_size: {chunk_size}, retrieve_size: {retrieve_size}'
        self.retrieve_size = retrieve_size
        self.chunk_size = chunk_size

        self.num_chunks = num_chunks
        self.chunk_idx = chunk_idx
        if num_chunks is not None:
            anno = self.get_chunk(anno, num_chunks, chunk_idx)
        self.anno = anno
        self.eval_grounding = 'temporal_windows' in anno[0]['conversations'][0]

        self.save_dir = save_dir
        self.choice_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        self.record = {(self.retrieve_size, self.chunk_size): []}
        
        # Setup prompt builder if specified
        if prompt_builder_type is not None:
            from video_qa.prompt_builders import get_prompt_builder
            builder_kwargs = prompt_builder_kwargs or {}
            # Add choice_letters to builder kwargs
            if 'choice_letters' not in builder_kwargs:
                builder_kwargs['choice_letters'] = self.choice_letters
            self.prompt_builder = get_prompt_builder(prompt_builder_type, **builder_kwargs)
            logger.info(f"Using prompt builder: {prompt_builder_type} with kwargs: {builder_kwargs}")
        else:
            self.prompt_builder = None

        if entropy_threshold is not None:
            self.entropy_threshold = entropy_threshold
        if entropy_window_layers is not None:
            self.entropy_window_layers = entropy_window_layers
        print(f"entropy_threshold: {self.entropy_threshold}, entropy_window_layers: {self.entropy_window_layers}")
        
        # Load precomputed retrieval results if provided
        self.precomputed_retrieval_map = None
        if precomputed_retrieval_file is not None:
            self.precomputed_retrieval_map = self._load_precomputed_retrieval(precomputed_retrieval_file)
            logger.info(f"Loaded precomputed retrieval results from {precomputed_retrieval_file}, {len(self.precomputed_retrieval_map)} cases")

    def split_list(self, lst, n):
        """Split a list into n (roughly) equal-sized chunks"""
        chunk_size = math.ceil(len(lst) / n)  # integer division
        return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]

    def get_chunk(self, lst, n, k):
        chunks = self.split_list(lst, n)
        return chunks[k]

    def load_video(self, video_path, max_frame_num=None):
        vr = VideoReader(video_path, ctx=cpu(0))
        fps = round(vr.get_avg_fps())
        frame_idx = [i for i in range(0, len(vr), int(fps / self.sample_fps))]
        if max_frame_num is not None and len(frame_idx) > max_frame_num:
            uniform_sampled_frames = np.linspace(
                0, len(vr) - 1, max_frame_num, dtype=int
            )
            frame_idx = uniform_sampled_frames.tolist()
        video = vr.get_batch(frame_idx).asnumpy()
        logger.debug(f'video shape: {video.shape}')
        return video
    
    def calc_recall_precision(self, gt_temporal_windows, retrieved_mask):
        total_intersection_length = 0.0
    
        for (start_sec, end_sec) in gt_temporal_windows:
            start = math.floor(start_sec)
            end = math.ceil(end_sec)
            for i in range(start, end):
                if i < len(retrieved_mask) and retrieved_mask[i]:
                    intersection_start = max(start_sec, i)
                    intersection_end = min(end_sec, i + 1)
                    total_intersection_length += intersection_end - intersection_start

        gt_len = sum([end_sec - start_sec for start_sec, end_sec in gt_temporal_windows])
        retrieved_len = sum(retrieved_mask).item()

        recall = total_intersection_length / gt_len if gt_len > 0 else 0
        precision = total_intersection_length / retrieved_len if retrieved_len > 0 else 0
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
        return recall, precision, f1
    
    def format_mcqa_prompt(self, question, candidates, **kwargs):
        """
        Format multiple-choice QA prompt.
        
        Args:
            question: The question string
            candidates: List of candidate answers
            **kwargs: Additional metadata (e.g., task, start_time, end_time) for custom prompt builders
            
        Returns:
            dict with formatted prompt information
        """
        assert len(question) > 0, f"Q: {question}"

        # Use custom prompt builder if available, otherwise use default
        if hasattr(self, 'prompt_builder') and self.prompt_builder is not None:
            prompt_dict = self.prompt_builder.build_prompt(question, candidates, **kwargs)
        else:
            # Default behavior (backward compatible)
            formatted_choices = "\n".join(["(" + self.choice_letters[i] + ") " + candidate for i, candidate in enumerate(candidates)])
            formatted_question = f"Question: {question}\nOptions:\n{formatted_choices}\nOnly give the best option."
            prompt_dict = {
                "question": f"{question}",
                "formatted_question": formatted_question,
            }
        
        # Set the final prompt using qa_model.get_prompt()
        if prompt_dict.get("prompt") is None:
            prompt_dict["prompt"] = self.qa_model.get_prompt(prompt_dict["formatted_question"], mc=True)
        
        return prompt_dict

    def extract_characters_regex(self, s):
        s = s.strip()
        if ")" in s:
            index = s.index(")")
            pred = s[index - 1 : index]
            return pred
        else:
            try:
                return s[0]
            except:
                return random.choice(self.choice_letters)

    def capture_retrieval_info_and_qa(self, input_text, max_new_tokens=128, retrieved_indices=None):
        """
        调用question_answering并捕获retrieval信息
        
        Args:
            input_text: 输入文本
            max_new_tokens: 最大新token数
            retrieved_indices: 预设的检索索引
            
        Returns:
            tuple: (qa_result, retrieval_info)
        """
        # 在调用question_answering之前，准备捕获retrieval信息
        device = self.qa_model.device
        
        # 获取问题的input_ids并设置retrieval模式
        query_text = input_text[self.query_type]
        if hasattr(self.qa_model, 'processor') and hasattr(self.qa_model.processor, 'tokenizer'):
            input_ids = self.qa_model.processor.tokenizer(query_text).input_ids
        else:
            # 兼容其他tokenizer
            input_ids = query_text
            
        input_ids = torch.as_tensor([input_ids], device=device)
        
        # 激活retrieval模式
        for layer_kv in self.qa_model.kv_cache:
            layer_kv.set_retrieval()
        
        # 执行retrieval步骤
        if retrieved_indices is None:  # Internal retrieval
            if hasattr(self.qa_model, 'language_model'):
                out = self.qa_model.language_model(input_ids=input_ids, use_cache=True, past_key_values=self.qa_model.kv_cache)
            else:
                out = self.qa_model(input_ids=input_ids, use_cache=True, past_key_values=self.qa_model.kv_cache)
            # breakpoint()
            past_key_values = out.past_key_values
        else:  # External retrieval
            for layer_kv in self.qa_model.kv_cache:
                layer_kv.set_retrieved_block_indices(retrieved_indices)
            if hasattr(self.qa_model, 'language_model'):
                out = self.qa_model.language_model(input_ids=input_ids, use_cache=True, past_key_values=self.qa_model.kv_cache)
            else:
                out = self.qa_model(input_ids=input_ids, use_cache=True, past_key_values=self.qa_model.kv_cache)
            past_key_values = out.past_key_values
        
        # 在reset之前捕获retrieval信息
        retrieval_info = self._extract_retrieval_info()
        
        # 重置retrieval模式
        for layer_kv in self.qa_model.kv_cache:
            layer_kv.reset_retrieval()

        # 继续执行question_answering的其余部分（生成答案）
        qa_result = self._continue_question_answering(input_text, max_new_tokens, past_key_values)
        
        # 每轮QA结束，关闭激活的base（恢复常规KV增长模式）
        if hasattr(self.qa_model, 'kv_cache') and self.qa_model.kv_cache is not None:
            for layer_kv in self.qa_model.kv_cache:
                if hasattr(layer_kv, 'deactivate_base'):
                    layer_kv.deactivate_base()
                    
        return qa_result, retrieval_info
    
    def _extract_retrieval_info(self):
        """
        从kv_cache中提取retrieval信息
        """
        retrieval_records = {}
        similarity_records = {}
        return {
            'retrieval_records': retrieval_records,
            'similarity_scores': similarity_records
        }
        if not hasattr(self.qa_model, 'kv_cache') or self.qa_model.kv_cache is None:
            return {'retrieval_records': retrieval_records, 'similarity_scores': similarity_records}
        
        # 获取所有层的retrieved_block_indices
        for layer_idx, layer_kv in enumerate(self.qa_model.kv_cache):
            layer_name = f'layer_{layer_idx}'
            
            # 获取retrieved_block_indices
            if hasattr(layer_kv, 'retrieved_block_indices') and layer_kv.retrieved_block_indices is not None:
                retrieved_indices = layer_kv.retrieved_block_indices
                
                # 转换为时间戳
                if isinstance(retrieved_indices, torch.Tensor):
                    retrieved_indices = retrieved_indices.cpu().tolist()
                
                # 计算时间戳
                timestamps = []
                if isinstance(retrieved_indices, list):
                    for batch_idx, batch_indices in enumerate(retrieved_indices):
                        batch_timestamps = []
                        if isinstance(batch_indices, list):
                            for frame_idx in batch_indices:
                                # 根据sample_fps转换frame索引为时间戳
                                timestamp = frame_idx / self.sample_fps
                                batch_timestamps.append(timestamp)
                        else:
                            # 单个索引
                            timestamp = batch_indices / self.sample_fps
                            batch_timestamps.append(timestamp)
                        timestamps.append(batch_timestamps)
                else:
                    # 处理单个batch的情况
                    for frame_idx in retrieved_indices:
                        timestamp = frame_idx / self.sample_fps
                        timestamps.append(timestamp)
                
                retrieval_records[layer_name] = {
                    'retrieved_indices': retrieved_indices,
                    'timestamps': timestamps
                }
            
            # 获取similarity分数
            if hasattr(layer_kv, 'similarity') and layer_kv.similarity is not None:
                similarity = layer_kv.similarity
                if isinstance(similarity, torch.Tensor):
                    similarity = similarity.cpu().tolist()
                similarity_records[layer_name] = similarity
        
        return {
            'retrieval_records': retrieval_records,
            # 'similarity_scores': similarity_records
        }
    
    def _load_precomputed_retrieval(self, filepath):
        """Load precomputed retrieval results from CSV file."""
        import ast
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Precomputed retrieval file not found: {filepath}")
        
        logger.info(f"Loading precomputed retrieval results from {filepath}")
        df = pd.read_csv(filepath)
        
        retrieval_map = {}
        num_valid = 0
        
        for idx, row in df.iterrows():
            video_id = row['video_id']
            question = row['question']
            
            choices = None
            if 'choices' in row and pd.notna(row['choices']):
                try:
                    choices = ast.literal_eval(row['choices'])
                    if isinstance(choices, list):
                        choices = tuple(choices)
                except:
                    choices = None
            
            if 'retrieval_info' not in row or pd.isna(row['retrieval_info']):
                continue
            
            try:
                retrieval_info = ast.literal_eval(row['retrieval_info'])
            except Exception as e:
                logger.warning(f"Failed to parse retrieval_info for row {idx}: {e}")
                continue
            
            if 'retrieval_records' not in retrieval_info:
                continue
            
            retrieval_records = retrieval_info['retrieval_records']
            
            # Debug: check why retrieval_records might be empty
            if not retrieval_records or len(retrieval_records) == 0:
                logger.warning(f"Empty retrieval_records for video_id={video_id}, question='{question}'")
                logger.warning(f"  Full retrieval_info keys: {list(retrieval_info.keys())}")
                logger.warning(f"  Strategy: {retrieval_info.get('strategy', 'N/A')}")
                logger.warning(f"  retrieval_records content: {retrieval_records}")
                raise ValueError(
                    f"Empty retrieval_records for video_id={video_id}, question='{question}'. "
                    f"Cannot use precomputed retrieval without valid retrieval data. "
                    f"The CSV file appears to contain cases with strategy='{retrieval_info.get('strategy')}' "
                    f"which did not perform retrieval. Please use a CSV file with actual retrieval results."
                )
            
            lookup_key = (video_id, question, choices)
            retrieval_map[lookup_key] = retrieval_records
            num_valid += 1
        
        logger.info(f"Loaded {num_valid} valid retrieval records")
        return retrieval_map
    
    def _convert_retrieval_records_to_indices(self, retrieval_records):
        """Convert retrieval_records to per-layer indices format."""
        if not retrieval_records:
            return None
        
        indices_per_layer = {}
        for layer_name, record in retrieval_records.items():
            if isinstance(layer_name, str) and layer_name.startswith('layer_'):
                try:
                    layer_idx = int(layer_name.split('_')[1])
                except:
                    continue
            else:
                continue
            
            if 'retrieved_indices' in record:
                indices = record['retrieved_indices']
                indices_per_layer[layer_idx] = indices
        
        return indices_per_layer if indices_per_layer else None
    
    def _continue_question_answering(self, input_text, max_new_tokens, past_key_values, retrieval_info=None):
        # """
        # 继续执行question_answering的生成部分
        # """
        device = self.qa_model.device
        stop_token_ids = [self.qa_model.processor.tokenizer.eos_token_id]
        
        output_ids = []
        stopped = False

        # 可选调试：打印底座/追加与可用长度（需设置 REKV_DEBUG=1）
        if os.environ.get('REKV_DEBUG', '0') == '1':
            try:
                past_len = past_key_values.get_seq_length() if hasattr(past_key_values, 'get_seq_length') else 'NA'
                usable_len_l0 = past_key_values.get_usable_length(1, 0) if hasattr(past_key_values, 'get_usable_length') else 'NA'
                base_len = app_len = 'NA'
                if hasattr(self.qa_model, 'kv_cache') and self.qa_model.kv_cache is not None:
                    try:
                        cm0 = self.qa_model.kv_cache[0]
                        base_len = cm0.active_base_k.size(-2) if hasattr(cm0, 'active_base_k') and cm0.active_base_k is not None else 0
                        app_len = cm0.appended_k.size(-2) if hasattr(cm0, 'appended_k') and cm0.appended_k is not None else 0
                    except Exception:
                        pass
                print(f"[ReKV][gen] past_len={past_len} usable_len_l0={usable_len_l0} base_len={base_len} app_len={app_len}")
            except Exception:
                pass
        
        for i in range(max_new_tokens):
            if i == 0:  # prefill
                input_ids = self.qa_model.processor.tokenizer(input_text['prompt']).input_ids
                input_ids = torch.as_tensor([input_ids], device=device)
                inputs_embeds = self.qa_model.get_input_embeddings()(input_ids)
                if hasattr(self.qa_model, 'language_model'):
                    out = self.qa_model.language_model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=past_key_values, output_hidden_states=False)
                else:
                    out = self.qa_model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=past_key_values, output_hidden_states=False)
                past_key_values = out.past_key_values
                # breakpoint()
                logits = out.logits
                
                 # 为每一层计算entropy
                
            else:  # decoding
                if hasattr(self.qa_model, 'language_model'):
                    out = self.qa_model.language_model(
                    input_ids=torch.as_tensor(
                        [[token]],
                        device=device,
                    ),
                    use_cache=True,
                    past_key_values=past_key_values,
                    )
                else:
                    out = self.qa_model(
                        input_ids=torch.as_tensor(
                            [[token]],
                            device=device,
                        ),
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
            
            if token in stop_token_ids:
                stopped = True
            else:
                stopped = False
            
            if i == max_new_tokens - 1 or stopped:
                break
        
        # 说明：避免以零填充对齐 past_length 的做法；仅使用真实提示词与 past_key_values 进行生成。
        output = self.qa_model.processor.tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )
        
        return output

    def video_open_qa(self, question, max_new_tokens=1024, retrieved_indices=None):
        """
        Perform open-ended question answering.
        
        Args:
            question: The question string
            max_new_tokens: Maximum number of tokens to generate
            retrieved_indices: Optional pre-specified retrieval indices
            
        Returns:
            dict: Contains 'pred_answer' and 'retrieval_info'
        """
        input_text = {
            "question": question,
            "prompt": self.qa_model.get_prompt(question)
        }
        t0 = time.time()
        # Check if layered retrieval should be used
        if hasattr(self, 'short_memory_layers') and self.short_memory_layers:
            # Use layered retrieval if available
            if hasattr(self, 'question_answering_with_layered_retrieval'):
                pred_answer, retrieval_info = self.question_answering_with_layered_retrieval(
                    input_text, max_new_tokens
                )
                logger.debug(f"使用分层检索策略，短期记忆层: {self.short_memory_layers}")
            else:
                # Fallback to standard retrieval

                pred_answer, retrieval_info = self.capture_retrieval_info_and_qa(
                    input_text, max_new_tokens=max_new_tokens, retrieved_indices=retrieved_indices
                )
        else:
            pred_answer, retrieval_info = self.capture_retrieval_info_and_qa(
                input_text, max_new_tokens=max_new_tokens, retrieved_indices=retrieved_indices
            )

        elapsed = float(time.time() - t0)
        try:
            fps_or_none = getattr(self, '_last_processing_fps', None)
            if fps_or_none is None:
                seg_stats = getattr(self, '_last_segment_encode_stats', None)
                fps_or_none = None if seg_stats is None else seg_stats.get('fps', None)
        except Exception:
            fps_or_none = None
        retrieval_info['response_time_s'] = elapsed
        retrieval_info['processing_fps'] = fps_or_none
        return {
            'pred_answer': pred_answer.replace('\n', ''),
            'retrieval_info': retrieval_info
        }

    def video_close_qa(self, question, candidates, correct_choice, retrieved_indices=None, **prompt_kwargs):
        """
        Perform multiple-choice question answering.
        
        Args:
            question: The question string
            candidates: List of candidate answers
            correct_choice: The correct choice letter (A, B, C, etc.)
            retrieved_indices: Optional pre-specified retrieval indices
            **prompt_kwargs: Additional metadata for prompt builder (e.g., task, start_time, end_time)
            
        Returns:
            dict: Contains 'pred_answer', 'pred_choice', 'acc', and 'retrieval_info'
        """
        # Check for precomputed retrieval results
        if retrieved_indices is None and hasattr(self, 'precomputed_retrieval_map') and self.precomputed_retrieval_map is not None:
            video_id = getattr(self, 'current_video_id', None)
            if video_id is not None:
                lookup_key = (video_id, question, tuple(candidates))
                if lookup_key in self.precomputed_retrieval_map:
                    retrieval_records = self.precomputed_retrieval_map[lookup_key]
                    retrieved_indices = self._convert_retrieval_records_to_indices(retrieval_records)
                    logger.info(f"Using precomputed retrieval for video {video_id}, question: {question[:50]}...")
        
        input_text = self.format_mcqa_prompt(question, candidates, **prompt_kwargs)
        t0 = time.time()
        # Check if layered retrieval should be used
        if hasattr(self, 'short_memory_layers') and self.short_memory_layers:
            # Use layered retrieval if available
            if hasattr(self, 'question_answering_with_layered_retrieval'):
                pred_answer, retrieval_info = self.question_answering_with_layered_retrieval(
                    input_text, max_new_tokens=16
                )
            else:
                # Fallback to standard retrieval
                if retrieved_indices is None:
                    if hasattr(self, 'get_recent_frame_indices'):
                        retrieved_indices = self.get_recent_frame_indices(retrieve_size=self.retrieve_size)
                    else:
                        retrieved_indices = None
                pred_answer, retrieval_info = self.capture_retrieval_info_and_qa(
                    input_text, max_new_tokens=16, retrieved_indices=retrieved_indices
                )
        else:
            # Standard retrieval with recent frames

            # Use base class method to capture retrieval info
            pred_answer, retrieval_info = self.capture_retrieval_info_and_qa(
                input_text, max_new_tokens=16, retrieved_indices=retrieved_indices
            )
        elapsed = float(time.time() - t0)
        try:
            fps_or_none = getattr(self, '_last_processing_fps', None)
            if fps_or_none is None:
                seg_stats = getattr(self, '_last_segment_encode_stats', None)
                fps_or_none = None if seg_stats is None else seg_stats.get('fps', None)
        except Exception:
            fps_or_none = None
        retrieval_info['response_time_s'] = elapsed
        retrieval_info['processing_fps'] = fps_or_none
        
        # Normalize prediction according to candidate types
        all_yes_no = all(c.lower() in ['yes', 'no'] for c in candidates)
        all_numbers = all(c.strip().isdigit() for c in candidates)

        if all_yes_no:
            pred_choice = pred_answer.strip().lower()
        elif all_numbers:
            pred_choice = pred_answer.strip()
        else:
            pred_choice = self.extract_characters_regex(pred_answer)
        
        return {
            'pred_answer': pred_answer.replace('\n', ''),
            'pred_choice': pred_choice,
            'acc': float(pred_choice == correct_choice),
            'retrieval_info': retrieval_info
        }
    
    def _record_qa_result(self, video_sample, sample, qa_results, is_close_qa=False):
        """
        Unified method to record QA results.
        
        Args:
            video_sample: The video sample dict
            sample: The conversation sample dict
            qa_results: Results dict from video_open_qa or video_close_qa
            is_close_qa: Whether this is a close-ended QA (multiple choice)
        """
        retrieval_info = qa_results.get('retrieval_info', {})
        retrieval_records = retrieval_info.get('retrieval_records', {})
        similarity_scores = retrieval_info.get('similarity_scores', {})
        layer_entropies = retrieval_info.get('layer_entropies', {})
        
        base_entry = {
            'video_id': video_sample['video_id'],
            'question': sample['question'],
            'answer': sample['answer'],
            'pred_answer': qa_results['pred_answer'],
            'retrieval_records': retrieval_records,
            'similarity_scores': similarity_scores,
        }
        
        # Add layer entropies if available
        if layer_entropies:
            base_entry['layer_entropies'] = layer_entropies
        
        # Add dynamic retrieve size if available
        if 'dynamic_retrieve_size' in retrieval_info:
            base_entry['dynamic_retrieve_size'] = retrieval_info.get('dynamic_retrieve_size', self.retrieve_size)
        
        # Add retrieval_info if available (for compatibility)
        if retrieval_info:
            base_entry['retrieval_info'] = retrieval_info
        
        if is_close_qa:
            # Determine correct_choice if not already in qa_results
            correct_choice = None
            if sample.get('answer') and 'choices' in sample:
                choices = sample['choices']
                all_yes_no = all(c.lower() in ['yes', 'no'] for c in choices)
                all_numbers = all(c.strip().isdigit() for c in choices)
                if all_yes_no or all_numbers:
                    correct_choice = sample['answer']
                else:
                    try:
                        correct_choice = self.choice_letters[choices.index(sample['answer'])]
                    except (ValueError, IndexError):
                        correct_choice = None
            
            base_entry.update({
                'choices': sample.get('choices', []),
                'correct_choice': qa_results.get('correct_choice', correct_choice),
                'pred_choice': qa_results.get('pred_choice', ''),
                'qa_acc': qa_results.get('acc', 0.0) * 100,
            })
        else:
            # Ensure all records have the same structure, even for open_qa
            # This prevents column misalignment when creating DataFrame
            base_entry.update({
                'choices': None,
                'correct_choice': None,
                'pred_choice': None,
                'qa_acc': None,
            })
        
        self.record[(self.retrieve_size, self.chunk_size)].append(base_entry)
        
        # Add task type if available
        if 'question_type' in sample:
            self.record[(self.retrieve_size, self.chunk_size)][-1]['task'] = sample['question_type']
        elif 'task' in sample:
            self.record[(self.retrieve_size, self.chunk_size)][-1]['task'] = sample['task']

    def analyze(self, debug=False):
        video_annos = self.anno[:1] if debug else self.anno
        for video_sample in tqdm(video_annos):
            logger.debug(f'video_id: {video_sample["video_id"]}')
            self.analyze_a_video(video_sample)

        dfs = []
        for (retrieve_size, chunk_size), dict_list in self.record.items():
            df = pd.DataFrame(dict_list)
            df['retrieve_size'] = retrieve_size
            df['chunk_size'] = chunk_size
            dfs.append(df)
        final_df = pd.concat(dfs, ignore_index=True)
        final_df.to_csv(f'{self.save_dir}/{self.num_chunks}_{self.chunk_idx}.csv', index=False)

import math
def ceil_time_by_fps(time: float, fps: int, min_time: float, max_time: float):
    return min(max(math.ceil(time * fps) / fps, min_time), max_time)

def floor_time_by_fps(time: float, fps: int, min_time: float, max_time: float):
    return min(max(math.floor(time * fps) / fps, min_time), max_time)

def window_to_frame_indices(start_time: float, end_time: float, sample_fps: float, video_len: int):
    """
    Convert a time window [start_time, end_time) in seconds into a list of frame indices
    according to sample_fps, clamped to [0, video_len].
    """
    start_idx = int(math.floor(max(0.0, start_time) * sample_fps))
    end_idx = int(math.ceil(max(0.0, end_time) * sample_fps))
    start_idx = max(0, min(start_idx, video_len))
    end_idx = max(0, min(end_idx, video_len))
    if end_idx <= start_idx:
        end_idx = min(start_idx + 1, video_len)
    return list(range(start_idx, end_idx))


class StreamVideoEncodingMixin:
    """Mixin for streaming video encoding (progressive encoding based on temporal windows)."""
    
    @torch.inference_mode()
    def analyze_a_video(self, video_sample):
        """
        Analyze a video sample with streaming encoding.
        Video is encoded progressively based on temporal windows from conversations.
        """
        video_path = video_sample['video_path']
        video_start_idx = video_end_idx = 0
        
        try:
            video = self.load_video(video_path)
        except Exception as e:
            logger.error(f"Error loading video: {e}")
            return
        if not isinstance(video, torch.Tensor):
            video_tensor = torch.from_numpy(video)
        else:
            video_tensor = video

        self.qa_model.clear_cache()
        self.qa_model.encode_init_prompt()

        # Use video_id to ensure fixed resolution
        video_id = video_sample.get('video_id', video_sample.get('video_path', 'unknown'))
        
        # Set current video_id for precomputed retrieval lookup
        self.current_video_id = video_id

        for sample in video_sample['conversations']:
            logger.debug(f'sample: {sample}')
            
            # # Get temporal windows (start_time, end_time in seconds)
            # temporal_windows = torch.tensor([sample['start_time'], sample['end_time']]) * self.sample_fps
            # temporal_windows = temporal_windows.tolist()
    
            # # Encode video up to the end of the temporal window
            # if temporal_windows[-1] > video_end_idx:
            #     video_end_idx = temporal_windows[-1]
            #     if int(video_end_idx) == 0:  # fix some case can not sample frame in low fps
            #         video_end_idx = temporal_windows[-1] + 1
                
            # get clip start and end index
            start_time_idx = floor_time_by_fps(sample['start_time'], self.sample_fps, 0, 9999999) * self.sample_fps
            end_time_idx = min(ceil_time_by_fps(sample['end_time'], self.sample_fps, 0, 9999) * self.sample_fps, video_tensor.shape[0] - 1) * self.sample_fps
            start_time_idx = int(start_time_idx)
            end_time_idx = max(int(end_time_idx), 1)
            
            if end_time_idx > video_end_idx:
                video_end_idx = end_time_idx
                    
                # Encode video segment
                segment_frames = video_tensor[int(video_start_idx):int(video_end_idx)]
                
                # Check if time prompt encoding should be used
                if hasattr(self, 'encode_video_with_time_prompts') and len(segment_frames) > 0:
                    if getattr(self, 'convert_to_streaming', 'false') == 'true':
                        encode_chunk_size = getattr(self, 'retrieve_size', 64) // 16
                    else:
                        encode_chunk_size = getattr(self, 'retrieve_size', 64)  // 16
                    _t0 = time.time()
                    self.encode_video_with_time_prompts(segment_frames, video_id=video_id, encode_chunk_size=encode_chunk_size)
                    _elapsed = time.time() - _t0
                    _frames_n = int(len(segment_frames))
                    _fps = (_frames_n / _elapsed) if (_frames_n > 0 and _elapsed > 0) else None
                    self._last_segment_encode_stats = {
                        'frames': _frames_n,
                        'elapsed': float(_elapsed),
                        'fps': float(_fps) if _fps is not None else None,
                    }
                    self._last_processing_fps = float(_fps) if _fps is not None else None
                else:
                    if len(segment_frames) > 0:
                        if getattr(self, 'convert_to_streaming', 'false') == 'true':
                            encode_chunk_size = getattr(self, 'retrieve_size', 64) // 16
                        else:
                            encode_chunk_size = getattr(self, 'retrieve_size', 64) // 16
                        _t0 = time.time()
                        self.qa_model.encode_video(segment_frames, video_id=video_id, encode_chunk_size=encode_chunk_size)
                        _elapsed = time.time() - _t0
                        _frames_n = int(len(segment_frames))
                        _fps = (_frames_n / _elapsed) if (_frames_n > 0 and _elapsed > 0) else None
                        self._last_segment_encode_stats = {
                            'frames': _frames_n,
                            'elapsed': float(_elapsed),
                            'fps': float(_fps) if _fps is not None else None,
                        }
                        self._last_processing_fps = float(_fps) if _fps is not None else None
                
                video_start_idx = video_end_idx

            # Perform QA
            self._process_qa_sample(video_sample, sample)
    
    def _process_qa_sample(self, video_sample, sample):
        """
        Process a single QA sample. Can be overridden by subclasses.
        Default implementation handles both CloseQA and OpenQA.
        """
        question = sample['question']
        answer = sample['answer']
        
        if 'choices' in sample:  # CloseQA
            choices = sample['choices']
            if answer is None:  # FIXME: an ugly fix for some benchmarks do not provide GT
                answer = choices[0]
            # Determine correct choice format
            all_yes_no = all(c.lower() in ['yes', 'no'] for c in choices)
            all_numbers = all(c.strip().isdigit() for c in choices)
            if all_yes_no or all_numbers:
                correct_choice = str(answer).strip().lower()
            else:
                correct_choice = self.choice_letters[choices.index(answer)]
            
            # Extract metadata for prompt builder
            prompt_kwargs = {}
            qa_results = self.video_close_qa(question, choices, correct_choice, **prompt_kwargs)
            self._record_qa_result(video_sample, sample, qa_results, is_close_qa=True)
        else:  # OpenQA
            qa_results = self.video_open_qa(question)
            self._record_qa_result(video_sample, sample, qa_results, is_close_qa=False)


class OfflineVideoEncodingMixin:
    """Mixin for offline video encoding (encode entire video at once)."""
    
    @torch.inference_mode()
    def analyze_a_video(self, video_sample):
        """
        Analyze a video sample with offline encoding.
        Entire video is encoded once before processing questions.
        """
        video_path = video_sample['video_path']
        
        try:
            video = self.load_video(video_path)
        except Exception as e:
            logger.error(f"Error loading video: {e}")
            return

        if not isinstance(video, torch.Tensor):
            video_tensor = torch.from_numpy(video)
        else:
            video_tensor = video

        self.qa_model.clear_cache()
        self.qa_model.encode_init_prompt()
        
        # Use video path as video_id to ensure fixed resolution
        video_id = video_sample.get('video_id', video_path)
        
        # Set current video_id for precomputed retrieval lookup
        self.current_video_id = video_id
        
        # Encode entire video at once
        if hasattr(self, 'encode_video_with_time_prompts'):
            if getattr(self, 'convert_to_streaming', 'false') == 'true':
                encode_chunk_size = getattr(self, 'retrieve_size', 64) // 16
            else:
                encode_chunk_size = getattr(self, 'retrieve_size', 64) // 16
            _t0 = time.time()
            self.encode_video_with_time_prompts(video_tensor, video_id=video_id, encode_chunk_size=encode_chunk_size)
            _elapsed = time.time() - _t0
            _frames_n = int(video_tensor.shape[0])
            _fps = (_frames_n / _elapsed) if (_frames_n > 0 and _elapsed > 0) else None
            self._last_segment_encode_stats = {
                'frames': _frames_n,
                'elapsed': float(_elapsed),
                'fps': float(_fps) if _fps is not None else None,
            }
            self._last_processing_fps = float(_fps) if _fps is not None else None
        else:
            if getattr(self, 'convert_to_streaming', 'false') == 'true':
                encode_chunk_size = getattr(self, 'retrieve_size', 64) // 16
            else:
                encode_chunk_size = getattr(self, 'retrieve_size', 64)  // 16
            _t0 = time.time()
            self.qa_model.encode_video(video_tensor, video_id=video_id, encode_chunk_size=encode_chunk_size)
            _elapsed = time.time() - _t0
            _frames_n = int(video_tensor.shape[0])
            _fps = (_frames_n / _elapsed) if (_frames_n > 0 and _elapsed > 0) else None
            self._last_segment_encode_stats = {
                'frames': _frames_n,
                'elapsed': float(_elapsed),
                'fps': float(_fps) if _fps is not None else None,
            }
            self._last_processing_fps = float(_fps) if _fps is not None else None

        # Process all questions
        for sample in video_sample['conversations']:
            logger.debug(f'sample: {sample}')
            self._process_qa_sample(video_sample, sample)
    
    def _process_qa_sample(self, video_sample, sample):
        """
        Process a single QA sample. Can be overridden by subclasses.
        Default implementation handles both CloseQA and OpenQA.
        """
        question = sample['question']
        answer = sample['answer']
        
        if 'choices' in sample:  # CloseQA
            choices = sample['choices']
            if answer is None:  # FIXME: an ugly fix for some benchmarks do not provide GT
                answer = choices[0]
            # Determine correct choice format
            all_yes_no = all(c.lower() in ['yes', 'no'] for c in choices)
            all_numbers = all(c.strip().isdigit() for c in choices)
            if all_yes_no or all_numbers:
                correct_choice = str(answer).strip().lower()
            else:
                correct_choice = self.choice_letters[choices.index(answer)]
            
            # Extract metadata for prompt builder
            prompt_kwargs = {}
            qa_results = self.video_close_qa(question, choices, correct_choice, **prompt_kwargs)
            self._record_qa_result(video_sample, sample, qa_results, is_close_qa=True)
        else:  # OpenQA
            qa_results = self.video_open_qa(question)
            self._record_qa_result(video_sample, sample, qa_results, is_close_qa=False)


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', '1', 'yes'):
        return True
    elif value.lower() in ('false', '0', 'no'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def work(QA_CLASS):
    logging.set_verbosity_error()

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_fps", type=float, default=1)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--anno_path", type=str, required=True)
    parser.add_argument("--model", type=str, default="llava_ov_7b")
    parser.add_argument("--model_path", type=str, default=None, 
                        help="自定义模型权重路径，如果提供则覆盖默认路径")
    parser.add_argument("--n_local", type=int, default=15000)
    parser.add_argument("--retrieve_size", type=int, default=64)
    parser.add_argument("--retrieve_chunk_size", type=int, default=1)
    parser.add_argument("--debug", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--short_memory_layers", type=int, nargs='*', default=None, 
                        help="使用短期记忆（最近帧）的层索引列表")
    parser.add_argument("--layer_weight_path", type=str, default=None)
    parser.add_argument("--head_weight_path", type=str, default=None)
    parser.add_argument("--use_hybrid_similarity", type=str2bool, nargs='?', const=True, default=True,
                        help="是否启用混合similarity (true/false)，否则使用avg similarity")
    parser.add_argument("--query_type", type=str, default='question', choices=['question', 'prompt'], 
                        help="指定查询类型：question或prompt")
    parser.add_argument("--use_dynamic_size", type=str2bool, nargs='?', const=True, default=False,
                        help="是否启用动态检索大小 (true/false)")
    parser.add_argument("--merge_load_kv", type=str2bool, nargs='?', const=True, default=False,
                        help="启用合并式KV加载：并入票数阈值相同的帧并合并相邻高相似度帧")
    parser.add_argument("--time_prompt", action="store_true", default=False,
                        help="启用时间提示模式（仅部分模型/任务支持）")
    # Time-prompt support (plumbed through optional_map if QA_CLASS accepts it)
    parser.add_argument("--input_fps", type=float, default=None,
                        help="时间提示输入频率(Hz)，必须 <= sample_fps；仅当目标类支持时生效")
    parser.add_argument("--entropy_threshold", type=float, default=None,
                        help="覆盖熵自适应检索阈值 (默认 0.6)")
    parser.add_argument("--entropy_window_layers", type=int, default=None,
                        help="熵平均时使用的最近层数 (默认 1)")
    # 可选：为特定QA_CLASS透传的参数（若其__init__包含）
    parser.add_argument("--case_file", type=str, default=None, help="错误案例JSON文件路径，可选")
    parser.add_argument("--strategy", type=str, default=None, help="单帧选择策略，可选")
    # Storage strategy (pre-model frame selection)
    parser.add_argument("--storage_mode", type=str, default="all",
                        help="帧存储模式: all | rate | similarity | none")
    parser.add_argument("--storage_save_rate", type=float, default=1.0,
                        help="当storage_mode=rate时，保存比例")
    parser.add_argument("--storage_sim_threshold", type=float, default=0.99,
                        help="当storage_mode=similarity时，相似度阈值(越低保留越多)")
    # Frame filter (post-feature selection)
    parser.add_argument("--frame_filter_mode", type=str, default="none",
                        help="帧过滤模式: none | rate | similarity")
    parser.add_argument("--frame_filter_rate", type=float, default=1.0,
                        help="当frame_filter_mode=rate时，保留比例")
    parser.add_argument("--frame_filter_sim_threshold", type=float, default=0.99,
                        help="当frame_filter_mode=similarity时，相似度阈值")
    parser.add_argument("--convert_to_streaming", type=str, default='false', choices=['true', 'false', 'baseline'],
                        help="是否转换为流式模型 (true/false)")
    parser.add_argument("--prompt_builder_type", type=str, default=None,
                        help="Prompt builder type: default, streamingbench, compact, instruction_rich, numbered")
    parser.add_argument("--precomputed_retrieval_file", type=str, default=None,
                        help="预计算检索结果的CSV文件路径，如果提供则使用其中的retrieval indices避免重复检索")
    args = parser.parse_args()

    if not args.debug:
        logzero.loglevel(logging.INFO)
        warnings.filterwarnings('ignore')

    os.makedirs(args.save_dir, exist_ok=True)

    # fix random seed
    random.seed(2024)
    logger.info('seed: 2024')

    # Detect time-prompt mode (explicit flag or implied by QA class)
    time_prompt_mode = args.time_prompt or issubclass(QA_CLASS, TimePromptMixin)
    setattr(args, 'time_prompt', time_prompt_mode)

    # VideoQA model
    model_path = args.model_path if args.model_path else MODELS[args.model]['model_path']
    load_func = MODELS[args.model]['load_func']
    logger.info(f"Loading VideoQA model: {model_path}")
    load_kwargs = {
        'model_path': model_path,
        'n_local': args.n_local,
        'topk': args.retrieve_size,
        'chunk_size': args.retrieve_chunk_size,
        'use_hybrid_similarity': args.use_hybrid_similarity,
        'convert_to_streaming': args.convert_to_streaming,
    }
    if 'qwen2vl' in args.model:
        load_kwargs['time_prompt'] = time_prompt_mode
    videoqa_model, videoqa_processor = load_func(**load_kwargs)

    # Load ground truth file
    anno = json.load(open(args.anno_path))

    # 为支持short_memory_layers的类传递额外参数
    init_arg_names = []
    if hasattr(QA_CLASS, '__init__'):
        init_arg_names = list(QA_CLASS.__init__.__code__.co_varnames)
    base_kwargs = {
        'anno': anno,
        'sample_fps': args.sample_fps,
        'qa_model': videoqa_model,
        'qa_processor': videoqa_processor,
        'retrieve_size': args.retrieve_size,
        'chunk_size': args.retrieve_chunk_size,
        'num_chunks': args.num_chunks,
        'chunk_idx': args.chunk_idx,
        'save_dir': args.save_dir,
        'query_type': args.query_type,
        'use_dynamic_size': args.use_dynamic_size,
        'prompt_builder_type': getattr(args, 'prompt_builder_type', None),
        'prompt_builder_kwargs': {},
    }
    optional_map = {
        'short_memory_layers': args.short_memory_layers,
        'layer_weight_path': args.layer_weight_path,
        'head_weight_path': args.head_weight_path,
        'case_file': args.case_file,
        'strategy': args.strategy,
        'merge_load_kv': args.merge_load_kv,
        # Support time-prompt input rate forwarding when QA class accepts it
        'input_fps': getattr(args, 'input_fps', None),
        'time_prompt': time_prompt_mode,
        'entropy_threshold': getattr(args, 'entropy_threshold', None),
        'entropy_window_layers': getattr(args, 'entropy_window_layers', None),
        'precomputed_retrieval_file': getattr(args, 'precomputed_retrieval_file', None),
    }
    
    # Build prompt_builder_kwargs if prompt_builder_type is specified
    if getattr(args, 'prompt_builder_type', None) is not None:
        prompt_builder_kwargs = {}
        prompt_builder_kwargs['model'] = args.model
        base_kwargs['prompt_builder_kwargs'] = prompt_builder_kwargs
    
    for k, v in optional_map.items():
        # if k in init_arg_names:
        base_kwargs[k] = v
    retrieve_analyzer = QA_CLASS(**base_kwargs)
    # Expose streaming flag to analyzer for encode chunk-size logic
    try:
        setattr(retrieve_analyzer, 'convert_to_streaming', args.convert_to_streaming)
    except Exception:
        pass

    # Wire strategies
    try:
        retrieve_analyzer.storage_strategy = make_storage_strategy(
            mode=args.storage_mode,
            save_rate=args.storage_save_rate,
            sim_threshold=args.storage_sim_threshold,
        )
    except Exception as e:
        logger.warning(f"Failed to create storage strategy: {e}")


    # Attach frame filter to model if requested
    try:
        from tools.frame_policies import make_frame_filter
        frame_filter = make_frame_filter(
            mode=args.frame_filter_mode,
            rate=args.frame_filter_rate,
            sim_threshold=args.frame_filter_sim_threshold,
        )
        if frame_filter is not None:
            setattr(videoqa_model, 'frame_filter', frame_filter)
    except Exception as e:
        logger.warning(f"Failed to create frame filter: {e}")

    retrieve_analyzer.analyze(debug=args.debug)
