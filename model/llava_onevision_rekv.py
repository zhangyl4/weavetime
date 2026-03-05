import torch
from transformers import LlavaOnevisionProcessor, LlavaOnevisionForConditionalGeneration
from logzero import logger

from model.patch import patch_hf
from model.abstract_rekv import Abstract_ReKV
from model.llava_ov.patch_model import convert_llavaov_to_streaming


class LlavaOneVision_ReKV(LlavaOnevisionForConditionalGeneration, Abstract_ReKV):
    def __init__(self, config, processor, n_frame_tokens, init_prompt_ids, n_local, topk, chunk_size):
        LlavaOnevisionForConditionalGeneration.__init__(self, config)
        Abstract_ReKV.__init__(self, processor, n_frame_tokens, init_prompt_ids, n_local, topk, chunk_size)

    def get_prompt(self, query, mc=False):
        prompt =  f"\n{query}<|im_end|><|im_start|>assistant\n"
        if mc:
            prompt += 'Best option: ('
        return prompt

    def _get_video_features(self, pixel_values_videos):
        batch_size, frames, channels, height, width = pixel_values_videos.shape
        pixel_values_videos = pixel_values_videos.view(batch_size * frames, channels, height, width)
        video_features = self.vision_tower(pixel_values_videos, output_hidden_states=True)
        selected_video_feature = video_features.hidden_states[self.config.vision_feature_layer]

        if self.config.vision_feature_select_strategy == "default":
            selected_video_feature = selected_video_feature[:, 1:]
        elif self.config.vision_feature_select_strategy == "full":
            selected_video_feature = selected_video_feature
        video_features = self.multi_modal_projector(selected_video_feature)

        video_features = self.apply_pooling(video_features)
        video_features = video_features.reshape(batch_size, frames * video_features.shape[1], -1)  # (B, Nv*196, D)
        return video_features

    @torch.inference_mode()
    def question_answering(self, input_text, max_new_tokens=128, retrieved_indices=None):
        device = self.device
        stop_token_ids = [self.processor.tokenizer.eos_token_id]

        output_ids = []
        stopped = False

        # NOTE: Only input the question to perform retrieval.
        input_ids = self.processor.tokenizer(input_text['question']).input_ids
        input_ids = torch.as_tensor([input_ids], device=device)
        for layer_kv in self.kv_cache:  # activate retrieval mode
            layer_kv.set_retrieval()

        if retrieved_indices is None:  # Internal retrieval
            out = self.language_model(input_ids=input_ids, use_cache=True, past_key_values=self.kv_cache)
            past_key_values = out.past_key_values  # Retrieved KV-Cache: L x 2 x (B, h, N, Dh)
        else:  # External retrieval
            for layer_kv in self.kv_cache:
                assert layer_kv.block_size == self.n_frame_tokens, f'block_size: {layer_kv.block_size}, n_frame_tokens: {self.n_frame_tokens}'
                layer_kv.set_retrieved_block_indices(retrieved_indices)
            out = self.language_model(input_ids=input_ids, use_cache=True, past_key_values=self.kv_cache)
            past_key_values = out.past_key_values  # Retrieved KV-Cache: L x 2 x (B, h, N, Dh)

        for layer_kv in self.kv_cache:  # reset to default
            layer_kv.reset_retrieval()

        for i in range(max_new_tokens):
            if i == 0:  # prefill
                input_ids = self.processor.tokenizer(input_text['prompt']).input_ids
                input_ids = torch.as_tensor([input_ids], device=device)
                inputs_embeds = self.get_input_embeddings()(input_ids)
                out = self.language_model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=past_key_values)
                past_key_values = out.past_key_values
                logits = out.logits
            else:  # decoding
                out = self.language_model(
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

        output = self.processor.tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )
        
        return output
    
    def simple_forward(self, video, prompt):
        user_content = [
            {'type': "video"},
            {"type": "text", "text": prompt}
        ]
        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a helpful assistant."},
                ],
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

        prompt_text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = self.processor(
            videos=[video],
            text=prompt_text,
            return_tensors="pt",
        ).to(0, torch.float16)

        output = self.generate(**inputs, max_new_tokens=1024, do_sample=False)
        # response = self.processor.decode(output[0][2:], skip_special_tokens=True) 
        # breakpoint()
        # delete input_ids in output_ids
        output_ids = output[0]
        input_ids = inputs['input_ids']
        output_ids = output_ids[input_ids.shape[1]:]
        response = self.processor.decode(output_ids, skip_special_tokens=True)
        return response


def load_model(model_path='model_zoo/LLaVA/llava-onevision-qwen2-7b-ov-hf',
               n_init=None, n_local=None, topk=64, chunk_size=1, use_hybrid_similarity=True, convert_to_streaming=False):
    device = 'cuda'
    n_frame_tokens = 196

    # Detect checkpoints and LoRA adapters (parity with qwen2vl_rekv.py)
    import os
    import glob

    original_model_path = model_path
    processor_path = model_path
    is_lora_model = False
    base_model_path = None

    if os.path.isdir(model_path):
        checkpoint_dirs = glob.glob(os.path.join(model_path, 'checkpoint-*'))
        if checkpoint_dirs:
            latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split('-')[-1]))
            print(f"检测到训练输出目录，使用最新检查点: {latest_checkpoint}")

            adapter_config_path = os.path.join(latest_checkpoint, 'adapter_config.json')
            if os.path.exists(adapter_config_path):
                import json
                with open(adapter_config_path, 'r') as f:
                    adapter_config = json.load(f)
                base_model_path = adapter_config.get('base_model_name_or_path', original_model_path)
                is_lora_model = True
                model_path = latest_checkpoint
                processor_path = base_model_path
                print(f"检测到LoRA模型，基础模型: {base_model_path}")
            elif os.path.exists(os.path.join(latest_checkpoint, 'preprocessor_config.json')):
                processor_path = latest_checkpoint
                model_path = latest_checkpoint
                print(f"使用检查点目录: {latest_checkpoint}")
            else:
                print("检查点目录缺少processor配置，使用默认基础模型")
                processor_path = original_model_path
                model_path = original_model_path

    # Load processor with fallback
    try:
        processor = LlavaOnevisionProcessor.from_pretrained(processor_path)
        print(f"成功从 {processor_path} 加载processor")
    except Exception as e:
        print(f"从 {processor_path} 加载processor失败: {e}")
        processor_path = original_model_path
        processor = LlavaOnevisionProcessor.from_pretrained(processor_path)
        print(f"降级使用默认基础模型加载processor: {processor_path}")

    init_prompt = '<|im_start|>system \nYou are a helpful assistant.<|im_end|><|im_start|>user '
    init_prompt_ids = processor.tokenizer(init_prompt, return_tensors="pt").input_ids.to(device)
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

    # Load model (base or LoRA)
    if is_lora_model:
        print(f"加载LoRA模型，基础模型: {base_model_path}, 适配器: {model_path}")
        model = LlavaOneVision_ReKV.from_pretrained(
            base_model_path,
            device_map="auto",
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            processor=processor,
            n_frame_tokens=n_frame_tokens,
            init_prompt_ids=init_prompt_ids,
            n_local=n_local,
            topk=topk,
            chunk_size=chunk_size,
        )
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, model_path)
            # Merge LoRA weights to reduce runtime memory and remove adapter modules
            try:
                model = model.merge_and_unload()
                print(f"成功合并LoRA权重并卸载适配器: {model_path}")
            except Exception:
                print(f"合并LoRA权重失败，保留适配器在线: {model_path}")
            print(f"成功加载LoRA适配器: {model_path}")
        except Exception as e:
            print(f"加载LoRA适配器失败: {e}")
            print("继续使用基础模型...")
    else:
        model = LlavaOneVision_ReKV.from_pretrained(
            model_path,
            device_map="auto",
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            processor=processor,
            n_frame_tokens=n_frame_tokens,
            init_prompt_ids=init_prompt_ids,
            n_local=n_local,
            topk=topk,
            chunk_size=chunk_size,
            # attn_implementation='flash_attention_2'
        )
    # Patch language model with ReKV attention
    if convert_to_streaming == 'true':
        model = convert_llavaov_to_streaming(model)
    elif convert_to_streaming == 'baseline':
        pass
    else:
        # pass
        model.language_model = patch_hf(model.language_model, **inf_llm_config)
    # Expose ReKV config to the inner language model for cache construction
    try:
        if hasattr(model, 'language_model') and hasattr(model.language_model, 'model'):
            model.language_model.model.rekv_config = inf_llm_config
    except Exception:
        logger.warning("Failed to attach rekv_config to language model; dynamic cache may use defaults.")
    
    for k, v in inf_llm_config.items():
        logger.info(f'{k}: {v}')
    logger.info(f'n_frame_tokens: {n_frame_tokens}')

    model.eval()

    return model, processor
