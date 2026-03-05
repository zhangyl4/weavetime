from typing import Optional, List, Tuple, Union
import torch
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import Cache
from transformers.utils import logging
from transformers.models.qwen2.modeling_qwen2 import repeat_kv
from model.attention.kv_cache_manager import ReKVDynamicCache

logger = logging.get_logger(__name__)



def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)



def apply_rotary_pos_emb_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_pos_ids: torch.Tensor,
    k_pos_ids: torch.Tensor,
    unsqueeze_dim: int = 1,
):
    cos_q = cos[q_pos_ids].unsqueeze(unsqueeze_dim)
    sin_q = sin[q_pos_ids].unsqueeze(unsqueeze_dim)
    cos_k = cos[k_pos_ids].unsqueeze(unsqueeze_dim)
    sin_k = sin[k_pos_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos_q) + (_rotate_half(q) * sin_q)
    k_embed = (k * cos_k) + (_rotate_half(k) * sin_k)
    return q_embed, k_embed


def Qwen2Model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    # Force cache when ReKV is enabled
    if getattr(self, 'rekv_config', None) is not None:
        use_cache = True

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
        )

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    use_legacy_cache = False
    # HACK: 
    # if use_cache and not isinstance(past_key_values, Cache) and not self.training:
    #     use_legacy_cache = True
    #     past_key_values = DynamicCache.from_legacy_cache(past_key_values)
    #     logger.warning_once(
    #         "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
    #         "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/internal/generation_utils#transformers.Cache)"
    #     )

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    # START HACK: Create ReKVDynamicCache on-demand
    if use_cache and past_key_values is None:
        rekv_cfg = getattr(self, 'rekv_config', None)
        dyn_block_size = getattr(self, 'dynamic_block_size', rekv_cfg.get('block_size', 0))
        dyn_exc_block_size = getattr(self, 'dynamic_exc_block_size', rekv_cfg.get('exc_block_size', 0))
        rekv_cfg['block_size'] = dyn_block_size
        rekv_cfg['exc_block_size'] = dyn_exc_block_size
        try:
            past_key_values = ReKVDynamicCache(rekv_cfg, num_layers=len(self.layers))
        except Exception as e:
            logger.warning(f"Failed to create ReKVDynamicCache, fallback to None. Error: {e}")
    # end HACK --------------------------------
    
    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)
    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )

    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = past_key_values if use_cache else None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            # Be robust to layer output tuple structure; take the last element as cache
            if isinstance(layer_outputs, (tuple, list)) and len(layer_outputs) >= 2:
                next_decoder_cache = layer_outputs[-1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = None
    if use_cache:
        next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )
    
    
# Try to import private flash attention helper; fallback handled at runtime
try:
    from transformers.utils import is_flash_attn_2_available
    if is_flash_attn_2_available():
        from transformers.modeling_flash_attention_utils import _flash_attention_forward
except Exception:  # pragma: no cover - version-dependent
    _flash_attention_forward = None  # type: ignore[assignment]



def Qwen2FlashAttention2_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
):
    # Fallback to SDPA if the private helper is not available in this transformers version
    if _flash_attention_forward is None:
        logger.warning_once("_flash_attention_forward not found in transformers.qwen2; falling back to SDPA attention.")
        return Qwen2SdpaAttention_forward(
            self,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    #START HACK --------------------------------

    # Update cache first to get final K/V (may include retrieval blocks) then compute RoPE
    if past_key_value is not None:
        cache_kwargs = {"cache_position": cache_position, "query_states": query_states}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
    
    past_seen_tokens = key_states.shape[-2] - q_len
    position_ids = torch.arange(past_seen_tokens, past_seen_tokens + q_len, device=key_states.device).unsqueeze(0).expand(bsz, -1)
    kv_seq_len = key_states.shape[-2]
    rotary_seq_len = (
        max(kv_seq_len, position_ids[:, -1].max().item() + 1) if position_ids is not None else kv_seq_len
    )
    cos, sin = self.rotary_emb(value_states, seq_len=rotary_seq_len)
    if position_ids is not None:
        kv_position_ids = torch.arange(kv_seq_len, device=position_ids.device).unsqueeze(0).expand(bsz, -1)
        query_states, key_states = apply_rotary_pos_emb_qk(
            query_states, key_states, cos, sin, position_ids, kv_position_ids
        )
    else:
        kv_position_ids = torch.arange(kv_seq_len, device=key_states.device).unsqueeze(0).expand(bsz, -1)
        query_states, key_states = apply_rotary_pos_emb_qk(
            query_states, key_states, cos, sin, kv_position_ids, kv_position_ids
        )
    # END HACK --------------------------------
    
    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    dropout_rate = 0.0 if not self.training else self.attention_dropout

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in float16 just to be sure everything works as expected.
    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    # Reashape to the expected shape for Flash Attention
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    if (
        self.config.use_sliding_window
        and getattr(self.config, "sliding_window", None) is not None
        and self.layer_idx >= self.config.max_window_layers
    ):
        sliding_window = self.config.sliding_window
    else:
        sliding_window = None

    attn_output = _flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        q_len,
        position_ids=position_ids,
        dropout=dropout_rate,
        sliding_window=sliding_window,
        is_causal=self.is_causal,
        use_top_left_mask=self._flash_attn_uses_top_left_mask,
    )

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


# Adapted from Qwen2Attention.forward

def Qwen2SdpaAttention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
        logger.warning_once(
            "Qwen2Model is using Qwen2SdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
            'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        )
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
    bsz, q_len, _ = hidden_states.size()

    # if q_len != 13:
    #     breakpoint()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    # START HACK --------------------------------
    # Update cache first to get final K/V (may include retrieval blocks) then compute RoPE
    
    if past_key_value is not None:
        cache_kwargs = {"cache_position": cache_position, "query_states": query_states}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
    
    past_seen_tokens = key_states.shape[-2] - q_len
    position_ids = torch.arange(past_seen_tokens, past_seen_tokens + q_len, device=key_states.device).unsqueeze(0).expand(bsz, -1)
    kv_seq_len = key_states.shape[-2]
    rotary_seq_len = (
        max(kv_seq_len, position_ids[:, -1].max().item() + 1) if position_ids is not None else kv_seq_len
    )
    cos, sin = self.rotary_emb(value_states, seq_len=rotary_seq_len)
    if position_ids is not None:
        kv_position_ids = torch.arange(kv_seq_len, device=position_ids.device).unsqueeze(0).expand(bsz, -1)
        query_states, key_states = apply_rotary_pos_emb_qk(
            query_states, key_states, cos, sin, position_ids, kv_position_ids
        )
    else:
        kv_position_ids = torch.arange(kv_seq_len, device=key_states.device).unsqueeze(0).expand(bsz, -1)
        query_states, key_states = apply_rotary_pos_emb_qk(
            query_states, key_states, cos, sin, kv_position_ids, kv_position_ids
        )
    # END HACK --------------------------------
    
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "cuda" and attention_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
    is_causal = True if causal_mask is None and q_len > 1 else False

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=is_causal,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value