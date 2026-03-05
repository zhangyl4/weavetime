from types import MethodType
from transformers import Qwen2VLForConditionalGeneration

# from model.qwen2vl.model_forward import (
#     Qwen2VLModel_forward,
#     Qwen2VLFlashAttention2_forward,
#     Qwen2VLSdpaAttention_forward,
#     Qwen2VLForConditionalGeneration_prepare_inputs_for_generation
# )

from model.qwen2vl.model_forward_rope import (
    Qwen2VLModel_forward,
    Qwen2VLFlashAttention2_forward,
    Qwen2VLSdpaAttention_forward,
    Qwen2VLForConditionalGeneration_prepare_inputs_for_generation
)


def convert_qwen2vl_to_streaming(model: Qwen2VLForConditionalGeneration):
    # Attach forward overrides for decoder model
    model.model.forward = MethodType(Qwen2VLModel_forward, model.model)
    # Bind prepare_inputs_for_generation to preserve mRoPE get_rope_index behavior
    model.prepare_inputs_for_generation = MethodType(
        Qwen2VLForConditionalGeneration_prepare_inputs_for_generation, model
    )
    for layer in model.model.layers:
        if layer.self_attn.__class__.__name__ == 'Qwen2VLSdpaAttention':
            layer.self_attn.forward = MethodType(Qwen2VLSdpaAttention_forward, layer.self_attn)
        else:
            layer.self_attn.forward = MethodType(Qwen2VLFlashAttention2_forward, layer.self_attn)
    return model