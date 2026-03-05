from types import MethodType
from transformers import LlavaOnevisionForConditionalGeneration
import torch
import os, torchvision
import re

from model.llava_ov.model_forward import Qwen2Model_forward, Qwen2FlashAttention2_forward, Qwen2SdpaAttention_forward


def convert_llavaov_to_streaming(model: LlavaOnevisionForConditionalGeneration):
    # Attach forward overrides
    model.language_model.model.forward = MethodType(Qwen2Model_forward, model.language_model.model)
    for layer in model.language_model.model.layers:
        if layer.self_attn.__class__.__name__ == 'Qwen2SdpaAttention':
            layer.self_attn.forward = MethodType(Qwen2SdpaAttention_forward, layer.self_attn)
        else:
            layer.self_attn.forward = MethodType(Qwen2FlashAttention2_forward, layer.self_attn)
    return model