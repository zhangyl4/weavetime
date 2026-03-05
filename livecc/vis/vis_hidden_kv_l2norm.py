import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from livecc_utils import _read_video_decord_plus, _spatial_resize_video
from qwen_vl_utils.vision_process import FPS


def l2_norm(x, axis=-1):
    return np.linalg.norm(x, ord=2, axis=axis)


def plot_l2norm(hidden_norm, k_norm, v_norm, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    # hidden_norm, k_norm, v_norm: [num_layers, seq_len]
    for ax, norm, name in zip(axes, [hidden_norm, k_norm, v_norm], [r'$h$', r'$k$', r'$v$']):
        # 区分第一个token和其余token
        ax.plot(norm[:, 0], label=f'{name}_1', marker='h')
        if norm.shape[1] > 1:
            ax.plot(norm[:, 1:].mean(axis=1), label=fr'{name}_{{t\neq1}}', marker='o')
        ax.set_xlabel('Layer')
        ax.set_ylabel(r'$\ell_2$-norm')
        ax.legend()
        ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_l2norm_tokenwise_split(hidden_norm, k_norm, v_norm, save_path):
    # hidden_norm: (num_layers+1, seq_len)
    # k_norm, v_norm: (num_layers, seq_len)
    hidden_mean = hidden_norm.mean(axis=0)  # (seq_len,)
    k_mean = k_norm.mean(axis=0)
    v_mean = v_norm.mean(axis=0)
    hidden_std = hidden_norm.std(axis=0)
    k_std = k_norm.std(axis=0)
    v_std = v_norm.std(axis=0)
    hidden_min = hidden_norm.min(axis=0)
    k_min = k_norm.min(axis=0)
    v_min = v_norm.min(axis=0)
    hidden_max = hidden_norm.max(axis=0)
    k_max = k_norm.max(axis=0)
    v_max = v_norm.max(axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    # Mean with std fill
    axes[0].plot(hidden_mean, marker='h', label='mean')
    axes[0].fill_between(np.arange(len(hidden_mean)), hidden_mean-hidden_std, hidden_mean+hidden_std, alpha=0.2, label='std')
    axes[0].plot(hidden_min, linestyle='--', color='gray', alpha=0.5, label='min')
    axes[0].plot(hidden_max, linestyle='--', color='red', alpha=0.5, label='max')
    axes[0].set_title(r'$h$')
    axes[1].plot(k_mean, marker='o', label='mean')
    axes[1].fill_between(np.arange(len(k_mean)), k_mean-k_std, k_mean+k_std, alpha=0.2, label='std')
    axes[1].plot(k_min, linestyle='--', color='gray', alpha=0.5, label='min')
    axes[1].plot(k_max, linestyle='--', color='red', alpha=0.5, label='max')
    axes[1].set_title(r'$k$')
    axes[2].plot(v_mean, marker='s', label='mean')
    axes[2].fill_between(np.arange(len(v_mean)), v_mean-v_std, v_mean+v_std, alpha=0.2, label='std')
    axes[2].plot(v_min, linestyle='--', color='gray', alpha=0.5, label='min')
    axes[2].plot(v_max, linestyle='--', color='red', alpha=0.5, label='max')
    axes[2].set_title(r'$v$')
    for ax in axes:
        ax.set_xlabel('Token')
        ax.set_ylabel(r'$\ell_2$-norm')
        ax.grid(True, alpha=0.4)
        ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default="chenjoya/LiveCC-7B-Instruct")
    parser.add_argument('--video-path', type=str, default='/2022233235/videollm-online/livecc/demo/sources/howto_fix_laptop_mute_1080p.mp4')
    parser.add_argument('--prompt', type=str, default='describe the two time periods of this video respectively.')
    parser.add_argument('--output-path', type=str, default='./output')
    parser.add_argument('--device', type=str, default='cuda:1')
    args = parser.parse_args()
    streaming_fps_frames = int(FPS)

    # 1. load model and processor
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype="auto", attn_implementation='flash_attention_2', device_map=args.device)
    processor = AutoProcessor.from_pretrained(args.model_path, padding_side='left')

    # 2. load video and build conversation
    video, sample_fps, clip_pts = _read_video_decord_plus({'video': args.video_path, 'remote_loader': None}, return_pts=True, strict_fps=True)
    video = _spatial_resize_video(video)
    video_inputs = []
    conversation = [{ 'role': 'user', 'content': [] }]
    for i in range(0, len(video), streaming_fps_frames):
        start_timestamp, end_timestamp = i / FPS, (i + streaming_fps_frames) / FPS
        conversation[-1]['content'].append({'type': 'text', 'text': f'Time={start_timestamp:.1f}-{end_timestamp:.1f}s'})
        conversation[-1]['content'].append({'type': 'video', 'video': video[i:i+streaming_fps_frames]})
        video_inputs.append(video[i:i+streaming_fps_frames])
    conversation[-1]['content'].append({'type': 'text', 'text': args.prompt})

    # 3. prepare inputs
    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=text,
        images=None,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(args.device) for k, v in inputs.items()}

    # 4. forward pass to get hidden_states and past_key_values
    model.eval()
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            # output_attentions=True,
            use_cache=True,
        )
    # hidden_states: tuple(num_layers+1) each (bsz, seq_len, hidden_dim)
    # past_key_values: tuple(num_layers) each (k, v, ...)
    hidden_states = outputs.hidden_states  # (num_layers+1, bsz, seq_len, hidden_dim)
    past_key_values = outputs.past_key_values  # (num_layers, 2, bsz, num_heads, seq_len, head_dim)

    # 5. 计算L2-norm
    # 只取batch=0
    hidden_norm = np.stack([
        l2_norm(h[0].float().cpu().numpy(), axis=-1) for h in hidden_states  # (seq_len,)
    ])  # (num_layers+1, seq_len)

    k_norm = []
    v_norm = []
    for pkv in past_key_values:
        # pkv: (k, v, ...)  k/v: (bsz, num_heads, seq_len, head_dim)
        k = pkv[0][0].float()  # (num_heads, seq_len, head_dim)
        v = pkv[1][0].float()
        # 合并head维度
        k = k.permute(1, 0, 2).reshape(k.shape[1], -1)  # (seq_len, num_heads*head_dim)
        v = v.permute(1, 0, 2).reshape(v.shape[1], -1)
        k_norm.append(l2_norm(k.cpu().numpy(), axis=-1))  # (seq_len,)
        v_norm.append(l2_norm(v.cpu().numpy(), axis=-1))
    k_norm = np.stack(k_norm)  # (num_layers, seq_len)
    v_norm = np.stack(v_norm)

    # 6. 绘图
    os.makedirs(args.output_path, exist_ok=True)
    save_path = os.path.join(args.output_path, f'{args.model_path.split("/")[-1]}_l2norm_hidden_kv.png')
    plot_l2norm(hidden_norm, k_norm, v_norm, save_path)
    print(f"L2-norm plot saved to {save_path}")

    # 7. token维度分图可视化
    save_path_token_split = os.path.join(args.output_path, f'{args.model_path.split("/")[-1]}_l2norm_tokenwise_split.png')
    plot_l2norm_tokenwise_split(hidden_norm, k_norm, v_norm, save_path_token_split)
    print(f"Tokenwise split L2-norm plot saved to {save_path_token_split}")

if __name__ == '__main__':
    main() 