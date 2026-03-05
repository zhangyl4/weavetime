# # conda create -n qwen25vl python==3.11
# # conda activate qwen25vl
pip install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install "transformers>=4.52.4" accelerate deepspeed peft opencv-python decord datasets tensorboard gradio pillow-heif gpustat timm sentencepiece openai av==12.0.0 qwen_vl_utils liger_kernel numpy==1.24.4 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install flash-attn --no-build-isolation -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install livecc-utils==0.0.2 -i https://pypi.tuna.tsinghua.edu.cn/simple

# PYTHONPATH=$(pwd) torchrun --standalone --nproc_per_node=8 evaluation/videomme/distributed_evaluate_videomme_stamp_lora.py --model_name_or_path Qwen/Qwen2-VL-7B-Instruct --lora_path outputs/qwen2vl_sft_timestamp_shuffle_llava_lr5e-5/checkpoint-498 --benchmark_path videomme_with_subtitles.jsonl