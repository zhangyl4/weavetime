export VIDEO_MIN_PIXELS=78400 # 100*28*28. the minimum visual frame tokens sent to llm is 100
export FPS_MAX_FRAMES=768 # maximum number of frames for each video (480/60/2 = 4min)
export VIDEO_MAX_PIXELS=4816896 # 19267584 = 24576*28*28. 4816896 the maximum overall video tokens sent to llm is 24k (leave 8k for language)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

learning_rate=1e-5 # sft uses 2e-5 lr
run_name="qwen2vl_sft_24k768x100_llava_lora_lr$learning_rate"

TOKENIZERS_PARALLELISM=false torchrun --standalone --nproc_per_node=8 train_lora.py \
  --deepspeed /2022233235/videollm-online/configs/deepspeed/zero1.json \
  --output_dir outputs/$run_name \
  --overwrite_output_dir True \
  --run_name $run_name \
  --save_on_each_node True \
  --do_train True \
  --eval_strategy no \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate $learning_rate \
  --warmup_ratio 0.03 \
  --optim adamw_torch \
  --lr_scheduler_type cosine \
  --num_train_epochs 0.25 \
  --logging_steps 5 \
  --save_steps 100 \
  --bf16 True \
  --tf32 True \
  --gradient_checkpointing True \
  --pretrained_model_name_or_path Qwen/Qwen2-VL-7B-Instruct \
  --annotation_paths \
        '/2022233235/videollm-online/EyeWO2/data/llava_video_178k_with_seeks_sample_valid.jsonl' \
  --dataloader_num_workers 16 \
  --freeze_modules visual \
  --use_liger_kernel True \
  --report_to tensorboard \
#   --resume_from_checkpoint outputs/$run_name/checkpoint-800

      #   '/2022233235/videollm-online/EyeWO2/data/llava_video_178k_with_seeks_sample_valid.jsonl' \
            #   '/2022233235/videollm-online/EyeWO2/data/cof_qwen2vl.jsonl' \
# learning_rate=1e-5 # sft uses 2e-5 lr
# run_name="qwen2vl_sft_24k768x100_llavaetbench_lora_lr$learning_rate"

# TOKENIZERS_PARALLELISM=false torchrun --standalone --nproc_per_node=8 train_lora.py \
#   --deepspeed /2022233235/videollm-online/configs/deepspeed/zero1.json \
#   --output_dir outputs/$run_name \
#   --overwrite_output_dir True \
#   --run_name $run_name \
#   --save_on_each_node True \
#   --do_train True \
#   --eval_strategy no \
#   --per_device_train_batch_size 1 \
#   --gradient_accumulation_steps 8 \
#   --learning_rate $learning_rate \
#   --warmup_ratio 0.03 \
#   --optim adamw_torch \
#   --lr_scheduler_type cosine \
#   --num_train_epochs 0.25 \
#   --logging_steps 5 \
#   --save_steps 100 \
#   --bf16 True \
#   --tf32 True \
#   --gradient_checkpointing True \
#   --pretrained_model_name_or_path Qwen/Qwen2-VL-7B-Instruct \
#   --annotation_paths \
#         '/2022233235/videollm-online/EyeWO2/data/etbench_qwen2vl.jsonl' \
#         '/2022233235/videollm-online/EyeWO2/data/llava_video_178k_with_seeks_sample_valid.jsonl' \
#   --dataloader_num_workers 16 \
#   --freeze_modules visual \
#   --use_liger_kernel True \
#   --report_to tensorboard \