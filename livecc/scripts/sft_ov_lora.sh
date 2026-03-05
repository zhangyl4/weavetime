export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset MASTER_ADDR MASTER_PORT WORLD_SIZE RANK LOCAL_RANK NODE_RANK
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

learning_rate=5e-5 # config come from egogpt
run_name="llavaov_sft_timestamp_et_llava_lr$learning_rate"

# sleep 7200

TOKENIZERS_PARALLELISM=false torchrun --nnodes=1 --nproc_per_node=8 --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
   train_llava_ov_lora.py \
  --deepspeed /2024233235/videollm-online/configs/deepspeed/zero1.json \
  --output_dir outputs/$run_name \
  --overwrite_output_dir True \
  --run_name $run_name \
  --save_on_each_node True \
  --do_train True \
  --eval_strategy no \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate $learning_rate \
  --warmup_ratio 0.05 \
  --optim adamw_torch \
  --lr_scheduler_type cosine  \
  --num_train_epochs 0.5 \
  --logging_steps 5 \
  --save_steps 50 \
  --bf16 True \
  --tf32 True \
  --gradient_checkpointing True \
  --pretrained_model_name_or_path llava-hf/llava-onevision-qwen2-7b-ov-hf \
  --annotation_paths \
    /2024233235/videollm-online/EyeWO2/data/llava_video_178k_with_seeks_sample_valid.jsonl \
    /2024233235/videollm-online/EyeWO2/data/etbench_qwen2vl_timestamp.jsonl \
  --dataloader_num_workers 16 \
  --freeze_modules vision_tower \
  --use_liger_kernel True \
  --report_to tensorboard \
  --frames_upbound 64 \
  # --resume_from_checkpoint outputs/$run_name/checkpoint-150 \
  # --enable_streaming_sampling True \
  # --resume_from_checkpoint outputs/$run_name/checkpoint-400